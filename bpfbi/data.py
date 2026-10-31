"""Chronological, origin-causal interpolation and exact window-level event injection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interpolate_short(values, maximum=3):
    """Interpolate only complete bounded gaps of at most maximum entries."""
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    edges = np.flatnonzero(np.diff(np.r_[False, missing, False]))
    for left, right in edges.reshape(-1, 2):
        if left > 0 and right < len(result) and right-left <= maximum:
            result[left:right] = np.linspace(result[left-1], result[right], right-left+2)[1:-1]
    return result


def prepare(args):
    table = pd.read_csv(args.csv, parse_dates=['timestamp'])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    clients = []
    for number, (client, frame) in enumerate(table.groupby('client', sort=True)):
        frame = frame.sort_values('timestamp').set_index('timestamp')
        if frame.index.has_duplicates:
            raise ValueError(f'Duplicate timestamp for {client}')
        frame = frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq='h'))
        raw = frame['load'].to_numpy(float)
        if np.isfinite(raw).mean() < args.minimum_completeness: continue
        bounds = (0,int(.6*len(frame)),int(.8*len(frame)),len(frame))
        mean = float(np.nanmean(raw[:bounds[1]])); scale = max(float(np.nanstd(raw[:bounds[1]])),1e-6)
        weather_mean = frame[args.weather].iloc[:bounds[1]].mean().to_numpy() if args.weather else np.empty(0)
        weather_scale = frame[args.weather].iloc[:bounds[1]].std().clip(lower=1e-6).to_numpy() if args.weather else np.empty(0)
        arrays = {}
        for part,left,right in zip(('train','val','test'),bounds[:-1],bounds[1:]):
            split = frame.iloc[left:right]; clean = split['load'].to_numpy(float); index = split.index
            calendar = np.stack([np.sin(2*np.pi*index.hour/24),np.cos(2*np.pi*index.hour/24),
                np.sin(2*np.pi*index.dayofweek/7),np.cos(2*np.pi*index.dayofweek/7),
                np.sin(2*np.pi*index.month/12),np.cos(2*np.pi*index.month/12)],1)
            weather = (split[args.weather].to_numpy(float)-weather_mean)/weather_scale if args.weather else np.empty((len(split),0))
            holiday = split['holiday'].fillna(0).to_numpy(float)[:,None] if 'holiday' in split else np.zeros((len(split),1))
            base = np.column_stack([np.nan_to_num((clean-mean)/scale),np.isfinite(clean),calendar,holiday,np.nan_to_num(weather)]).astype(np.float32)
            origins = np.arange(args.window-1,len(split)-args.horizon)
            origins = origins[np.isfinite(clean[origins+args.horizon])]
            if not len(origins): raise ValueError(f'No {part} windows for {client}')
            rng = np.random.default_rng(args.seed+number*1009+left)
            chosen = set(rng.choice(len(origins),round(args.anomaly_fraction*len(origins)),replace=False).tolist())
            windows,observed,labels,ledger = [],[],[],[]
            for row,origin in enumerate(origins):
                start = origin-args.window+1; target = origin+args.horizon
                # Only values at or before this forecast origin may fill input gaps.
                source = interpolate_short(clean[start:origin+1])
                target_value = clean[target]
                features = base[start:origin+1].copy()
                kind,begin,end = -1,-1,-1
                if row in chosen:
                    kind = int(rng.integers(4)); duration = (3,12,6,4)[kind]
                    # Exactly one event per selected window. Start is within the input.
                    begin = int(rng.integers(1,max(2,args.window))) if args.window>1 else 0
                    end = min(args.window+args.horizon,begin+duration)
                    segment = source[begin:min(end,args.window)]
                    if kind==0:
                        source[begin:min(end,args.window)] = segment+3*scale
                        if end>args.window+args.horizon-1:target_value+=3*scale
                    elif kind==1:
                        source[begin:min(end,args.window)] = segment+.25*scale*np.arange(1,len(segment)+1)
                        if end>args.window+args.horizon-1:target_value+=.25*scale*(args.window+args.horizon-begin)
                    elif kind==2:
                        previous = source[:begin][np.isfinite(source[:begin])]
                        frozen = previous[-1] if len(previous) else mean
                        source[begin:min(end,args.window)] = frozen
                        if end>args.window+args.horizon-1:target_value=frozen
                    else:
                        source[begin:min(end,args.window)] = np.nan
                        if end>args.window+args.horizon-1:target_value=np.nan
                features[:,0] = np.nan_to_num((source-mean)/scale)
                features[:,1] = np.isfinite(source)
                for weather_index in range(len(args.weather)):
                    features[:,9+weather_index] = np.nan_to_num(interpolate_short(weather[start:origin+1,weather_index]))
                windows.append(features); observed.append((target_value-mean)/scale); labels.append(float(row in chosen))
                ledger.append((kind,begin,end))
            arrays.update({f'{part}_features':base,f'{part}_window_features':np.stack(windows),
                f'{part}_origin':origins,f'{part}_y':((clean[origins+args.horizon]-mean)/scale).astype(np.float32),
                f'{part}_observed':np.asarray(observed,np.float32),f'{part}_anomaly':np.asarray(labels,np.float32),
                f'{part}_events':np.asarray(ledger,np.int64),f'{part}_time':index[origins+args.horizon].asi8})
        name=f'client_{number:03d}.npz'
        np.savez_compressed(output/name,**arrays,mean=mean,scale=scale,window=args.window,horizon=args.horizon)
        clients.append(dict(client=str(client),file=name))
    if not clients:raise ValueError('No eligible clients')
    (output/'clients.json').write_text(json.dumps(clients,indent=2),encoding='utf-8')


class Windows:
    def __init__(self,data,part):
        self.features=data[f'{part}_features']; self.origins=data[f'{part}_origin']
        self.windows=data.get(f'{part}_window_features'); self.y=data[f'{part}_y']
        self.anomaly=data[f'{part}_anomaly']; self.window=int(data['window'])
    def __len__(self):return len(self.y)
    def __getitem__(self,index):
        end=int(self.origins[index])+1
        x=self.windows[index] if self.windows is not None else self.features[end-self.window:end]
        return x,self.y[index],self.anomaly[index]
