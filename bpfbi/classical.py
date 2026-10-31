"""Seasonal-naive and actual XGBoost forecasts on the common building protocol."""
import argparse,json,time
from pathlib import Path

import joblib
import numpy as np

from .data import Windows
from .cli import evaluate


def windows(data,part):
    dataset=Windows(data,part)
    return np.stack([dataset[i][0] for i in range(len(dataset))])


class Classical:
    def __init__(self,method,period=24,trees=100,seed=11):
        self.method,self.period,self.trees,self.seed=method,period,trees,seed
    def fit(self,x,y,anomaly,horizon=1):
        if self.method=='seasonal_naive':
            if x.shape[1]<self.period:raise ValueError('Seasonal period must fit the input window')
            point=x[:,x.shape[1]-self.period+horizon-1,0]
            self.offset=np.quantile(y-point,[.05,.95])
            self.anomaly_reference=np.sort(self.anomaly_statistic(x))
        else:
            from xgboost import XGBRegressor,XGBClassifier
            flat=x.reshape(len(x),-1)
            settings=dict(n_estimators=self.trees,max_depth=3,learning_rate=.05,n_jobs=1,random_state=self.seed,tree_method='hist')
            self.regressors=[XGBRegressor(objective='reg:quantileerror',quantile_alpha=q,**settings).fit(flat,y) for q in (.5,.05,.95)]
            if len(np.unique(anomaly))==2:self.classifier=XGBClassifier(objective='binary:logistic',**settings).fit(flat,anomaly)
            else:self.classifier=float(np.mean(anomaly))
        return self
    def anomaly_statistic(self,x):
        if x.shape[1]<=self.period:return abs(x[:,:,0]-np.median(x[:,:,0],axis=1,keepdims=True)).max(1)
        return abs(x[:,self.period:,0]-x[:,:-self.period,0]).max(1)
    def predict(self,x,horizon):
        if self.method=='seasonal_naive':
            index=x.shape[1]-self.period+horizon-1
            if not 0<=index<x.shape[1]:raise ValueError('Horizon must be no longer than seasonal period')
            median=x[:,index,0];low=median+self.offset[0];high=median+self.offset[1]
            p=np.searchsorted(self.anomaly_reference,self.anomaly_statistic(x),side='right')/(len(self.anomaly_reference)+1)
        else:
            flat=x.reshape(len(x),-1);median,low,high=[model.predict(flat) for model in self.regressors]
            low=np.minimum(low,median);high=np.maximum(high,median)
            p=self.classifier.predict_proba(flat)[:,1] if not isinstance(self.classifier,float) else np.full(len(x),self.classifier)
        p=np.clip(p,1e-6,1-1e-6)
        return np.column_stack([median,low,high,np.log(p/(1-p))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--method',choices=['seasonal_naive','xgboost'],required=True)
    parser.add_argument('--checkpoint-dir');parser.add_argument('--period',type=int,default=24)
    parser.add_argument('--trees',type=int,default=100);parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args();root=Path(args.data);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    metrics=[];started=time.perf_counter()
    for index,row in enumerate(json.loads((root/'clients.json').read_text())):
        with np.load(root/row['file'],allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
        if args.checkpoint_dir:
            state=joblib.load(Path(args.checkpoint_dir)/f'model_{index:03d}.joblib')
            if state['client']!=row['client'] or state['horizon']!=int(data['horizon']):raise ValueError('Checkpoint client/horizon differs')
            model=Classical(args.method);model.__dict__.update(state['model_state']);calibration=state['calibration']
        else:
            model=Classical(args.method,args.period,args.trees,args.seed)
            model.fit(windows(data,'train'),data['train_y'],data['train_anomaly'],int(data['horizon']))
            val=model.predict(windows(data,'val'),int(data['horizon']))
            scores=np.maximum.reduce([val[:,1]-data['val_y'],data['val_y']-val[:,2],np.zeros(len(val))])
            rank=min(len(scores),int(np.ceil((len(scores)+1)*.9)))
            calibration=float(np.sort(scores)[rank-1]);joblib.dump(dict(model_state=model.__dict__,client=row['client'],horizon=int(data['horizon']),calibration=calibration),output/f'model_{index:03d}.joblib')
        prediction=model.predict(windows(data,'test'),int(data['horizon']))
        result,arrays=evaluate(prediction,data,calibration);result['client']=row['client'];metrics.append(result)
        np.savez_compressed(output/f'prediction_{index:03d}.npz',**arrays)
    nrmse=np.array([row['nrmse'] for row in metrics]);rng=np.random.default_rng(args.seed)
    boot=rng.choice(nrmse,(args.bootstrap,len(nrmse)),replace=True).mean(1)
    result=dict(config=vars(args),clients=metrics,mean_nrmse=float(nrmse.mean()),worst_nrmse=float(nrmse.max()),
        client_std=float(nrmse.std()),mean_nrmse_ci=np.quantile(boot,[.025,.975]).tolist(),uploaded_mib=0.,downloaded_mib=0.,flop_proxy=None,completed_rounds=0,runtime_seconds=time.perf_counter()-started)
    for key in ('picp','mpiw','winkler','event_f1','auprc','detection_delay_hours'):
        values=[row[key] for row in metrics if row[key] is not None];result[key]=float(np.mean(values)) if values else None
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
