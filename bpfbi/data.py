import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(args):
    table = pd.read_csv(args.csv, parse_dates=["timestamp"])
    required = {"client", "timestamp", "load"}
    if not required.issubset(table):
        raise ValueError("CSV requires client,timestamp,load and optional declared weather columns")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clients = []
    for number, (client, frame) in enumerate(table.groupby("client", sort=True)):
        frame = frame.sort_values("timestamp").set_index("timestamp")
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate hourly timestamp for {client}")
        frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h"))
        raw = frame["load"].to_numpy(float)
        if np.mean(np.isfinite(raw)) < args.minimum_completeness:
            continue
        boundaries = (0, int(0.6 * len(frame)), int(0.8 * len(frame)), len(frame))
        mean = float(np.nanmean(raw[:boundaries[1]]))
        scale = float(max(np.nanstd(raw[:boundaries[1]]), 1e-6))
        weather_mean = frame[args.weather].iloc[:boundaries[1]].mean().to_numpy() if args.weather else np.empty(0)
        weather_scale = frame[args.weather].iloc[:boundaries[1]].std().clip(lower=1e-6).to_numpy() if args.weather else np.empty(0)
        arrays = dict()
        for part, left, right in zip(("train", "val", "test"), boundaries[:-1], boundaries[1:]):
            split = frame.iloc[left:right]
            clean = split["load"].to_numpy(float)
            valid = np.isfinite(clean)
            observed = pd.Series(clean).ffill(limit=3).to_numpy()
            anomaly = np.zeros(len(split), dtype=np.float32)
            rng = np.random.default_rng(args.seed + 1009 * number + left)
            if args.anomaly_fraction:
                occupied = np.zeros(len(split), dtype=bool)
                candidates = rng.permutation(max(0, len(split) - 12))
                target = int(args.anomaly_fraction * len(split))
                for start in candidates:
                    if occupied.sum() >= target:
                        break
                    kind = int(rng.integers(4))
                    length = (3, 12, 6, 4)[kind]
                    end = start + length
                    if occupied[start:end].any() or not valid[start:end].all() or start == 0:
                        continue
                    previous = observed[:start][np.isfinite(observed[:start])]
                    if kind == 2 and not len(previous):
                        continue
                    occupied[start:end] = True
                    anomaly[start:end] = 1
                    if kind == 0:
                        observed[start:end] += 3 * scale
                    elif kind == 1:
                        observed[start:end] += 0.25 * scale * np.arange(1, length + 1)
                    elif kind == 2:
                        observed[start:end] = previous[-1]
                    else:
                        observed[start:end] = np.nan
            mask = np.isfinite(observed)
            values = np.nan_to_num((observed - mean) / scale, nan=0)
            index = split.index
            calendar = np.stack([np.sin(2 * np.pi * index.hour / 24), np.cos(2 * np.pi * index.hour / 24),
                                 np.sin(2 * np.pi * index.dayofweek / 7), np.cos(2 * np.pi * index.dayofweek / 7),
                                 np.sin(2 * np.pi * index.month / 12), np.cos(2 * np.pi * index.month / 12)], 1)
            weather = np.nan_to_num((split[args.weather].to_numpy(float) - weather_mean) / weather_scale) if args.weather else np.empty((len(split), 0))
            holiday = split["holiday"].fillna(0).to_numpy(float)[:, None] if "holiday" in split else np.zeros((len(split), 1))
            features = np.column_stack([values, mask.astype(float), calendar, holiday, weather]).astype(np.float32)
            origins = np.arange(args.window - 1, len(split) - args.horizon)
            origins = origins[valid[origins + args.horizon]]
            if len(origins) == 0:
                raise ValueError(f"Client {client} has no usable {part} windows")
            arrays[f"{part}_features"] = features
            arrays[f"{part}_origin"] = origins
            arrays[f"{part}_y"] = ((clean[origins + args.horizon] - mean) / scale).astype(np.float32)
            arrays[f"{part}_observed"] = ((observed[origins + args.horizon] - mean) / scale).astype(np.float32)
            arrays[f"{part}_anomaly"] = anomaly[origins + args.horizon]
            arrays[f"{part}_time"] = index[origins + args.horizon].asi8
        name = f"client_{number:03d}.npz"
        np.savez_compressed(output / name, **arrays, mean=mean, scale=scale, window=args.window, horizon=args.horizon)
        clients.append(dict(client=str(client), file=name))
    if not clients:
        raise ValueError("No client meets the requested completeness")
    (output / "clients.json").write_text(json.dumps(clients, indent=2), encoding="utf-8")


class Windows:
    def __init__(self, data, part):
        self.features = data[f"{part}_features"]
        self.origins = data[f"{part}_origin"]
        self.y = data[f"{part}_y"]
        self.anomaly = data[f"{part}_anomaly"]
        self.window = int(data["window"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        end = int(self.origins[index]) + 1
        return self.features[end - self.window:end], self.y[index], self.anomaly[index]
