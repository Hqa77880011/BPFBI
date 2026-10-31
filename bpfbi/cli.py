import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import norm
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .data import Windows, prepare
from .model import BuildingModel, objective


def predict(model, dataset, device, batch_size):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, _, _ in DataLoader(dataset, batch_size=batch_size):
            rows.append(torch.stack(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(rows)


def intervals(mask, times):
    result, current = [], []
    for active, timestamp in zip(mask, times):
        if current and (not active or timestamp - current[-1] > 3_600_000_000_000):
            result.append(current)
            current = []
        if active:
            current.append(int(timestamp))
    return result + ([current] if current else [])


def evaluate(prediction, data, calibration):
    median, lower, upper, logit = prediction.T
    lower, upper = lower - calibration, upper + calibration
    y, observed = data["test_y"], data["test_observed"]
    excess = np.maximum.reduce([lower - observed, observed - upper, np.zeros(len(y))])
    excess[~np.isfinite(observed)] = upper[~np.isfinite(observed)] - lower[~np.isfinite(observed)]
    anomaly_score = 0.6 / (1 + np.exp(-np.clip(logit, -50, 50))) + 0.4 * np.minimum(excess / (upper - lower + 1e-6), 1)
    actual = intervals(data["test_anomaly"] > 0, data["test_time"])
    detected = intervals(anomaly_score >= 0.8, data["test_time"])
    matches, used, delays = 0, set(), []
    for event in actual:
        candidates = [i for i, alarm in enumerate(detected) if i not in used and alarm[0] <= event[-1] and alarm[-1] >= event[0]]
        if candidates:
            selected = min(candidates, key=lambda i: max(detected[i][0], event[0]))
            used.add(selected)
            matches += 1
            delays.append(max(0, detected[selected][0] - event[0]) / 3_600_000_000_000)
    scale, mean = float(data["scale"]), float(data["mean"])
    truth = y * scale + mean
    error = (median - y) * scale
    width = (upper - lower) * scale
    winkler = width + 20 * scale * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    metrics = dict(nrmse=float(np.sqrt(np.mean(error ** 2)) / (np.mean(np.abs(truth)) + 1e-6)),
                   picp=float(np.mean((y >= lower) & (y <= upper))), mpiw=float(width.mean()),
                   winkler=float(winkler.mean()), event_f1=2 * matches / max(1, len(actual) + len(detected)),
                   auprc=float(average_precision_score(data["test_anomaly"], anomaly_score)) if len(np.unique(data["test_anomaly"])) > 1 else None,
                   detection_delay_hours=float(np.mean(delays)) if delays else None)
    return metrics, dict(y=truth, prediction=median * scale + mean, lower=lower * scale + mean,
                         upper=upper * scale + mean, anomaly_score=anomaly_score, squared_error=error ** 2,
                         time=data["test_time"])


def run(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "clients.json").read_text(encoding="utf-8"))
    data = []
    for row in manifest:
        with np.load(root / row["file"], allow_pickle=False) as values:
            data.append({key: values[key] for key in values.files})
    training = [Windows(values, "train") for values in data]
    validation = [Windows(values, "val") for values in data]
    models = [BuildingModel(values.features.shape[1], args.width).to(args.device) for values in training]
    shared = copy.deepcopy(models[0].encoder.state_dict())
    global_head = copy.deepcopy(models[0].head.state_dict())
    count = len(data)
    upload_bytes = sum(value.numel() * value.element_size() for value in shared.values())
    head_bytes = sum(value.numel() * value.element_size() for value in global_head.values())
    global_method = args.method in ("fedavg", "fedprox", "scaffold", "ditto", "centralized", "eacs_fl")
    if args.method=="scaffold" and args.optimizer!="sgd":
        raise ValueError("SCAFFOLD option II requires --optimizer sgd; use the same optimizer for matched neural comparisons")
    if args.method=="centralized":models=[models[0]]*len(data)
    personal=[copy.deepcopy(model) for model in models] if args.method=="ditto" else None
    control={name:torch.zeros_like(p) for name,p in models[0].named_parameters()}
    client_control=[copy.deepcopy(control) for _ in models]
    message_bytes = upload_bytes + (head_bytes if global_method else 0)
    if args.method=="scaffold":message_bytes*=2
    debt, utility, participation = np.zeros(count), np.ones(count), np.zeros(count, dtype=int)
    last_loss = np.full(count, np.nan)
    acsp_loss=np.zeros(count)
    operations = np.array([2 * sum(p.numel() for p in model.parameters()) * len(dataset) * dataset.window * args.local_epochs
                           for model, dataset in zip(models, training)], dtype=float)
    limit_count = min(args.clients_per_round, count)
    byte_budget = args.upload_budget if args.upload_budget is not None else limit_count * message_bytes+(count*8 if args.method=="acsp_fl" else limit_count*16 if args.method=="eacs_fl" else 0)
    flop_budget = args.flop_budget if args.flop_budget is not None else float(np.sort(operations)[-limit_count:].sum())
    schedule = np.load(args.availability, allow_pickle=False) if args.availability else rng.random((args.rounds, count)) < args.availability_probability
    if schedule.shape != (args.rounds, count):
        raise ValueError("Availability must have shape rounds by client count")
    matched = json.loads(Path(args.match_run).read_text(encoding="utf-8")) if args.match_run else None
    if matched is not None and len(matched) != args.rounds:
        raise ValueError("Matched reference history must have exactly --rounds rounds")
    resource=None
    if args.method=="eacs_fl":
        if not args.resource_trace:raise ValueError("EACS-FL requires --resource-trace with measured delay/energy and budgets")
        from .resource_protocol import EACS
        resource=EACS(args.resource_trace,count,args.rounds,limit_count,args.resource_weight,args.maximum_delay)
    steps = np.zeros(count,dtype=int)
    history, total_bytes, total_flops, downloaded = [], 0, 0.0, 0
    started = time.perf_counter()
    for round_index in range(args.rounds):
        available = schedule[round_index].astype(bool)
        if matched is not None:
            selected = np.asarray(matched[round_index]["clients"],dtype=int)
            if (selected<0).any() or (selected>=count).any() or len(set(selected))!=len(selected):
                raise ValueError("Invalid matched client identities")
        elif args.method in ("local","centralized"):
            raise ValueError("Local/centralized matched-budget baseline requires --match-run reference/history.json")
        elif args.method=="eacs_fl":
            selected=[];used_bytes=0;used_flops=0.
            for client in resource.ranking(round_index,available):
                if len(selected)<limit_count and used_bytes+message_bytes+16<=byte_budget and used_flops+operations[client]<=flop_budget:
                    selected.append(client);used_bytes+=message_bytes+16;used_flops+=operations[client]
            selected=np.asarray(selected,dtype=int)
        elif args.method=="acsp_fl":
            from .selection import acsp_candidates
            candidates=acsp_candidates(acsp_loss,round_index,args.acsp_decay)
            selected=[];used_bytes=count*8;used_flops=0.
            for client in candidates:
                if available[client] and len(selected)<limit_count and used_bytes+message_bytes<=byte_budget and used_flops+operations[client]<=flop_budget:
                    selected.append(client);used_bytes+=message_bytes;used_flops+=operations[client]
            selected=np.asarray(selected,dtype=int)
        else:
            value = utility + (0 if args.method == "no_debt" else 0.30 * debt) - 0.20 * operations / max(operations.max(), 1)
            if args.method in ("fedavg", "fedprox", "fedper", "scaffold", "ditto"):
                value = rng.random(count) + 1
            value = value + rng.uniform(0, 1e-9, count)
            result = milp(-value, integrality=np.ones(count), bounds=Bounds(np.zeros(count), available.astype(float)),
                          constraints=LinearConstraint(np.stack([np.full(count, message_bytes), operations, np.ones(count)]),
                                                       np.zeros(3), [byte_budget, flop_budget, limit_count]),
                          options={"time_limit": 10})
            if result.x is None:
                raise RuntimeError(f"Budget selection failed: {result.message}")
            selected = np.flatnonzero(result.x > 0.5)
        updates, heads, weights, values = [], [], [], []
        control_delta=[]
        for client in selected:
            model = models[client]
            if args.method not in ("local","centralized"):
                model.encoder.load_state_dict(shared)
                if global_method:
                    model.head.load_state_dict(global_head)
            reference = {name: tensor.detach().clone() for name, tensor in model.encoder.named_parameters()}
            optimizer_class=torch.optim.SGD if args.optimizer=="sgd" else torch.optim.AdamW
            optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            start_state={name:p.detach().clone() for name,p in model.named_parameters()}
            if personal is not None:
                personal_optimizer=optimizer_class(personal[client].parameters(),lr=args.lr,weight_decay=args.weight_decay)
                personal[client].train()
            round_steps=0
            model.train()
            generator = torch.Generator().manual_seed(args.seed + round_index * count + int(client))
            for _ in range(args.local_epochs):
                for x, y, anomaly in DataLoader(training[client], batch_size=args.batch_size, shuffle=True, generator=generator):
                    active_model=personal[client] if personal is not None and round_steps%2 else model
                    active_optimizer=personal_optimizer if active_model is not model else optimizer
                    loss = objective(active_model(x.to(args.device)), y.to(args.device), anomaly.to(args.device))
                    if active_model is not model:
                        loss=loss+args.ditto_lambda/2*sum((p-start_state[name]).square().sum() for name,p in active_model.named_parameters())
                    prox = 0.001 if args.method in ("full", "no_debt", "no_calibration", "fedprox") else 0
                    if prox:
                        loss = loss + prox / 2 * sum((p - reference[name]).square().sum() for name, p in model.encoder.named_parameters())
                    active_optimizer.zero_grad()
                    loss.backward()
                    if args.method=="scaffold":
                        for name,p in model.named_parameters():p.grad.add_(control[name]-client_control[client][name])
                    active_optimizer.step()
                    round_steps+=1
                    steps[client] += 1
            if personal is not None and round_steps<2:raise ValueError("Ditto requires at least two total steps per selected client")
            if args.method=="scaffold":
                new_control={name:client_control[client][name]-control[name]+(start_state[name]-p.detach())/(round_steps*args.lr) for name,p in model.named_parameters()}
                control_delta.append({name:new_control[name]-client_control[client][name] for name in control})
                client_control[client]=new_control
            estimate = predict(personal[client] if personal is not None else model, validation[client], args.device, args.batch_size)
            val_loss = float(np.mean((estimate[:, 0] - validation[client].y) ** 2))
            improvement = max(0, last_loss[client] - val_loss) if np.isfinite(last_loss[client]) else 0.001
            utility[client] = 0.8 * utility[client] + 0.2 * improvement
            last_loss[client] = val_loss
            weights.append(min(len(training[client]), 2 * np.median([len(ds) for ds in training])) * max(utility[client], 0.001))
            updates.append({name: value.detach().clone() for name, value in model.encoder.state_dict().items()})
            heads.append({name: value.detach().clone() for name, value in model.head.state_dict().items()})
            values.append(val_loss)
            participation[client] += 1
        if resource is not None:
            on_time=resource.observe(round_index,selected)
            updates=[value for value,valid in zip(updates,on_time) if valid];heads=[value for value,valid in zip(heads,on_time) if valid]
            weights=[weight for weight,valid in zip(weights,on_time) if valid]
        if len(selected) and args.method not in ("local","centralized"):
            total_bytes += len(selected) * (message_bytes+(16 if resource is not None else 0))
            downloaded += len(selected) * message_bytes
        if len(updates) and args.method not in ("local","centralized"):
            if global_method or args.method in ("fedper","acsp_fl"):
                weights = [1 if args.method=="scaffold" else len(training[client]) for j,client in enumerate(selected) if resource is None or on_time[j]]
            normalized = np.asarray(weights) / np.sum(weights)
            shared = {key: sum(float(weight) * value[key] for weight, value in zip(normalized, updates)) for key in shared}
            if global_method:
                global_head = {key: sum(float(weight) * value[key] for weight, value in zip(normalized, heads)) for key in global_head}
            if args.method=="scaffold":
                control={name:control[name]+sum(delta[name] for delta in control_delta)/count for name in control}
        if args.method=="centralized":
            shared=copy.deepcopy(models[0].encoder.state_dict());global_head=copy.deepcopy(models[0].head.state_dict())
        total_flops += float(operations[selected].sum())
        if args.method=="acsp_fl":
            for client,model in enumerate(models):
                model.encoder.load_state_dict(shared)
                estimate=predict(model,validation[client],args.device,args.batch_size)
                acsp_loss[client]=float(np.mean((estimate[:,0]-validation[client].y)**2))
            total_bytes+=count*8;downloaded+=count*message_bytes
        selected_mask = np.zeros(count)
        selected_mask[selected] = 1
        debt = np.clip(debt + limit_count / count - selected_mask, 0, 1)
        history.append(dict(round=round_index + 1, clients=selected.tolist(), validation_mse=values, debt=debt.tolist(),
                            uploaded_bytes=total_bytes, flop_proxy=total_flops, optimizer_steps=steps.tolist(), acsp_validation_mse=acsp_loss.tolist() if args.method=="acsp_fl" else None, resource_protocol=resource.report() if resource is not None else None))
    all_metrics = []
    for client, (model, values) in enumerate(zip(models, data)):
        if args.method=="ditto":model=personal[client]
        if args.method not in ("local","ditto","centralized"):
            model.encoder.load_state_dict(shared)
            if global_method:
                model.head.load_state_dict(global_head)
        val = predict(model, validation[client], args.device, args.batch_size)
        score = np.maximum.reduce([val[:, 1] - values["val_y"], values["val_y"] - val[:, 2], np.zeros(len(val))])
        rank = min(len(score), int(np.ceil((len(score) + 1) * 0.9)))
        calibration = 0.0 if args.method == "no_calibration" else float(np.partition(score, rank - 1)[rank - 1])
        prediction = predict(model, Windows(values, "test"), args.device, args.batch_size)
        metrics, predictions = evaluate(prediction, values, calibration)
        metrics.update(client=manifest[client]["client"], calibration=calibration)
        all_metrics.append(metrics)
        np.savez_compressed(output / f"prediction_{client:03d}.npz", **predictions)
        torch.save(dict(state=model.state_dict(), calibration=calibration, mean=float(values["mean"]),
                        scale=float(values["scale"]), config=vars(args)), output / f"model_{client:03d}.pt")
    nrmse = np.array([row["nrmse"] for row in all_metrics])
    boot = rng.choice(nrmse, (args.bootstrap, count), replace=True).mean(1)
    summary = dict(config=vars(args), clients=all_metrics, mean_nrmse=float(nrmse.mean()), worst_nrmse=float(nrmse.max()),
                   client_std=float(nrmse.std()), mean_nrmse_ci=np.quantile(boot, [0.025, 0.975]).tolist(),
                   uploaded_mib=total_bytes / 1024 ** 2, downloaded_mib=downloaded / 1024 ** 2,
                   flop_proxy=total_flops, completed_rounds=len(history), participation=participation.tolist(), optimizer_steps=steps.tolist(),
                   runtime_seconds=time.perf_counter() - started, resource_protocol=resource.report() if resource is not None else None, shared_parameters=sum(t.numel() for t in shared.values()))
    for key in ("picp", "mpiw", "winkler", "event_f1", "auprc", "detection_delay_hours"):
        finite = [row[key] for row in all_metrics if row[key] is not None]
        summary[key] = float(np.mean(finite)) if finite else None
    for name, value in (("metrics", summary), ("history", history)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def compare(args):
    first = Path(args.first)
    rows = []
    for baseline in args.second:
        second = Path(baseline)
        for path in sorted(first.glob("prediction_*.npz")):
            with np.load(path, allow_pickle=False) as a, np.load(second / path.name, allow_pickle=False) as b:
                if not np.array_equal(a["time"], b["time"]) or not np.array_equal(a["y"], b["y"]):
                    raise ValueError("Paired comparisons require identical target records")
                difference = a["squared_error"] - b["squared_error"]
                centered = difference - difference.mean()
                lag = min(args.hac_lag, len(difference) - 1)
                variance = np.dot(centered, centered) / len(difference)
                for offset in range(1, lag + 1):
                    variance += 2 * (1 - offset / (lag + 1)) * np.dot(centered[offset:], centered[:-offset]) / len(difference)
                statistic = float(difference.mean() / np.sqrt(max(variance, 1e-12) / len(difference)))
                rows.append(dict(baseline=second.name, client=path.stem, mse_difference=float(difference.mean()),
                                 dm=statistic, p=float(2 * norm.sf(abs(statistic)))))
    for client in {row["client"] for row in rows}:
        family = sorted([row for row in rows if row["client"] == client], key=lambda row: row["p"])
        maximum = 0.0
        for rank, row in enumerate(family):
            maximum = max(maximum, min(1, (len(family) - rank) * row["p"]))
            row["holm_p"] = maximum
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def evaluate_checkpoint(args):
    """Evaluate a frozen client checkpoint using the saved local calibration."""
    root = Path(args.data)
    manifest = json.loads((root / "clients.json").read_text(encoding="utf-8"))
    with np.load(root / manifest[args.client]["file"], allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    model = BuildingModel(values["train_features"].shape[1], checkpoint["config"]["width"]).to(args.device)
    model.load_state_dict(checkpoint["state"])
    prediction = predict(model, Windows(values, "test"), args.device, args.batch_size)
    metrics, arrays = evaluate(prediction, values, checkpoint["calibration"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **arrays)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--csv", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--weather", nargs="*", default=[])
    data.add_argument("--minimum-completeness", type=float, default=0.95)
    data.add_argument("--window", type=int, default=168)
    data.add_argument("--horizon", type=int, choices=[1, 6, 24], default=1)
    data.add_argument("--anomaly-fraction", type=float, default=0.3)
    data.add_argument("--seed", type=int, default=11)
    train = sub.add_parser("run")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--method", choices=["full", "fedavg", "fedprox", "fedper", "local", "no_debt", "no_calibration", "centralized", "scaffold", "ditto", "acsp_fl", "eacs_fl"], default="full")
    train.add_argument("--rounds", type=int, default=100)
    train.add_argument("--clients-per-round", type=int, default=12)
    train.add_argument("--local-epochs", type=int, default=2)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--optimizer", choices=["adamw","sgd"], default="adamw")
    train.add_argument("--acsp-decay", type=float, default=.01)
    train.add_argument("--ditto-lambda", type=float, default=.1)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.00001)
    train.add_argument("--upload-budget", type=int)
    train.add_argument("--flop-budget", type=float)
    train.add_argument("--resource-trace")
    train.add_argument("--resource-weight",type=float,default=1.)
    train.add_argument("--maximum-delay",type=float,default=3600.)
    train.add_argument("--availability")
    train.add_argument("--match-run", help="Reference history.json for exact per-client optimizer-step matching")
    train.add_argument("--availability-probability", type=float, default=0.9)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--bootstrap", type=int, default=10000)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--client", type=int, default=0)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--device", default="cpu")
    paired = sub.add_parser("compare")
    paired.add_argument("--first", required=True)
    paired.add_argument("--second", nargs="+", required=True)
    paired.add_argument("--output", required=True)
    paired.add_argument("--hac-lag", type=int, default=24)
    plot = sub.add_parser("plot")
    plot.add_argument("--metrics", nargs="+", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.window < 1 or not 0 <= args.anomaly_fraction < 1:
            parser.error("Positive window and anomaly fraction in [0,1) required")
        prepare(args)
    elif args.command == "run":
        if min(args.rounds, args.local_epochs, args.batch_size, args.clients_per_round, args.bootstrap, args.lr) <= 0:
            parser.error("Training sizes and learning rate must be positive")
        run(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    elif args.command == "compare":
        compare(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for metric in ("mean_nrmse", "worst_nrmse", "picp", "mpiw", "winkler", "event_f1", "uploaded_mib"):
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.bar([row["config"]["method"] for row in records], [row[metric] for row in records])
            axis.set_ylabel(metric)
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
