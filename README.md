# BPFBI

Implementation of **Budgeted Personalized Federated Building Intelligence for Load Forecasting, Calibrated Intervals, and Anomaly Detection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. CPU validation: `python verify.py` checks chronological train-only normalization, finite local losses, encoder/head gradients and parameter updates, all seven methods for one communication round, frozen checkpoint evaluation, hand-calculated nRMSE/PICP/MPIW/Winkler examples, paired comparison and figures. Generated two-client readings test functionality; full BDG2/REFIT experiments were not run.

## Data and preprocessing

Obtain [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) or the [REFIT electrical measurements](https://pureportal.strath.ac.uk/en/datasets/31da3ece-f902-4e95-a093-e0a9536983c4). Export hourly readings in a long CSV:

```csv
client,timestamp,load,air_temperature,holiday
office_01,2016-01-01 00:00:00,95.2,8.1,1
```

For BDG2, melt the electricity meter columns, join building/site metadata and weather by timestamp, and choose the building IDs before examining outcomes. For REFIT, aggregate the household's aggregate-power column to hourly mean Watts. Keep one physical building per `client`, consistent units within each federation, and one row per hour. Additional weather columns are explicitly declared; the holiday column is optional and defaults to zero.

```bash
bpfbi prepare --csv data/meters.csv --weather air_temperature --output data/hour1 --horizon 1 --seed 11
```

Clients below 95% completeness are excluded. Each client's timeline is split 60/20/20. Within each input window, complete bounded missing runs of at most three hours are linearly interpolated. Longer and unbounded runs remain masked. Interpolation never reads after the forecast origin or across chronological splits. Training statistics determine each client's load/weather normalization. Window extraction stays inside each split. Inputs contain 168 past hourly readings, masks, cyclic calendar features, holidays and declared weather. Original missing targets are excluded from supervision.

`--anomaly-fraction` selects exactly round(fraction × usable-window-count) windows in each split after chronological splitting (default 0.3). Each selected window contains exactly one seeded spike/drift/freeze/missing event; its class label denotes event presence in that window. Per-window inputs and event kind/start/end are frozen in the prepared NPZ, so overlapping windows can carry different simulated events. Forecast targets remain the clean readings; detection uses the window head and the observed target exceedance when an event covers that timestamp. Use `--horizon 1`, `6` or `24` in separate prepared directories.

## Training and inference

```bash
bpfbi run --data data/hour1 --output outputs/full --method full --rounds 100 --clients-per-round 12 --local-epochs 2 --seed 11
```

## Evaluation

The run fits validation calibration and evaluates each building on its held-out timeline.

## Comparisons and ablations

```bash
bpfbi run --data data/hour1 --output outputs/fedper --method fedper --seed 11
bpfbi run --data data/hour1 --output outputs/no_debt --method no_debt --seed 11
bpfbi run --data data/hour1 --output outputs/local --method local --match-run outputs/full/history.json --rounds 100 --local-epochs 2 --seed 11
```

Also supported: `fedavg`, `fedprox`, `local`, `no_calibration`. Fix the seed, availability schedule and resource limits.

## Outputs

Each `--output` contains `model_NNN.pt`, `prediction_NNN.npz`, `history.json` and `metrics.json` for per-client models/calibration, aligned forecasts, participation/resource history and metrics.

```bash
bpfbi compare --first outputs/full --second outputs/fedper --output outputs/paired.json --hac-lag 24
bpfbi plot --metrics outputs/full/metrics.json outputs/fedper/metrics.json outputs/no_debt/metrics.json --output outputs/figures
```

## Frozen evaluation and implementation mapping

```bash
bpfbi evaluate --data data/hour1 --model outputs/full/model_000.pt --client 0 --output outputs/frozen_client0
python verify.py --output outputs/functional-check
```

`--client` is the zero-based index in `clients.json`; it must match the checkpoint's model suffix. Frozen evaluation reuses the saved calibration without refitting or selecting on test labels.

| Paper component / metric | Code | Command / configuration |
|---|---|---|
| Causal TCN, median/ordered quantiles/anomaly head, Huber+pinball+BCE+proximal loss | `bpfbi/model.py`, `bpfbi/cli.py:run` | `bpfbi run --method full ...` |
| Availability/budget selection, debt and capped utility aggregation | `bpfbi/cli.py:run` | `--upload-budget`, `--flop-budget`, `--availability` |
| nRMSE (mean/worst/SD/CI), PICP, MPIW, Winkler, event F1/AUPRC/delay, bytes/FLOP proxy/participation | `bpfbi/cli.py:evaluate/run` | `metrics.json`, `prediction_NNN.npz` |
| Paired DM/HAC and Holm; comparison plots | `bpfbi/cli.py:compare` | `bpfbi compare ...`, `bpfbi plot ...` |

The supplied implementation is separate from the paper's reported results. Local/FedAvg/FedProx/FedPer and no-debt/no-calibration variants are executable; the local baseline requires the reference `--match-run history.json` and uses exactly the same selected clients in each round; with the same batch size/local epochs it records equal per-client optimizer-step counts; Seasonal Naive, XGBoost, pooled centralized TCN, SCAFFOLD, Ditto and external selector baselines are not bundled. Synthetic corruption now targets windows exactly and exports the event ledger. Event precision/recall therefore refers to runs of window-end detection labels on the decision grid. The debt increment generalizes 12/48 to `clients_per_round/client_count`. The finite-sample validation quantile is used as a fixed local calibration expansion. The script reports communication payload and FLOP proxies, not measured distributed-network or energy costs.

Actual environment: Python 3.13.5, PyTorch 2.7.1+cu118, NumPy 2.1.3, SciPy 1.15.3, CPU, seed 11.

For the paper's multiplicity family, supply all primary baseline runs together. `compare` applies Holm correction across those baselines separately within each building:

```bash
bpfbi compare --first outputs/full --second outputs/fedavg outputs/fedper --output outputs/paired_primary.json --hac-lag 24
```

## Classical baseline entry

```bash
python -m pip install -e .[classical]
python -m bpfbi.classical --data data/hour1 --output outputs/seasonal --method seasonal_naive --period 24
python -m bpfbi.classical --data data/hour1 --output outputs/xgboost --method xgboost --trees 100
python -m bpfbi.classical --data data/hour1 --output outputs/xgboost_frozen --method xgboost --checkpoint-dir outputs/xgboost
python verify.py
python verify_classical.py
```

SeasonalNaive repeats the reading one declared period before the forecast target; period=24 covers horizons1/6/24. Train residual quantiles form local intervals, and the empirical training CDF of window seasonal deviations supplies its anomaly score. XGBoost uses the actual XGBRegressor quantile objective at 0.5/0.05/0.95 and XGBClassifier for the same window labels; both consume the same prepared causal inputs and reuse validation calibration/common metrics. Source/API: [XGBoost quantile regression](https://xgboost.readthedocs.io/en/release_2.1.0/python/examples/quantile_regression.html). No replacement tree model is called XGBoost. Classical training is outside federated optimizer-step accounting; the neural FLOP proxy is null, networking bytes zero, wall time measured.

CPU checks fitted both baselines on the generated two-client fixture, loaded independent checkpoints and matched predictions, reused all common forecast/interval/event metrics, and hand-checked the seasonal index. Local XGBoost3.4.1 was installed into ignored outputs/xgb-deps only. SCAFFOLD, Ditto, pooled central and literature selector integrations continue separately.
