# BPFBI

**Budgeted Personalized Federated Building Intelligence for Load Forecasting, Calibrated Intervals, and Anomaly Detection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

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

## Frozen evaluation and modules

```bash
bpfbi evaluate --data data/hour1 --model outputs/full/model_000.pt --client 0 --output outputs/frozen_client0
python verify.py --output outputs/functional-check
```

`--client` is the zero-based index in `clients.json`; it must match the checkpoint's model suffix. Frozen evaluation reuses the saved calibration without refitting or selecting on test labels.

| Component / metric | Code | Command / configuration |
|---|---|---|
| Causal TCN, median/ordered quantiles/anomaly head, Huber+pinball+BCE+proximal loss | `bpfbi/model.py`, `bpfbi/cli.py:run` | `bpfbi run --method full ...` |
| Availability/budget selection, debt and capped utility aggregation | `bpfbi/cli.py:run` | `--upload-budget`, `--flop-budget`, `--availability` |
| nRMSE (mean/worst/SD/CI), PICP, MPIW, Winkler, event F1/AUPRC/delay, bytes/FLOP proxy/participation | `bpfbi/cli.py:evaluate/run` | `metrics.json`, `prediction_NNN.npz` |
| Paired DM/HAC and Holm; comparison plots | `bpfbi/cli.py:compare` | `bpfbi compare ...`, `bpfbi plot ...` |

Local training requires `--match-run history.json` to replay selected clients and match per-client optimizer-step counts at equal batch size/local epochs. Event metrics use runs of window-end labels. Debt increments by `clients_per_round/client_count`; local calibration uses a fixed finite-sample validation quantile. Communication counts payload bytes and compute uses a FLOP proxy. EACS requires measured resource telemetry.

Supply all primary comparison runs together; `compare` applies Holm correction across baselines within each building:

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

SeasonalNaive repeats the value one period before the target (`--period 24` supports horizons 1/6/24). Training residual quantiles define intervals; the training CDF of seasonal deviations defines anomaly scores. XGBoost fits quantile regressors at 0.5/0.05/0.95 and a classifier using the same causal inputs, validation calibration and metrics. API: [XGBoost quantile regression](https://xgboost.readthedocs.io/en/release_2.1.0/python/examples/quantile_regression.html). Classical fits report measured wall time, zero network bytes and no neural FLOP proxy.

The common neural CLI also provides `--method centralized|scaffold|ditto`. Centralized TCN uses a single pooled model and the predeclared reference client-batch ordering; `--match-run reference/history.json` gives exactly the corresponding per-client/total optimizer-step budget and no network messages. Its optimizer resets at the same client-chunk boundaries as the federation simulator.

SCAFFOLD is an independent implementation of [Algorithm 1, option II](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf): gradients are corrected by c-c_i, c_i is updated by c_i-c+(w_start-w_end)/(K*lr), and server control deltas are divided by the full client population. Model aggregation is uniform as in that algorithm. `--optimizer sgd` is required; use the same flag for all neural comparisons involving SCAFFOLD. Controls count as an additional full-model payload in both upload and download ledgers. AdamW remains available for the other common-optimizer experiments.

Ditto follows [Algorithm 1](https://proceedings.mlr.press/v139/li21h/li21h.pdf), maintaining private models and separate aggregated global models. Private gradients add `lambda*(v-w_start)` (`--ditto-lambda 0.1`). Minibatches alternate global/private updates within a total K-step budget: ceil(K/2) global and floor(K/2) private, requiring K>=2. Both count toward compute; only global weights are transmitted and frozen evaluation uses private weights.

`--method acsp_fl` adapts [ACSP-FL](https://arxiv.org/abs/2411.17833) equations (4)-(7) / Algorithms 1-2 to the shared TCN and regression validation metric. The original MIT [AllanMSouza/ACSP-FL](https://github.com/AllanMSouza/ACSP-FL) source was checked at commit 187fe05f64746e1b68b57319d32903df645e1410; its license is retained in bpfbi/vendor/acsp/LICENSE. The original accuracy ranking is mapped to negative validation MSE: after all-client validation with the newly aggregated shared encoder and private heads, clients at or above mean MSE remain eligible, ordered worst first, and their number decays as ceil(eligible_count*(1-decay)^round), with `--acsp-decay .01`. Equation (4)'s inclusive comparison is followed, including ties. Encoder-only sample-count aggregation and private heads preserve the layer-sharing mechanism. First-round clients are eligible before filtering, as in Algorithm 1. Availability, byte, compute and cardinality caps then trim the list. A round's all-client validation broadcast and float64 metric replies are included in download/upload ledgers; default upload caps include those replies.

`--method eacs_fl --resource-trace path.npz` implements the EACS-FL resource protocol as the single-global-model special case of [Zhu et al., Computer Networks 2024](https://par.nsf.gov/servlets/purl/10535691), equations (3)-(10) and Algorithm 1. A required external trace supplies energy/delay arrays shaped [rounds+1,clients] and per-client budget[clients]. Row0 holds preflight observations already available before replay; budgets apply to the subsequent replay. In each round, the selection score is V*historical_mean_delay + energy_queue*historical_mean_energy - sqrt((K+1)*log(t)/pulls). Costs of unselected/future rounds do not determine a decision. Selected clients return observed costs, historical means update incrementally, and every queue updates by max(queue+spent_this_round-budget/rounds,0). The common availability, byte, FLOP and cardinality caps apply; only clients with estimated energy fitting the remaining budget remain eligible. Actual finite-horizon budget overshoots are explicitly reported, never clipped or asserted impossible.

A selected client exceeding `--maximum-delay` is trained/accounted but excluded from server aggregation, exactly preserving the deadline failure condition. Both selected updates and sixteen-byte cost telemetry replies count toward upload, while model downloads and total optimizer steps remain in the common ledger. The resource report exports spent energy, budgets, virtual queues, budget violations and deadline failures; physical energy is not inferred from the FLOP proxy. `--resource-weight` specifies V.
