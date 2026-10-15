# BPFBI

Implementation of **Budgeted Personalized Federated Building Intelligence for Load Forecasting, Calibrated Intervals, and Anomaly Detection**.

The model shares a causal temporal encoder across buildings, retains four local heads, selects clients under upload and computation budgets, and calibrates each building's prediction intervals separately.

## Installation

Use Python 3.11 or later. Create and activate a virtual environment, then install from this directory:

```bash
python -m pip install -e .
```

PyTorch runs on CPU by default. Install a suitable CUDA PyTorch build and pass `--device cuda` to use a GPU.

## Data

Obtain [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) or the [REFIT electrical measurements](https://pureportal.strath.ac.uk/en/datasets/31da3ece-f902-4e95-a093-e0a9536983c4). Export hourly readings in a long CSV:

```csv
client,timestamp,load,air_temperature,holiday
office_01,2016-01-01 00:00:00,95.2,8.1,1
```

For BDG2, melt the electricity meter columns, join building/site metadata and weather by timestamp, and choose the building IDs before examining outcomes. For REFIT, aggregate the household's aggregate-power column to hourly mean Watts. Keep one physical building per `client`, consistent units within each federation, and one row per hour. Additional weather columns are explicitly declared; the holiday column is optional and defaults to zero.

```bash
bpfbi prepare --csv data/meters.csv --weather air_temperature --output data/hour1 --horizon 1 --seed 11
```

Clients below 95% completeness are excluded. Each client's timeline is split 60/20/20. Interpolation fills only complete missing runs of at most three hours inside a split; longer gaps retain an observation-mask feature. Training statistics determine each client's load/weather normalization. Window extraction stays inside each split. Inputs contain 168 past hourly readings, masks, cyclic calendar features, holidays and declared weather. Original missing targets are excluded from supervision.

Anomaly injection happens after splitting. `--anomaly-fraction 0.3` targets 30% affected hours using nonoverlapping spike, drift, freeze and missing events; `0` creates clean data. Separate split-specific random streams prevent identical event placement across partitions. Forecast targets retain the original clean load; the observed, potentially corrupted meter value and event label are stored separately. This convention measures recovery of clean demand and detection of meter corruption. The anomaly head predicts from the historical window; interval exceedance becomes available when the target reading arrives. Missing targets give maximum exceedance. Forecasting never receives that future reading.

Prepare separate data directories with `--horizon 6` and `--horizon 24` for longer forecasts. A 48-client experiment requires selecting 48 eligible meters in the input CSV; use the building metadata to select 12 from each required group.

## Federated training and evaluation

```bash
bpfbi run --data data/hour1 --output outputs/full --method full --rounds 100 --clients-per-round 12 --local-epochs 2 --seed 11
bpfbi run --data data/hour1 --output outputs/fedper --method fedper --seed 11
bpfbi run --data data/hour1 --output outputs/no_debt --method no_debt --seed 11
```

Four residual TCN blocks use 64 channels, kernel size three and dilations 1, 2, 4 and 8. Left padding keeps every convolution causal. Local heads produce the median, noncrossing 0.05/0.95 quantiles and anomaly logit. The loss combines Huber, pinball and anomaly BCE with the paper's weights. The shared encoder receives a proximal penalty of 0.001 in the full method. Local AdamW optimizers start anew for each communication round.

The selector solves a binary linear program with separate upload-byte, local-operation and client-count constraints. Utility is the exponentially smoothed validation-MSE reduction from previous participation. The debt update uses the selected fraction, giving 0.25 for 12 of 48 clients. Successful encoder updates use capped sample count times positive utility. Personal heads and client normalization/calibration statistics stay local in this single-process federation simulation.

| Option | Default | Meaning |
|---|---:|---|
| `--rounds`, `--local-epochs` | 100, 2 | Communication rounds and local passes |
| `--clients-per-round` | 12 | Maximum selected clients |
| `--upload-budget` | derived | Per-round bytes, by actual transmitted parameter size |
| `--flop-budget` | derived | Per-round operation proxy |
| `--availability-probability` | 0.9 | Seeded independent availability when no schedule is supplied |
| `--availability` | absent | Boolean NPY array shaped rounds by client count |
| `--lr`, `--weight-decay` | 0.001, 0.00001 | Local AdamW settings |
| `--width`, `--batch-size` | 64, 64 | Encoder width and minibatch size |
| `--bootstrap` | 10000 | Building-level resamples for mean nRMSE confidence interval |

Default budgets permit the maximum client count even for the most expensive available subset. Set explicit budgets to study resource restrictions. The operation proxy is `2 * model_parameters * windows * input_length * local_epochs`; it is a consistent workload proxy, not a hardware instruction count or measured energy. Upload/download logs use actual parameter tensor bytes. The implemented encoder parameter count depends on feature count; its measured size determines the budget rather than assuming 178,176 parameters.

Methods are `full`, `fedavg`, `fedprox`, `fedper`, `local`, `no_debt`, and `no_calibration`. FedAvg/FedProx share the heads as well as the encoder and use sample-weighted aggregation. FedPer retains local heads and uniformly randomized feasible client selection. Local training shares no parameters. The two ablations remove the debt priority or interval expansion while retaining the other full-method components. Use identical prepared files, schedules, seeds and resource limits for comparisons.

## Outputs and interpretation

The selected model for each client is saved as `model_NNN.pt` with its calibration radius and normalization. `history.json` records selected clients, validation losses, debt and cumulative resource use. `prediction_NNN.npz` contains timestamp-aligned clean targets, point predictions, intervals, anomaly scores and squared errors.

`metrics.json` reports mean/worst/client-standard-deviation nRMSE; per-building and aggregate coverage, width and 90% Winkler score; anomaly AUPRC, event F1 and detection delay; client participation, completed rounds, upload/download MiB, operation proxy and elapsed time. AUPRC is null for a one-class test set and delay is null without a matched detection. Event matching is one-to-one timestamp overlap. A detected episode spanning two reference events receives one match. Width and errors use original meter units. A confidence interval uses buildings as sampling units.

Calibration uses only validation residuals and a finite-sample higher quantile for 90% coverage. Test data do not update calibration or parameters. The alarm threshold is 0.80, combining 0.60 sigmoid probability with 0.40 normalized calibrated interval exceedance.

```bash
bpfbi compare --first outputs/full --second outputs/fedper --output outputs/paired.json --hac-lag 24
bpfbi plot --metrics outputs/full/metrics.json outputs/fedper/metrics.json outputs/no_debt/metrics.json --output outputs/figures
```

`compare` checks target alignment, calculates building-level Diebold–Mariano statistics with Bartlett HAC variance, and applies Holm correction across buildings in that comparison. Negative squared-error differences favor the first method. `plot` creates forecast, interval, anomaly and communication comparisons from generated results. Repeat fitting with seeds 11, 23, 37, 53 and 71 and all three horizons to construct the full experiment matrix.
