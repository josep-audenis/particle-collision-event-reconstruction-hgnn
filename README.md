# LHCb HGNN for DFEI and IFT

Heterogeneous graph neural network (HGNN) workflow for LHCb event reconstruction:

- **DFEI** — track-level reconstruction: which tracks come from b-hadron decays, the
  track–track decay topology (LCA), and primary-vertex (PV) association, with node/edge pruning.
- **IFT** — inclusive flavour tagging (B / B̄ origin) on top of the DFEI-reconstructed graph,
  with calibrated effective tagging power as the figure of merit.

Everything is PyTorch Lightning based and driven by YAML configs.

## Repository layout

```text
config_files/                 YAML configs (train_*/evaluate_*), see config_files/README.md
  cache_10000/ cache_50000/ cache_100000/   per-dataset-size experiment configs
  cache_200000/README.md                    plan for the 200k run
scripts/
  cache_montecarlo.py         raw ROOT -> cached PyG graphs (3-stage driver)
  montecarlo_cache_support/   stage code: create_particles_relations/, cacheing/, full_sim_processing/
  run.sh, fancy_run.sh        run a list of configs sequentially (fancy_run adds a summary)
  run_nopid_bd_pipeline.sh    end-to-end de-contaminated pipeline (cache -> DFEI -> IFT -> val-selected test)
  run_experiment_suite.sh     unattended multi-seed experiment queue (bd/bs channels)
  run_tracktype_ablation.sh   long+down / long+up track-type ablation
  suite_record.py             parse one finished run, append a row to results TSV
  aggregate_suite.py          mean ± std per (experiment, channel) from the suite TSV
  calibrate_tagging_power.py  isotonic/linear calibrated tagging power for one reco CSV
  calibrate_all_configs.py    calibrate every suite run on the same candidate set
  aggregate_calibration.py    aggregate the calibrated results
wmpgnn/
  analysis/                   trainer.py, evaluate.py, evaluate_reco_effiency.py, config handling
  calibration/                calibration mask and decay dictionaries
  data_loader/                chunked / default / PV-associated loaders, class weights
  lightning_module/           DFEI and IFT Lightning modules (+ data modules)
  model/                      HGNN blocks, graph networks (GraphCoder, GN), output transforms
  reconstruction/             decay reconstruction, quantity adders, signal definitions
  reconstruction_data/        reconstruction on cached data
  performance/                plotting, reco accuracy, tagging power
  mass_fit/                   B-mass fit (fit_mass.py, channel_dict.py)
  util/                       pruners, BFS components, PV association
Dataset.md                    dataset description (imported snapshot)
requirements.txt
```

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

For PyTorch Geometric CUDA wheels, `requirements.txt` points to the PyG wheel
index for Torch 2.6.0 and CUDA 12.6.

## Configs

Configs live in `config_files`. Dataset-size-specific experiment configs are in
`config_files/cache_10000`, `cache_50000` and `cache_100000` (each has `train_DFEI*`,
`train_IFT*`, `evaluate_DFEI`, `evaluate_IFT`; the 50k/100k folders also have the DFEI-chain
IFT variants and threshold-grid evaluation configs). The top-level `config_files/*.yaml`
are the default 10k templates. `config_files/cache_200000/` only holds a plan README.

Variants include PV association source (`PVassoTrue`, `PVassoIP`, `PVassoDFEI`,
`PVassoDFEI_DFEI`), DFEI target variants (`b_daughters_only`, `cpt`, `full`) and DFEI-chain
IFT (`train_IFT_DFEIchain_*`, `evaluate_IFT_DFEIchain_*`).

Before running, update `settings.data_dir` and any checkpoint/log path for your
environment. See [config_files/README.md](config_files/README.md) for:

- PID input modes (`use_pid`: `realistic` / `true` / `None`). `true` is an oracle ceiling
  that leaks MC truth and inflates tagging power; use `realistic` (or `None`) for headline results.
- Honest threshold/model selection: choose prune thresholds and model versions with
  `evaluate.split: val`, then report once on `split: tst`.

## Cached dataset creation

Training configs expect cached PyG graph files, not raw ROOT files. For example,
`config_files/cache_50000/train_DFEI.yaml` points to:

```text
data/montecarlo/cache_50000/DFEI_LHCb_LHCbcollision_normed_pt_data/
```

That directory must contain one subdirectory per sample:

```text
00353851_inclusive/
  trn_data_*.pt.zst
  val_data_*.pt.zst
  tst_data_*.pt.zst
  config.yaml
```

Create them from raw LHCb Monte Carlo `*.mc.root` tuples:

```bash
python scripts/cache_montecarlo.py \
  --input-dir data/montecarlo/root \
  --work-dir data/montecarlo_work/cache_50000 \
  --cache-out data/montecarlo/cache_50000 \
  --sample inclusive \
  --identifier 00353851 \
  --max-events 50000
```

The script runs three stages: raw `EvtTuple/Tracks` conversion, particle/relation
ROOT construction, then compressed graph caching. Other useful options:
`--input-glob`, `--chunk-size` (default 1000 events/file), `--max-events-per-file`,
`--mode {default,trn_only,tst_only}`, `--no-norm`, `--overwrite`.

Verify the trainer can see files:

```bash
find data/montecarlo/cache_50000 -type f -name 'trn_data_*' | head
```

If training fails with `ValueError: min() arg is an empty sequence`, the loader
found no `trn_data_*` files at `settings.data_dir/<sample>/`. Check
`settings.data_dir`, `settings.sample`, and the cache output path.

## Training

Run from the repository root:

```bash
python wmpgnn/analysis/trainer.py --config config_files/cache_10000/train_DFEI.yaml
python wmpgnn/analysis/trainer.py --config config_files/cache_10000/train_IFT.yaml
```

IFT configs that use DFEI-based PV association or DFEI-chain input need a trained DFEI
checkpoint: train DFEI first and point the IFT config at its `LHCb_logs/DFEI/version_N`.

Sequential config lists: `scripts/run.sh` (run from `config_files`) or `scripts/fancy_run.sh`
(coloured OK/FAIL summary).

## Evaluation

```bash
python wmpgnn/analysis/evaluate.py --config config_files/cache_10000/evaluate_DFEI.yaml
python wmpgnn/analysis/evaluate.py --config config_files/cache_10000/evaluate_IFT.yaml
```

Outputs go to the log/output paths set in the YAML files (`LHCb_logs/`, git-ignored).
Validation-split outputs are tagged `__val` so they cannot be mistaken for test results.

Reconstruction efficiency and mass fit:

```bash
python wmpgnn/analysis/evaluate_reco_effiency.py --version <LHCb_logs/IFT/version_N> --sample <sample>
python wmpgnn/mass_fit/fit_mass.py --version <LHCb_logs/IFT/version_N> --sample <sample>
```

## Flavour-tagging calibration and experiment suites

Calibrated effective tagging power (per-event mistag calibration, tagger-level and full-chain,
bootstrap errors):

```bash
python scripts/calibrate_tagging_power.py \
  --csv LHCb_logs/IFT/version_N/signal_reco_df_<sample>__<tag>.csv \
  --n-generated 10111 --label my_run --bootstrap 500
```

Full pipelines (written for the GPU server layout: `cd ~/tfg/hgnn`, `/opt/miniconda3/bin/python`;
adjust `PY` and paths for other machines):

- `scripts/run_nopid_bd_pipeline.sh` — `MODE=smoke|full`, `CHANNEL=bd|bs`, `USE_PID=true|None`.
  Re-caches mc24 samples, trains DFEI then IFT, selects the reco threshold on validation and
  reports the final tagging power on test.
- `scripts/run_experiment_suite.sh` — multi-seed queue over architecture/loss/data variants
  (`SEEDS`, `CHANNELS`, `EXPERIMENTS_FILTER`, `GPU_ALLOWED`, …); resumable via `DONE_*` markers.
  Results: `outputs/suite/results.tsv`, summarised with `scripts/aggregate_suite.py` and
  `scripts/calibrate_all_configs.py` + `scripts/aggregate_calibration.py`.
- `scripts/run_tracktype_ablation.sh` — track-type (long/down/up) ablation.

## Dataset notes

Dataset documentation from the imported snapshot is in `Dataset.md`. Large local
datasets, model checkpoints and generated outputs (`data/`, `LHCb_logs/`, `outputs/`) are
git-ignored and not part of the repository.
