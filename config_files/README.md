# Config Layout

Dataset-specific configs live in folders named after the cache dataset:

```text
config_files/cache_10000/
```

Run a config with:

```bash
python wmpgnn/analysis/trainer.py --config config_files/cache_10000/train_DFEI.yaml
python wmpgnn/analysis/evaluate.py --config config_files/cache_10000/evaluate_DFEI.yaml
```

For a larger cache, copy `cache_10000` to a new folder and update only
`settings.data_dir`, `settings.nfiles`, and `evaluate.nfiles`.

Example:

```bash
cp -r config_files/cache_10000 config_files/cache_50000
```

Then point `data_dir` to:

```text
data/montecarlo/cache_50000/DFEI_LHCb_LHCbcollision_normed_pt_data/
```
