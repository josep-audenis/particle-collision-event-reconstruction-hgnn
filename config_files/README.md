# Config Layout

Dataset-specific configs live in folders named after the cache dataset:

```text
config_files/cache_10000/
config_files/cache_50000/
config_files/cache_100000/
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

## PID inputs: realistic vs true (leakage)

`use_pid` selects the particle-ID input:

- `"realistic"` — momentum-conditioned sampled detector response (`real_prob_*`).
  **Use this for headline results.**
- `"true"` — true Monte-Carlo PID (`Prob_*`). Perfect species ID an experiment
  never has → inflated tagging power. **Oracle ceiling only**, label it as such.
- `"None"` — no PID input.

Both `Prob_*` and `real_prob_*` are cached (see `cacheing/main.py` node features),
so switching does not require re-caching once the cache was built with both. The
lightning modules emit a `CONTAMINATION` warning whenever `use_pid: "true"` is used.

## Honest threshold / model selection (no test-set leakage)

Prune thresholds (`inference.edge_prune_thr`, `inference.node_prune_thr`) and the
choice between model versions must be selected on the **validation** split, then
reported **once** on the test split. Selecting them on the test set (running the
threshold grid on `tst` and picking the best) optimistically biases the result.

Set the eval split in the config's `evaluate` block:

```yaml
evaluate:
  sample: [ "00353851_inclusive" ]
  nfiles: [ 10 ]
  over_write: ""
  split: val   # "val" for selection sweeps, "tst" (default) for the final number
```

Validation-split outputs are auto-tagged with `__val` in their filenames so they
can never overwrite or be mistaken for the final test results. Protocol:

1. Sweep thresholds / versions with `split: val`; compare the `__val` outputs.
2. Fix the winning thresholds + version.
3. Run that single config once with `split: tst` (default) → report this number.

## Event counts

The Monte Carlo cache pipeline writes chunks of 1000 events by default. The
training/evaluation configs therefore use:

| Config folder | Events | `nfiles` |
| --- | ---: | ---: |
| `cache_10000` | 10,000 | 10 |
| `cache_50000` | 50,000 | 50 |
| `cache_100000` | 100,000 | 100 |

## Size estimate

No generated `data/montecarlo/cache_10000` directory is present in this checkout,
so cached graph size cannot be measured directly yet. Local raw ROOT files in
`data/montecarlo/root` average about 10.1 KB/event:

| Events | Raw input estimate |
| ---: | ---: |
| 10,000 | 96 MiB |
| 50,000 | 482 MiB |
| 100,000 | 964 MiB |

Actual `.pt.zst` graph cache size depends on graph density and compression.
After building `cache_10000`, measure it and scale linearly as a first estimate:

```powershell
(Get-ChildItem -Recurse data\montecarlo\cache_10000 | Measure-Object Length -Sum).Sum / 1MB
```
