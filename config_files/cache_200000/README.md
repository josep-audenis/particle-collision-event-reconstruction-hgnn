# cache_200000

Generated experiment configs for the 200k-cache run should be written here by:

```bash
scripts/run_cache200000_experiments.sh
```

The wrapper starts from `config_files/cache_100000` templates, replaces:

- `cache_100000` -> `cache_200000`
- `nfiles: [ 100 ]` -> `nfiles: [ 200 ]`
- PV/DFEI/IFT model versions detected from freshly created `LHCb_logs` versions

Main planned chain:

1. Build `data/montecarlo/cache_200000`.
2. Audit cache.
3. Train fresh DFEI.
4. Train IFT truth/PV control using fresh DFEI for PV association.
5. Evaluate truth/PV control.
6. Train/evaluate real DFEI-chain IFT.
7. Evaluate real DFEI-chain with tuned threshold `edge=0.4`, `node=0.5`.
