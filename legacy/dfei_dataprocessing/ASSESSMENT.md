# DFEI Dataprocessing Assessment

## What this legacy code does

The CERN/DFEI preprocessing is a three-stage pipeline:

1. `full_sim_processing/new_main.py`
   - Reads raw `EvtTuple/Tracks` from LHCb ROOT files.
   - Rebuilds particle and mother tables.
   - Adds physics-derived labels such as `PrimaryHeavyHadronIndex` and `NumVisDaugh`.
   - Writes intermediate `GeneratedParticles_*.root` files.

2. `create_particles_relations/main.py`
   - Converts the intermediate particle table into event-wise particle and relation trees.
   - Computes track-pair truth and geometry quantities such as:
     - `FromSameAssociatedPV_reco`
     - `FromSameHeavyHadron`
     - `LCA`, `TopoLCA`
     - `theta_reco`, `trdist_reco`, `delta_z0_reco`, `log_DOCA_reco`
   - Carries detector-side inputs such as PID scores and ghost probability.

3. `data_cacheing/main.py`
   - Builds PyG `HeteroData` graphs.
   - Separates track node features, PID features, track-track edges, track-PV edges, PV nodes, and globals.
   - Normalizes features and serializes compressed graph batches.

## Features that look important

The legacy cacheing stage uses:

- Track node features:
  - `px_reco`, `py_reco`, `pz_reco`
  - `xProd_reco`, `yProd_reco`, `zProd_reco`
  - `charge`, `ghost_prob`
- Track PID features:
  - `Prob_k`, `Prob_e`, `Prob_mu`, `Prob_p`, `Prob_pi`
- Track-track edge features:
  - `FromSameAssociatedPV_reco`
  - `theta_reco`
  - `trdist_reco`
  - `delta_z0_reco`
  - `log_DOCA_reco`
- Track-PV edge feature:
  - log impact parameter to each reconstructed PV
- Global features:
  - number of tracks
  - number of PVs

These are richer than the current repo's ROOT converter.

## What the current repo does instead

`wmpgnn/datasets/root_converter.py` currently skips the full three-stage DFEI pipeline and builds graphs directly from `EvtTuple/Tracks`.

Current serialized node tensor:

- `xProd_reco`, `yProd_reco`, `zProd_reco`
- `px_reco`, `py_reco`, `pz_reco`
- `xPV_reco`, `yPV_reco`, `zPV_reco`
- `Charge`
- `xPV`, `yPV`, `zPV`

Current serialized edge tensor:

- same-PV flag from exact reconstructed PV match
- opening angle
- transverse distance proxy
- `delta_z`

Current edge supervision:

- 4-way one-hot label from first common ancestor order, clamped to classes `0..3`

So the current repo keeps the broad graph idea, but it is materially simpler than the DFEI reference.

## What we can reuse safely

Good candidates to adopt or align with:

- The feature definitions in `data_cacheing/main.py`
- The relation construction logic in `create_particles_relations/create_events_helper.py`
- The PV-association idea used for `FromSameAssociatedPV_reco`
- The richer edge geometry features, especially `log_DOCA_reco`
- The explicit PID and ghost-probability channels
- The split between track nodes, PV nodes, track-track edges, and track-PV edges

## What we should not reuse verbatim

This legacy code is useful as reference, but not as drop-in production code:

- It assumes intermediate ROOT formats that our current converter does not produce.
- It has hardcoded EOS output paths.
- `create_particles_relations/main.py` still contains `pdb.set_trace()`.
- `data_cacheing/main.py` writes the same buffered graphs to both `tst_data_*` and `val_data_*`.
- Some scripts are tightly coupled to LHCb/CERN filesystem conventions and local branch naming.

## Recommended integration path

Short term:

- Keep the current repo's direct ROOT-to-graph flow.
- Use the DFEI code as the source of truth for feature semantics.
- Extend `wmpgnn/datasets/root_converter.py` to add the missing high-value inputs:
  - PID scores
  - ghost probability
  - `log_DOCA_reco`
  - explicit associated-PV relation feature

Medium term:

- Decide whether to keep the current 4-class ancestor target or move closer to the richer `LCA`/`TopoLCA` supervision from DFEI.
- Decide whether the training pipeline should consume:
  - the existing serialized homogeneous format, or
  - a more direct heterogeneous graph format closer to `data_cacheing/class_holder.py`

## Bottom line

The most valuable part of `legacy/dfei_dataprocessing` is not the exact scripts. It is the physics-informed feature engineering and relation-building logic.

That means:

- keep the current repo architecture where it is simpler,
- borrow the DFEI feature definitions and labels where they improve fidelity,
- avoid copying the legacy scripts as-is.
