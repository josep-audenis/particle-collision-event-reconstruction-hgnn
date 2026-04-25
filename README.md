# particle-collision-event-reconstruction-hgnn

This repository is being built as a reimplementation of the HGNN approach from
the paper `Scalable Multi-Task Learning for Particle Collision Event Reconstruction
with Heterogeneous Graph Neural Networks`, while keeping the structure and coding
style closer to the existing `deep-full-event-interpretation-gnn` project.

Current scope:
- config-driven experiment layout under `src/`
- typed paper-specific schema and task definitions
- a training entrypoint that validates the experiment surface with `--dry-run`

Current status:
- the repo backbone is in place
- the actual dataset adapter and HGNN backend still need to be ported

Starter command:

```bash
python -m src.training.train_hgnn --config src/training/configs/small_hetero_hgnn.yaml --dry-run
```
