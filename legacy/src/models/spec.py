from __future__ import annotations

from dataclasses import dataclass

from legacy.src.data.schema import DEFAULT_HETERO_GRAPH_SCHEMA
from legacy.src.pcer_types import ExperimentConfig, GraphSchema


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    graph_schema: GraphSchema
    latent_size: int
    num_layers: int
    num_message_passing_steps: int
    use_weighted_message_passing: bool
    use_node_pruning: bool
    use_edge_pruning: bool


def build_model_spec(config: ExperimentConfig) -> ModelSpec:
    return ModelSpec(
        name=config.model.name,
        backend=config.model.backend,
        graph_schema=DEFAULT_HETERO_GRAPH_SCHEMA,
        latent_size=config.model.latent_size,
        num_layers=config.model.num_layers,
        num_message_passing_steps=config.model.num_message_passing_steps,
        use_weighted_message_passing=config.model.use_weighted_message_passing,
        use_node_pruning=config.model.use_node_pruning,
        use_edge_pruning=config.model.use_edge_pruning,
    )

