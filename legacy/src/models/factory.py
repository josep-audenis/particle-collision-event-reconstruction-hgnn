from __future__ import annotations

from legacy.src.models.pyg_hetero import ParticleCollisionHeteroGNN
from legacy.src.models.spec import ModelSpec, build_model_spec
from legacy.src.pcer_types import ExperimentConfig, ModelBuildResult


def build_model(config: ExperimentConfig) -> ModelBuildResult:
    spec: ModelSpec = build_model_spec(config)

    if spec.backend == "pytorch_geometric":
        model = ParticleCollisionHeteroGNN.create(
            track_input_dim=len(spec.graph_schema.track_features),
            pv_input_dim=len(spec.graph_schema.pv_features),
            track_edge_dim=len(spec.graph_schema.track_to_track_features),
            pv_edge_dim=len(spec.graph_schema.track_to_pv_features),
            latent_size=spec.latent_size,
            num_layers=spec.num_message_passing_steps,
            use_node_pruning=spec.use_node_pruning,
            use_edge_pruning=spec.use_edge_pruning,
        )
        parameter_count = None
        if hasattr(model, "parameters"):
            try:
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
            except Exception:
                parameter_count = None

        return ModelBuildResult(
            model=model,
            backend=spec.backend,
            parameter_count=parameter_count,
            notes=("PyG hetero model initialized.",),
        )

    if spec.backend == "tensorflow_graph_nets":
        raise NotImplementedError(
            "The TensorFlow Graph Nets backend has not been ported yet."
        )

    raise ValueError(f"Unsupported backend: {spec.backend}")
