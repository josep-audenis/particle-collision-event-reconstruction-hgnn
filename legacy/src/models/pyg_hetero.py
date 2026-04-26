from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import HeteroConv, SAGEConv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The PyG backend requires torch and torch_geometric to be installed."
        ) from exc
    return torch, nn, HeteroConv, SAGEConv


class ParticleCollisionHeteroGNN:
    @staticmethod
    def create(
        track_input_dim: int,
        pv_input_dim: int,
        track_edge_dim: int,
        pv_edge_dim: int,
        latent_size: int,
        num_layers: int,
        use_node_pruning: bool,
        use_edge_pruning: bool,
    ):
        torch, nn, HeteroConv, SAGEConv = _require_torch()

        class MLP(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> None:
                super().__init__()
                layers = []
                current_dim = input_dim
                for _ in range(max(depth - 1, 0)):
                    layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
                    current_dim = hidden_dim
                layers.append(nn.Linear(current_dim, output_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        class MessagePassingBlock(nn.Module):
            def __init__(self, hidden_dim: int) -> None:
                super().__init__()
                self.conv = HeteroConv(
                    {
                        ("tracks", "to", "tracks"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("tracks", "to", "pvs"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                        ("pvs", "rev_to", "tracks"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    },
                    aggr="sum",
                )
                self.track_norm = nn.LayerNorm(hidden_dim)
                self.pv_norm = nn.LayerNorm(hidden_dim)
                self.track_edge_gate = MLP(hidden_dim * 3, 1, hidden_dim, 2)
                self.pv_edge_gate = MLP(hidden_dim * 3, 1, hidden_dim, 2)
                self.track_node_gate = MLP(hidden_dim, 1, hidden_dim, 2)

            def forward(self, x_dict, edge_index_dict, edge_attr_dict):
                updated = self.conv(x_dict, edge_index_dict)
                x_dict = {
                    "tracks": self.track_norm(x_dict["tracks"] + updated["tracks"]),
                    "pvs": self.pv_norm(x_dict["pvs"] + updated["pvs"]),
                }

                edge_scores = {}
                track_track_index = edge_index_dict[("tracks", "to", "tracks")]
                tt_source = x_dict["tracks"][track_track_index[0]]
                tt_target = x_dict["tracks"][track_track_index[1]]
                tt_features = torch.cat((tt_source, tt_target, edge_attr_dict[("tracks", "to", "tracks")]), dim=-1)
                edge_scores[("tracks", "to", "tracks")] = self.track_edge_gate(tt_features)

                track_pv_index = edge_index_dict[("tracks", "to", "pvs")]
                tp_source = x_dict["tracks"][track_pv_index[0]]
                tp_target = x_dict["pvs"][track_pv_index[1]]
                tp_features = torch.cat((tp_source, tp_target, edge_attr_dict[("tracks", "to", "pvs")]), dim=-1)
                edge_scores[("tracks", "to", "pvs")] = self.pv_edge_gate(tp_features)

                node_scores = {"tracks": self.track_node_gate(x_dict["tracks"])}
                return x_dict, node_scores, edge_scores

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.track_encoder = MLP(track_input_dim, latent_size, latent_size, 2)
                self.pv_encoder = MLP(pv_input_dim, latent_size, latent_size, 2)
                self.track_edge_encoder = MLP(track_edge_dim, latent_size, latent_size, 2)
                self.pv_edge_encoder = MLP(pv_edge_dim, latent_size, latent_size, 2)
                self.blocks = nn.ModuleList(MessagePassingBlock(latent_size) for _ in range(num_layers))

                self.track_origin_head = MLP(latent_size, 1, latent_size, 2)
                self.same_b_head = MLP(latent_size * 3, 1, latent_size, 2)
                self.pv_assignment_head = MLP(latent_size * 3, 1, latent_size, 2)
                self.lcag_head = MLP(latent_size * 3, 4, latent_size, 2)

                self.use_node_pruning = use_node_pruning
                self.use_edge_pruning = use_edge_pruning

            def _edge_features(self, x_dict, edge_index, edge_attr, src_type: str, dst_type: str):
                src = x_dict[src_type][edge_index[0]]
                dst = x_dict[dst_type][edge_index[1]]
                return torch.cat((src, dst, edge_attr), dim=-1)

            def forward(self, data):
                x_dict = {
                    "tracks": self.track_encoder(data["tracks"].x),
                    "pvs": self.pv_encoder(data["pvs"].x),
                }
                edge_attr_dict = {
                    ("tracks", "to", "tracks"): self.track_edge_encoder(data["tracks", "to", "tracks"].edge_attr),
                    ("tracks", "to", "pvs"): self.pv_edge_encoder(data["tracks", "to", "pvs"].edge_attr),
                    ("pvs", "rev_to", "tracks"): self.pv_edge_encoder(data["pvs", "rev_to", "tracks"].edge_attr),
                }
                edge_index_dict = {
                    ("tracks", "to", "tracks"): data["tracks", "to", "tracks"].edge_index,
                    ("tracks", "to", "pvs"): data["tracks", "to", "pvs"].edge_index,
                    ("pvs", "rev_to", "tracks"): data["pvs", "rev_to", "tracks"].edge_index,
                }

                block_outputs = []
                for block in self.blocks:
                    x_dict, node_scores, edge_scores = block(x_dict, edge_index_dict, edge_attr_dict)
                    block_outputs.append({"node_scores": node_scores, "edge_scores": edge_scores})

                track_track_features = self._edge_features(
                    x_dict,
                    edge_index_dict[("tracks", "to", "tracks")],
                    edge_attr_dict[("tracks", "to", "tracks")],
                    "tracks",
                    "tracks",
                )
                track_pv_features = self._edge_features(
                    x_dict,
                    edge_index_dict[("tracks", "to", "pvs")],
                    edge_attr_dict[("tracks", "to", "pvs")],
                    "tracks",
                    "pvs",
                )

                return {
                    "track_origin_logits": self.track_origin_head(x_dict["tracks"]),
                    "same_b_hadron_logits": self.same_b_head(track_track_features),
                    "pv_assignment_logits": self.pv_assignment_head(track_pv_features),
                    "lcag_logits": self.lcag_head(track_track_features),
                    "block_outputs": block_outputs,
                    "uses_node_pruning": self.use_node_pruning,
                    "uses_edge_pruning": self.use_edge_pruning,
                }

        return Model()
