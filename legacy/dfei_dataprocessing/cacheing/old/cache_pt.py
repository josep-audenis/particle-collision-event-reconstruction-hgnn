import numpy as np
import pandas as pd

import torch
from torch_geometric.data import HeteroData

from DecayChainReconstruction import *
from pid_helper import *


def find_row_indices(t1, t2):
    # Expand t1 and t2 to compare every row in t1 with every row in t2
    t1_expanded = t1.unsqueeze(1)  # Shape (n1, 1, cols)
    t2_expanded = t2.unsqueeze(0)  # Shape (1, n2, cols)

    # Check for equality along the last dimension
    matches = (t1_expanded == t2_expanded).all(dim=2)  # Shape (n1, n2)

    # Convert matches to an appropriate type for argmax
    matches_int = matches.int()  # Convert to int

    # Find the indices in t2 for each row in t1
    try:
        indices = torch.argmax(matches_int, dim=1)
    except:
        import pdb;
        pdb.set_trace()

    # Verify matches exist (if no match, indices would be arbitrary)
    valid = matches.any(dim=1)
    indices[~valid] = -1  # Set unmatched rows to -1

    # Number of classes (largest value in the tensor + 1)
    num_classes = t2.shape[0]

    # One-hot encoding
    one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return indices, one_hot_encoded


def hgnn_caching(graph, graph_target, norm_graph=False, perf_mode=False, ft_bool=False, real_pid=False, pid_map=None):
    labels = np.array(graph_target["edges"])
    indices = np.unique(graph['receivers'])

    remapping = {a: i for a, i in zip(indices, list(range(0, len(indices))))}
    senders = torch.from_numpy(np.array([remapping[x] for x in graph["senders"]])).long()
    receivers = torch.from_numpy(np.array([remapping[x] for x in graph["receivers"]])).long()

    new_nodes = torch.from_numpy(graph["nodes"][indices])
    new_edges = torch.from_numpy(graph['edges'])

    # recoPVs = torch.unique(new_nodes[:, 12:15], dim=0)
    recoPVs = torch.unique(new_nodes[:, -3:], dim=0)
    nPVs = recoPVs.shape[0]

    true_nodes_PVs = new_nodes[:, 12:15]
    # true_nodes_PVs = new_nodes[:, -3:]  # old one
    if true_nodes_PVs.numel() == 0:
        print("error")
        return 0

    y, y_one_hot = find_row_indices(true_nodes_PVs, recoPVs)

    xyz = new_nodes[:, :3]
    P = new_nodes[:, 3:6]
    xyz_repeated = xyz.unsqueeze(1).repeat(1, recoPVs.shape[0], 1)
    P_repeated = P.unsqueeze(1).repeat(1, recoPVs.shape[0], 1)
    r = xyz_repeated - recoPVs
    IPs = torch.sqrt(
        torch.sum(r ** 2, dim=-1) - torch.sum(P_repeated * r, dim=-1) ** 2 / torch.sum(P_repeated ** 2, dim=-1))

    if torch.isnan(IPs).any().item():
        IPs[torch.isnan(IPs)] = torch.max(IPs[~torch.isnan(IPs)]).item()

    permutations = torch.cartesian_prod(torch.arange(true_nodes_PVs.shape[0]), torch.arange(recoPVs.shape[0]))
    data = HeteroData()

    # Removing the pv information from the node
    # new_nodes = torch.hstack([new_nodes[:, :6], new_nodes[:, 9:10]])  # old one
    new_nodes = new_nodes[:, :12]

    # Node features
    features = new_nodes[:, :7]
    pid_features = new_nodes[:, 7:12]

    if real_pid:
        corr_data, inter_data = pid_map  # make this to a class so only load it once
        p = torch.sqrt(new_nodes[:, 3] ** 2 + new_nodes[:, 4] ** 2 + new_nodes[:, 5] ** 2)
        pid = torch.argmax(pid_features, dim=1)
        prob_vars = ['k', 'e', 'mu', 'p', 'pi']
        df = pd.DataFrame({"p": p, "pid": pid, "Prob_k": 0., "Prob_e": 0., "Prob_mu": 0., "Prob_p": 0., "Prob_pi": 0.})

        for i, prob in enumerate(prob_vars):
            selbool = df["pid"] == i
            if np.sum(selbool) == 0:
                continue
            new_prob = get_prob_at_p(df[selbool]["p"], inter_data[prob], corr_data[prob])
            df.loc[selbool, ['p', 'Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi']] = new_prob.values

        pid_features = torch.from_numpy(df[['Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi']].values)

    if norm_graph:  # normalization
        # origin x, y, z, momentum x, y, z
        median = torch.tensor([-0.0281298, -0.0201243, 8.34071, 0.258748, -0.427731, 5.61396, 0])
        iqr = torch.tensor([5.28475, 5.10537, 143.23, 122.845, 147.825, 11.2696, 1])
        features = (features - median) / iqr

    data['tracks'].x = features
    data["tracks"].pid = pid_features

    if norm_graph:
        median = torch.tensor([-0.00185994, 0.000301253, 1.20331])
        iqr = torch.tensor([0.0500673, 0.0496163, 136.995])
        recoPVs = (recoPVs - median) / iqr

    data['pvs'].x = recoPVs
    data['globals'].x = torch.hstack(
        [torch.from_numpy(graph["globals"]), torch.tensor(nPVs, dtype=torch.float32)]).unsqueeze(0)

    if data["tracks"].x.shape[0] < 255:
        data['tracks', 'pvs'].edge_index = permutations.to(torch.uint8).T
    else:
        data['tracks', 'pvs'].edge_index = permutations.to(torch.uint16).T
    data['tracks', 'pvs'].y = y_one_hot.to(torch.uint8).flatten().unsqueeze(-1)
    data['tracks', 'pvs'].edges = IPs.flatten().unsqueeze(-1)

    if norm_graph:
        median = torch.tensor([0, 0.121977, 10.0289, 18.3388])
        iqr = torch.tensor([1, 0.118058, 30.8912, 559.72])
        new_edges = (new_edges - median) / iqr
    if data["tracks"].x.shape[0] < 255:
        data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers]).to(torch.uint8)
    else:
        data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers]).to(torch.uint16)
    data['tracks', 'tracks'].y = torch.from_numpy(labels).argmax(dim=1).to(torch.uint8)
    data['tracks', 'tracks'].edges = new_edges


    # Ground truth for the FT
    if ft_bool:
        data['tracks'].frag = torch.tensor(graph["fragmentation"], dtype=torch.int32)
        data['tracks'].ft = torch.tensor(graph['isfromb'] + 1, dtype=torch.uint8)

    if perf_mode:

        # bdt relic
        # data['init_senders'] = torch.from_numpy(graph["init_y"]["senders"]).long()
        # data['init_receivers'] = torch.from_numpy(graph["init_y"]["receivers"]).long()
        # data['init_y'] = torch.from_numpy(graph["init_y"]["edges"])
        # data['init_keys'] = torch.from_numpy(graph["init_keys"])
        # data['init_moth_ids'] = torch.from_numpy(graph["init_ids"])
        # data['init_partids'] = torch.from_numpy(graph["init_part_ids"])
        data['final_keys'] = torch.from_numpy(graph["keys"])
        data['moth_ids'] = torch.from_numpy(graph["ids"])
        data['part_ids'] = torch.from_numpy(graph["part_ids"])
        data['lca_chain'] = torch.from_numpy(graph["lca_chain"])
        data['truth_senders'] = torch.from_numpy(graph["truth_senders"]).long()
        data['truth_receivers'] = torch.from_numpy(graph["truth_receivers"]).long()
        data['truth_y'] = torch.from_numpy(graph["truth_y"])
        data['truth_moth_ids'] = torch.from_numpy(graph["truth_ids"])
        data['truth_part_ids'] = torch.from_numpy(graph["truth_part_ids"])
        data['truth_part_keys'] = torch.from_numpy(graph["truth_part_keys"])
        data["tracks"].asso_hh = torch.from_numpy(graph["primariy_idx"])
        data["tracks"].nfinals = torch.from_numpy(graph["nfinals"])
        import pdb; pdb.set_trace()
        """
        init_senders = data.init_keys[data.init_senders]
        init_receivers = data.init_keys[data.init_receivers]
        senders =  data.final_keys[torch.from_numpy(old_senders).long()]
        receivers =  data.final_keys[torch.from_numpy(old_receivers).long()]
        init_cantor = 0.5 * (init_senders + init_receivers - 2) * (
                    init_senders + init_receivers - 1) + init_senders
        final_cantor = 0.5 * (senders + receivers - 2) * (senders + receivers - 1) + senders
        data['old_y'] = data.init_y[~torch.isin(init_cantor, final_cantor)]
        """
    return data
