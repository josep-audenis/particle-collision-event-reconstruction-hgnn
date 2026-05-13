import numpy as np
import torch
from torch_geometric.data import HeteroData

from helper import *
from calculation_helper import *


class GraphCache:
    def __init__(self, file: str, branch: str,
                 features: dict[str, list[str]],
                 use_long_only: bool
                 ) -> None:
        self.features = features
        self.feature_df, self.pv_df, self.identifier_df = get_part_df(file, branch, features, use_long_only)
        # add a cut on ghost prob
        self.nevts = len(self.identifier_df)
        self.normalization = None

    def process_data(self, evt):
        stable_df = self.feature_df.loc[evt]
        pv_df = self.pv_df.loc[evt]
        stable_df, pv_df = apply_cuts(stable_df, pv_df)

        rel_df = create_relation(stable_df)

        identifier_df = self.identifier_df.loc[evt]

        # add a cut on ghost prob of 0.6 and adjust the pv


        data = HeteroData()

        """Track nodes"""
        node_features = self.features['node_features']
        pid_features = self.features['pid_features']
        data['tracks'].x = torch.tensor(stable_df[node_features].values, dtype=torch.float32)
        data["tracks"].pid = torch.tensor(stable_df[pid_features].values, dtype=torch.float32)

        """Track to track edges"""
        edge_features = self.features['edge_features']
        edge_features = torch.tensor(rel_df[edge_features].to_numpy(np.float32))
        data['tracks', 'tracks'].edges = edge_features

        senders = torch.tensor(rel_df['FirstParticleIndex'].values)
        receivers = torch.tensor(rel_df['SecondParticleIndex'].values)
        data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers]).to(torch.int32)
        """PVs nodes"""
        pv_features = ['xPV_reco', 'yPV_reco', 'zPV_reco']
        pvs = torch.from_numpy(pv_df[pv_features].to_numpy()).to(torch.float32)
        data['pvs'].x = torch.unique(pvs, dim=0).to(torch.float32)

        """Tracks to PVs edges"""
        ips = calculate_ip(stable_df, pv_df, reco_suffix="_reco", log_ip=True)
        data['tracks', 'pvs'].edges = ips.flatten().unsqueeze(-1).to(torch.float32)
        permutations = torch.cartesian_prod(torch.arange(data['tracks'].x.shape[0]), torch.arange(data['pvs'].x.shape[0]))
        data['tracks', 'pvs'].edge_index = permutations.to(torch.int32).T

        '''Global features'''
        # global feature which contains the num of tracks and nPVs
        data["globals"].x = torch.tensor([data["tracks"].x.shape[0], data["pvs"].x.shape[0]]).unsqueeze(0).to(
            torch.float32)

        data["EVENTNUMBER"] = torch.tensor([int(identifier_df["EVENTNUMBER"])])
        data["RUNNUMBER"] = torch.tensor([int(identifier_df["RUNNUMBER"])])

        return data

    def normalize_graph(self, data):
        # The current normalization only performs normalization for track nodes and track to track edges
        if self.normalization is None:
            normalization_dict = torch.load("../cacheing/normalization_dict.pt", weights_only=False)
            self.normalization = adjust_normalization(normalization_dict, self.features)


        node_norm = torch.tensor(self.normalization.loc[self.features["node_features"]].values).T
        data["tracks"].x = ((data["tracks"].x - node_norm[0]) / node_norm[1]).to(torch.float32)
        pid_norm = torch.tensor(self.normalization.loc[self.features["pid_features"]].values).T
        data["tracks"].pid = ((data["tracks"].pid - pid_norm[0]) / pid_norm[1]).to(torch.float32)


        # normalize the track to track edges
        edge_norm = torch.tensor(self.normalization.loc[self.features["edge_features"]].values).T
        data[("tracks", "tracks")].edges = ((data[("tracks", "tracks")].edges - edge_norm[0]) / edge_norm[1]).to(
            torch.float32)

        # Manually norm some quantities
        data["globals"].x = data["globals"].x / torch.tensor([100, 10])
        data["pvs"].x = data["pvs"].x / torch.tensor([1, 1, 100])

        return data
