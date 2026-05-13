import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import Dict

from calculation_helper import derive_quantities, match_pvs_by_distance, calculate_ip
from helper import *
from normalization_utils import *

import torch
from torch_geometric.data import HeteroData


# Class creating graphs of the data
class GraphConstruction:
    def __init__(self, dir: str, file: str,
                 node_features=None, edge_features=None, topo_classes=None, identifier=None,
                 use_ft: bool = False, edge_prune: bool = False,
                 configs: Dict = None):
        self.dir = dir
        self.decay = get_decay(file)
        self.start, self.end = get_range(file)
        if identifier is None:
            self.part_file = f'{dir}/{self.decay}/{file}'
            self.rel_file = f'{dir}/{self.decay}/{file.replace("particles", "relations")}'
        else:
            self.part_file = f'{dir}/{identifier}_{self.decay}/{file}'
            self.rel_file = f'{dir}/{identifier}_{self.decay}/{file.replace("particles", "relations")}'

        # If we pass a config file we grab everything from there
        if configs is not None:
            self.configs = obtain_configs(configs=configs)
        else:
            self.configs = obtain_configs(edge_features=edge_features, node_features=node_features,
                                          topo_classes=topo_classes,
                                          use_ft=use_ft, edge_prune=edge_prune,
                                          allBfinal=False, include_neutral=False)

        # If reco_suffix exists we apply selection on reco variables, in principle either everything is reco or none
        self.reco_suffix = '_reco' if any('reco' in feature for feature in node_features) else ''
        self.simulation_type = get_simulation_type(dir)
        self.configs["sim"] = self.simulation_type

        # Defining variables which needs to be load to construct graph
        part_data, rel_data, derive = get_part_rel(self.part_file, self.rel_file,
                                                   node_features, edge_features,
                                                   self.configs, reco_suffix=self.reco_suffix)
        self._process_data(part_data, rel_data, derive)

        # Initializing per event data frames
        self.b_particles = None
        self.final_state_particles, self.signal_final_state_particles = None, None
        self.final_state_relations, self.signal_final_state_relations = None, None
        self.evt_number = None
        self.use_only_signal_particles = False
        self.normalization = None
        self.reset()

    def reset(self):
        self.use_only_signal_particles = False
        # Here we reset the dataframes which are used per single event
        self.final_state_particles = pd.DataFrame()  # list containing all final state particles
        self.final_state_relations = pd.DataFrame()
        self.b_particles = pd.DataFrame()
        self.signal_final_state_particles = pd.DataFrame()
        self.signal_final_state_relations = pd.DataFrame()

    def _process_data(self, part_data, rel_data, to_derive):
        # Filtering and adding quantities which are not present in the root file
        part_data, rel_data = derive_quantities(part_data, rel_data, to_derive)
        res = filter_data(part_data, rel_data, configs=self.configs, reco_suffix=self.reco_suffix)
        self.stable_part, self.rel_data, self.sig_part, self.sig_rel = res


    def get_event(self, evt_number):
        # adjust it here as saveguard of b particles with only one daughter
        self.reset()
        self.evt_number = evt_number

        # B hadrons information
        self.b_particles = self.sig_part[
            self.sig_part['ExpandedEventNumber'] == self.evt_number].copy().reset_index(drop=True)

        # Particles: all stable
        self.final_state_particles = self.stable_part[
            self.stable_part['ExpandedEventNumber'] == self.evt_number].copy().reset_index(drop=True)
        # Particles: only from b decays
        self.signal_final_state_particles = self.final_state_particles[
            self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] > 0].copy().reset_index(drop=True)

        # Relations: all stable
        self.final_state_relations = self.rel_data[
            self.rel_data['ExpandedEventNumber'] == self.evt_number].copy().reset_index(drop=True)
        particle_key_to_list_index = pd.Series(self.final_state_particles.index,
                                               index=self.final_state_particles.ExpandedParticleIndex.values).to_dict()
        self.final_state_relations['list_index_1'] = self.final_state_relations[
            'FirstParticleIndex'].map(particle_key_to_list_index)
        self.final_state_relations['list_index_2'] = self.final_state_relations[
            'SecondParticleIndex'].map(particle_key_to_list_index)
        # Relations: only from b decays
        self.signal_final_state_relations = self.sig_rel[
            self.sig_rel['ExpandedEventNumber'] == self.evt_number].copy().reset_index(drop=True)

        particle_key_to_list_index = pd.Series(self.signal_final_state_particles.index,
                                               index=self.signal_final_state_particles.ExpandedParticleIndex.values).to_dict()
        self.signal_final_state_relations['list_index_1'] = self.signal_final_state_relations[
            'FirstParticleIndex'].map(particle_key_to_list_index)
        self.signal_final_state_relations['list_index_2'] = self.signal_final_state_relations[
            'SecondParticleIndex'].map(particle_key_to_list_index)

    def get_graph(self):
        # Obtain basic graph with tracks and pvs nodes, (tracks to tracks) and (tracks to pvs) edges
        if self.use_only_signal_particles:
            tracks_df = self.signal_final_state_particles
            tracks_to_tracks_df = self.signal_final_state_relations
            #
        else:
            tracks_df = self.final_state_particles
            tracks_to_tracks_df = self.final_state_relations

        data = HeteroData()

        """Track nodes"""
        # split node features by pid and create a second storage
        features = self.configs['node_features']
        real_pid_features = [x for x in features if x.startswith("real_prob")]
        pid_features = [x for x in features if x.startswith("Prob")]
        node_features = [x for x in features if x not in real_pid_features and x not in pid_features]
        data['tracks'].x = torch.tensor(tracks_df[node_features].values, dtype=torch.float32)
        if pid_features:
            data["tracks"].pid = torch.tensor(tracks_df[pid_features].values, dtype=torch.float32)
        if real_pid_features:
            data["tracks"].real_pid = torch.tensor(tracks_df[real_pid_features].values, dtype=torch.float32)

        # 0=b, 1=bkg, 2=bbar and functions as label for node pruning, as shift by + 1 for allowing uint usage
        data['tracks'].ft = torch.tensor(np.array(tracks_df['isfromb']) + 1, dtype=torch.uint8)

        # Events require at least 2 signal tracks
        if len(self.signal_final_state_particles) < 2:
            return 0


        """Track to track edges"""  # Only 1 directional stored, bidirectional during loading to cpu memory
        edge_features = torch.tensor(tracks_to_tracks_df[self.configs['edge_features']].to_numpy(np.float32))
        data['tracks', 'tracks'].edges = edge_features

        senders = torch.tensor(tracks_to_tracks_df['list_index_1'].values)
        receivers = torch.tensor(tracks_to_tracks_df['list_index_2'].values)
        data['tracks', 'tracks'].edge_index = torch.vstack([senders, receivers]).to(torch.int32)
        labels = torch.from_numpy(np.argmax(tracks_to_tracks_df[self.configs['topo_classes']].to_numpy(), axis=1))
        data['tracks', 'tracks'].y = labels.to(torch.uint8)

        """PVs nodes"""
        pv_features = [f'xPV{self.reco_suffix}', f'yPV{self.reco_suffix}', f'zPV{self.reco_suffix}']
        self.configs["pv_features"] = pv_features
        pvs = torch.from_numpy(tracks_df[pv_features].to_numpy()).to(torch.float32)
        data['pvs'].x = torch.unique(pvs, dim=0).to(torch.float32)

        """Tracks to PVs edges"""
        # Two problems:
        # 1. Ghost tracks do not have a pv
        # 2. Tracks true pv can be not reconstructed
        # Solution: Match true pv to reco pv, if true pv is not recoed, remove from pv association inference
        # this does not prevent the model from not doing any decision
        # TODO: Check how much it affects signal tracks (primarily for the second point)

        # Match true pv to the reco pv (distance <= 2)
        pv_indices = torch.from_numpy(match_pvs_by_distance(tracks_df))
        # Calculate the ip as edge feature
        ips = calculate_ip(tracks_df, reco_suffix=self.reco_suffix, log_ip=True)
        data['tracks', 'pvs'].edges = ips.flatten().unsqueeze(-1).to(torch.float32)

        permutations = torch.cartesian_prod(torch.arange(pvs.shape[0]), torch.arange(data['pvs'].x.shape[0]))
        data['tracks', 'pvs'].edge_index = permutations.to(torch.int32).T

        # Get the true value
        num_classes = data['pvs'].x.shape[0]
        one_hot = torch.nn.functional.one_hot(pv_indices.clamp(min=0), num_classes=num_classes)
        one_hot[pv_indices == -1] = 0
        data['tracks', 'pvs'].y = one_hot.to(torch.uint8).flatten().unsqueeze(-1)
        one_hot[pv_indices != -1] = 1
        data['tracks', 'pvs'].filter = one_hot.to(torch.uint8).flatten()

        '''Global features'''
        # global feature which contains the num of tracks and nPVs
        data["globals"].x = torch.tensor([data["tracks"].x.shape[0], data["pvs"].x.shape[0]]).unsqueeze(0).to(
            torch.float32)
        return data

    def normalize_graph(self, data):
        # The current normalization only performs normalization for track nodes and track to track edges
        if self.normalization is None:
            normalization_dict = obtain_normalization(self.part_file.replace(self.decay, "inclusive"),
                                                      self.configs)
            self.normalization = adjust_normalization(normalization_dict, self.configs)

        features = self.configs['node_features']
        real_pid_features = [x for x in features if x.startswith("real_prob")]
        pid_features = [x for x in features if x.startswith("Prob")]
        node_features = [x for x in features if x not in real_pid_features and x not in pid_features]

        node_norm = torch.tensor(self.normalization.loc[node_features].values).T
        data["tracks"].x = ((data["tracks"].x - node_norm[0]) / node_norm[1]).to(torch.float32)

        # normalize the pid features of nodes if exists
        if pid_features:
            pid_norm = torch.tensor(self.normalization.loc[pid_features].values).T
            data["tracks"].pid = ((data["tracks"].pid - pid_norm[0]) / pid_norm[1]).to(torch.float32)
        # realistic pid response
        if real_pid_features:
            real_pid_norm = torch.tensor(self.normalization.loc[real_pid_features].values).T
            data["tracks"].real_pid = ((data["tracks"].real_pid - real_pid_norm[0]) / real_pid_norm[1]).to(
                torch.float32)

        # normalize the track to track edges
        edge_norm = torch.tensor(self.normalization.loc[self.configs["edge_features"]].values).T
        data[("tracks", "tracks")].edges = ((data[("tracks", "tracks")].edges - edge_norm[0]) / edge_norm[1]).to(
            torch.float32)

        # Manually norm some quantities
        data["globals"].x  = data["globals"].x / torch.tensor([100, 10])
        data["pvs"].x = data["pvs"].x / torch.tensor([1, 1, 100])

        return data

    def add_particle_info(self, data, particle_keys, node=None):
        if self.use_only_signal_particles:
            tracks_df = self.signal_final_state_particles
        else:
            tracks_df = self.final_state_particles
        for key, feature in particle_keys.items():
            value = torch.from_numpy(tracks_df[feature].values)
            if node is None:
                data[key] = value
            else:
                data[node][key] = value
        return data

    def add_relation_info(self, data, particle_keys, node=None):
        if self.use_only_signal_particles:
            tracks_to_tracks_df = self.signal_final_state_relations
        else:
            tracks_to_tracks_df = self.final_state_relations
        for key, feature in particle_keys.items():
            value = torch.from_numpy(tracks_to_tracks_df[feature].values)
            if node is None:
                data[key] = value
            else:
                data[node][key] = value
        return data

    def get_true_graph(self, data):
        true_data = self.get_graph()
        if true_data == 0:
            return 0
        # here only 1 directional as well
        data['truth_senders'] = true_data[("tracks", "to", "tracks")].edge_index[0]
        data['truth_receivers'] = true_data[("tracks", "to", "tracks")].edge_index[1]
        data['truth_y'] = true_data[("tracks", "to", "tracks")].y  # here we need to expand later on
        return data

    def add_evt_run_number(self, data):
        EVENTNUMBER = np.unique(self.final_state_particles["EVENTNUMBER"])
        RUNNUMBER = np.unique(self.final_state_particles["RUNNUMBER"])

        if len(EVENTNUMBER) != 1 or len(RUNNUMBER) != 1:
            import pdb;
            pdb.set_trace()
        data["EVENTNUMBER"] = torch.from_numpy(EVENTNUMBER).to(torch.int)
        data["RUNNUMBER"] = torch.from_numpy(RUNNUMBER).to(torch.int)
        return data
