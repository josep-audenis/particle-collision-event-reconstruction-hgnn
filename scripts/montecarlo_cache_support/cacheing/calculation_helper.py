import numpy as np
import torch
import pandas as pd
from create_pid_response import *


def get_derive_base_quant(to_derive):

    _to_derive = {k: v for k, v in to_derive.items()}
    part_base_quant = []
    rel_base_quant = []
    """track node features"""
    if _to_derive['part']:
        if any("real_prob" in p for p in _to_derive["part"]):
            part_base_quant.append(["px_reco", "py_reco", "pz_reco"])
            part_base_quant.append([f"Prob_{x.split('_')[-1]}" for x in _to_derive["part"] if "real_prob" in x])
            _to_derive["part"] = [x for x in _to_derive["part"] if "real_prob" not in x]
        if "ghost_prob" in _to_derive['part']:
            part_base_quant.append(["Prob_ghost"])
            _to_derive["part"] = [x for x in _to_derive["part"] if "ghost_prob" != x]

    "track to track edge features"
    if _to_derive['rel']:
        if "log_DOCA_reco" in _to_derive["rel"]:
            rel_base_quant.append(["DOCA_reco"])
            _to_derive["rel"] = [x for x in _to_derive["rel"] if "DOCA_reco" not in x]


    if _to_derive['part'] or _to_derive['rel']:
        print("Missing particle feature definition:", _to_derive['part'])
        print("Missing relation feature definition:", _to_derive['rel'])

    return sum(part_base_quant, []), sum(rel_base_quant, [])


def derive_quantities(stable_part, rel_data, to_derive):
    print("=" * 15)
    print("Deriving additional quantities")
    if to_derive['part']:
        if any("real_prob" in p for p in to_derive["part"]):
            print("Deriving realistic probability through sampling")
            p_reco = np.sqrt(
                stable_part["px_reco"] ** 2 + stable_part["py_reco"] ** 2 + stable_part["pz_reco"] ** 2).values
            real_prob_feature = [p.split("_")[-1] for p in to_derive["part"] if "real_prob" in p]
            real_prob_list = [p for p in to_derive["part"] if "real_prob" in p]
            pid_feature_list = ["Prob_" + p for p in real_prob_feature]
            pid_features = stable_part[pid_feature_list]

            pid = np.argmax(pid_features.values, axis=1)
            real_pid_df = pd.DataFrame({"p": p_reco, "pid": pid, **{k: 0. for k in pid_feature_list}, })
            corr_data, inter_data = get_pid_data(real_prob_feature)
            for i, prob in enumerate(real_prob_feature):
                selbool = real_pid_df["pid"] == i
                if np.sum(selbool) == 0:
                    continue
                new_prob = get_prob_at_p(real_pid_df.loc[selbool, "p"].values, pid_feature_list, inter_data[prob],
                                         corr_data[prob])
                real_pid_df.loc[selbool, ["p"] + pid_feature_list] = new_prob.values
            real_pid_df = real_pid_df.rename(columns=dict(zip(pid_feature_list, real_prob_list)))
            stable_part = pd.concat([stable_part, real_pid_df[real_prob_list]], axis=1)
        if "ghost_prob" in to_derive['part']:
            stable_part["ghost_prob"] = stable_part["Prob_ghost"]
            del stable_part["Prob_ghost"]
    if to_derive['rel']:
        if "log_DOCA_reco" in to_derive["rel"]:
            rel_data["log_DOCA_reco"] = np.log(rel_data["DOCA_reco"].values + 1e-5)
            del rel_data["DOCA_reco"]
    return stable_part, rel_data

    """this cane be done later on"""
    """# Obtain information of the additional feature

    print('Adding derived quantities to the particles ...')
    self.stable_part.loc[:, f'eta{self.reco_suffix}'] = calculate_eta(self.stable_part, self.reco_suffix)
    self.stable_part.loc[:, f'phi{self.reco_suffix}'] = calculate_phi(self.stable_part, self.reco_suffix)
    self.stable_part.loc[:, f'ZDist{self.reco_suffix}'] = calculate_zdist(self.stable_part, self.reco_suffix)
    self.stable_part.loc[:, f'RadialDist{self.reco_suffix}'] = calculate_radialdist(self.stable_part, self.reco_suffix)
    self.stable_part.loc[:, f'log_associated_ip{self.reco_suffix}'] = calculate_log_associated_ip(self.stable_part, self.reco_suffix)
    print('Done.')

    # Addition of derived quantities to the relations
    print('Adding derived quantities to the relations ...')
    self.rel_data['abs_delta_z0_reco'] = np.abs(data_relations['delta_z0_reco'])
    import pdb; pdb.set_trace()
    self.rel_data['NotFromSameHeavyHadron'] = (data_relations['FromSameHeavyHadron'] == 0)
    data_stable_particles.set_index(['ExpandedEventNumber', 'ExpandedParticleIndex@'], inplace=True)
    p1 = data_stable_particles.reindex(pd.MultiIndex.from_arrays(
        [data_relations['ExpandedEventNumber'].values, data_relations['FirstParticleIndex'].values]))[
        ['pt_reco', 'eta_reco', 'phi_reco', 'RadialDist_reco']]
    p2 = data_stable_particles.reindex(pd.MultiIndex.from_arrays(
        [data_relations['ExpandedEventNumber'].values, data_relations['SecondParticleIndex'].values]))[
        ['pt_reco', 'eta_reco', 'phi_reco', 'RadialDist_reco']]
    data_relations.loc[:, 'min_pt_reco'] = np.minimum(p1['pt_reco'].to_numpy(np.float32),
                                                      p2['pt_reco'].to_numpy(np.float32))
    data_relations.loc[:, 'delta_eta_reco'] = np.abs(
        p1['eta_reco'].to_numpy(np.float32) - p2['eta_reco'].to_numpy(np.float32))
    data_relations.loc[:, 'delta_phi_reco'] = np.abs(
        p1['phi_reco'].to_numpy(np.float32) - p2['phi_reco'].to_numpy(np.float32))
    data_relations.loc[:, 'delta_RadialDist_reco'] = np.abs(
        p1['RadialDist_reco'].to_numpy(np.float32) - p2['RadialDist_reco'].to_numpy(np.float32))
    data_stable_particles.reset_index(inplace=True)
    del p1, p2
    print('Done.')

    # if self.current_event_number == -1:
    #    with open('../PYTHIA_gen_files/' + self.trigger_mapping_file_name) as file:
    #        self.trigger_mapping = yaml.load(file, Loader=yaml.CLoader)

    # Add information on the particle multiplicity
    print('Computing the particle multiplicity per event ...')
    if self.include_neutrals == 0:
        num_tracks_map = data_stable_particles['ExpandedEventNumber'].value_counts()
        data_stable_particles.loc[:, 'NumTracksInEvent'] = data_stable_particles['ExpandedEventNumber'].map(
            num_tracks_map).to_numpy(np.int32)
        data_relations.loc[:, 'NumTracksInEvent'] = data_relations['ExpandedEventNumber'].map(
            num_tracks_map).to_numpy(np.int32)
        del num_tracks_map
    print('Done.')"""


def calculate_eta(data, reco_suffix: str = ""):
    pt = data[f'pt{reco_suffix}'].to_numpy(np.float32)
    pz = data[f'pz{reco_suffix}'].to_numpy(np.float32)

    top = np.sqrt(pt ** 2 + pz ** 2) + pz
    bot = np.sqrt(pt ** 2 + pz ** 2) - pz

    eta = 0.5 * np.log(top / bot)
    return eta


def calculate_phi(data, reco_suffix: str = ""):
    px = data[f'px{reco_suffix}'].to_numpy(np.float32)
    py = data[f'py{reco_suffix}'].to_numpy(np.float32)

    phi = np.arctan2(px, py)
    return phi


def calculate_zdist(data, reco_suffix: str = ""):
    zProd = data[f'zProd{reco_suffix}'].to_numpy(np.float32)
    zPV = data[f'associated_zPV{reco_suffix}'].to_numpy(np.float32)
    zdist = zProd - zPV
    return zdist


def calculate_radialdist(data, reco_suffix: str = ""):
    xProd = data[f'xProd{reco_suffix}'].to_numpy(np.float32)
    yProd = data[f'yProd{reco_suffix}'].to_numpy(np.float32)
    xPV = data[f'associated_xPV{reco_suffix}'].to_numpy(np.float32)
    yPV = data[f'associated_yPV{reco_suffix}'].to_numpy(np.float32)

    radialdst = np.sqrt((xProd - xPV) ** 2 + (yProd - yPV) ** 2)
    return radialdst


def calculate_log_associated_ip(data, reco_suffix: str = ""):
    log_ip = np.log(data[f'associated_ip{reco_suffix}'].to_numpy(np.float32) + 1.e-6)
    return log_ip


def find_row_indices(t1, t2):

    t1_expanded = t1.unsqueeze(1)
    t2_expanded = t2.unsqueeze(0)


    matches = (t1_expanded == t2_expanded).all(dim=2)


    matches_int = matches.int()



    indices = torch.argmax(matches_int, dim=1)


    valid = matches.any(dim=1)
    indices[~valid] = -1


    num_classes = t2.shape[0]


    one_hot_encoded = torch.nn.functional.one_hot(indices, num_classes=num_classes)

    return indices, one_hot_encoded


def match_pvs_by_distance(df):
    pvs_true = np.unique(df[['xPV', 'yPV', 'zPV']], axis=0)
    pvs_true = pvs_true[~(pvs_true == 0).all(axis=1)]
    pvs_reco = np.unique(df[['xPV_reco', 'yPV_reco', 'zPV_reco']], axis=0)

    pvs_true_expanded = pvs_true[:, np.newaxis, :]
    pvs_reco_expanded = pvs_reco[np.newaxis, :, :]

    distances_matrix = np.sqrt(np.sum((pvs_true_expanded - pvs_reco_expanded) ** 2, axis=2))

    indices = np.argmin(distances_matrix, axis=1)
    distances = np.min(distances_matrix, axis=1)

    mapping = {}
    for i in range(len(pvs_true)):
        if distances[i] <= 2:
            mapping[tuple(pvs_true[i])] = indices[i]
        else:
            mapping[tuple(pvs_true[i])] = -1

    pv_values = df[['xPV', 'yPV', 'zPV']].values
    pv_indices = np.zeros(len(df), dtype=int) - 1

    for i in range(len(df)):
        pv = tuple(pv_values[i])
        if not (pv_values[i] == 0).all() and pv in mapping:
            pv_indices[i] = mapping[pv]
    return pv_indices


def calculate_ip(df, reco_suffix: str = "", log_ip: bool = True):

    pvs = torch.from_numpy(df[[f'xPV{reco_suffix}', f'yPV{reco_suffix}', f'zPV{reco_suffix}']].to_numpy())
    pvs = torch.unique(pvs, dim=0).to(torch.float32)


    tracks_prod = df[[f'xProd{reco_suffix}', f'yProd{reco_suffix}', f'zProd{reco_suffix}']]
    tracks_p = df[[f'px{reco_suffix}', f'py{reco_suffix}', f'pz{reco_suffix}']]


    tracks_prod = torch.from_numpy(tracks_prod.to_numpy()).unsqueeze(1).repeat(1, pvs.shape[0], 1)
    tracks_p = torch.from_numpy(tracks_p.to_numpy()).unsqueeze(1).repeat(1, pvs.shape[0], 1)

    r = tracks_prod - pvs
    ips = torch.sqrt(
        torch.sum(r ** 2, dim=-1) - torch.sum(tracks_p * r, dim=-1) ** 2 / torch.sum(tracks_p ** 2, dim=-1))
    ips[torch.isnan(ips)] = 0

    if log_ip:
        ips = torch.log(ips + 1.e-6)
    return ips
