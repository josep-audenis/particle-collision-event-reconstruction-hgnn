import uproot
import awkward as ak
import pandas as pd
from itertools import chain

from calculation_helper import *

def read_file(file, branch, features, cut=None):
    with uproot.open(file) as f:
        tree = f[branch]
        arrays = tree.arrays(features, library="ak", cut=cut)
    df = ak.to_dataframe(arrays)
    return df

def get_part_df(file, branch, features, use_long_only):
    print("="*15)
    print("Reading in the data of", branch.split("/")[0])
    # track and edge features translation
    all_features = list(chain.from_iterable(features.values()))
    feature_vars = get_base_quant(all_features)

    # PV vars
    pv_vars = ["xPV_reco", "yPV_reco", "zPV_reco"]
    pv_vars = [feature_translation[x] for x in pv_vars]

    # identifier vars
    identifier_vars = ["EVENTNUMBER", "RUNNUMBER"]
    identifier_vars =  [feature_translation[x] for x in identifier_vars]

    cut= None
    if use_long_only:
        cut = "B_Tr_T_TRACKISLONG == 1"
    feature_df = read_file(file, branch, feature_vars, cut=cut)

    pv_df = read_file(file, branch, pv_vars)

    identifier_df = read_file(file, branch, identifier_vars)

    inverse_mapping = {v: k for k, v in feature_translation.items()}
    feature_df = feature_df.rename(columns=inverse_mapping)
    # convert momenta to GeV
    feature_df[["px_reco", "py_reco", "pz_reco"]] = feature_df[["px_reco", "py_reco", "pz_reco"]] / 1000
    pv_df = pv_df.rename(columns=inverse_mapping)
    identifier_df = identifier_df.rename(columns=inverse_mapping)

    return feature_df, pv_df, identifier_df


def apply_cuts(stable_df, pv_df):
    return stable_df, pv_df
    ghost_selbool = stable_df['ghost_prob'] < 0.6
    stable_df = stable_df[ghost_selbool].copy()
    stable_df["ParticleIndex"] = np.arange(len(stable_df))

    # Create a relation df
    idx = np.triu_indices(len(stable_df), k=1)
    relations_buffer = pd.DataFrame()
    relations_buffer['FirstParticleIndex'] = idx[0]
    relations_buffer['SecondParticleIndex'] = idx[1]
    data = {"p1": stable_df.iloc[idx[0]], "p2": stable_df.iloc[idx[1]]}
    features = ["px", "py", "pz"]
    for particle in data.keys():
        for feature in features:
            relations_buffer[f"{particle}_{feature}_reco"] = data[particle][f"{feature}_reco"].to_numpy(
                dtype=np.float64)

    # Calculate the opening angle between momenta.
    # For track paris with small opening angle remove the one with the largest ghost prob
    relations_buffer = calculate_angle(relations_buffer, 1)
    angle_selbool = relations_buffer["theta_reco"] < 0.0005
    if np.sum(angle_selbool) != 0:
        indx = relations_buffer[angle_selbool][["FirstParticleIndex", "SecondParticleIndex"]].values
        to_false = []
        for idx in indx:
            to_false.append(idx[np.argmax(stable_df[stable_df["ParticleIndex"].isin(idx)]["ghost_prob"].values)])
        stable_df = stable_df[~stable_df["ParticleIndex"].isin(to_false)]

    return stable_df, pv_df

def create_relation(stable_df):
    idx = np.triu_indices(len(stable_df), k=1)
    relations_buffer = pd.DataFrame()
    relations_buffer['FirstParticleIndex'] = idx[0]
    relations_buffer['SecondParticleIndex'] = idx[1]
    data = {"p1": stable_df.iloc[idx[0]], "p2": stable_df.iloc[idx[1]]}

    features = ["xProd", "yProd", "zProd", "px", "py", "pz"]
    for particle in data.keys():
        for feature in features:
            relations_buffer[f"{particle}_{feature}_reco"] = data[particle][f"{feature}_reco"].to_numpy(
                dtype=np.float64)

    relations_buffer['FromSameAssociatedPV_reco'] = np.array(
        (data["p1"]['associated_zPV_reco'].to_numpy() == data["p2"]['associated_zPV_reco'].to_numpy()),
        dtype=np.int32)

    relations_buffer = calculate_doca(relations_buffer, 1)
    relations_buffer = calculate_transversedist(relations_buffer, 1)
    relations_buffer = calculate_angle(relations_buffer, 1)
    return relations_buffer


def adjust_normalization(normalization_dict, features, mode: str = "auto"):
    normalization = pd.DataFrame(normalization_dict["LHCbcollision"]).T.loc[features["node_features"] + features["pid_features"] + features["edge_features"]]

    if mode == "auto":
        selected = pd.DataFrame(index=normalization.index, columns=['value'])
        skew_thresh = 1.0
        for feature in normalization.index:
            if abs(normalization.loc[feature, 'skew']) > skew_thresh:
                selected.loc[feature, 'value'] = (normalization.loc[feature, 'median'],
                                                  normalization.loc[feature, 'iqr'])
            else:
                selected.loc[feature, 'value'] = (normalization.loc[feature, 'mean'],
                                                  normalization.loc[feature, 'std'])

        selected[['val1', 'val2']] = pd.DataFrame(selected['value'].tolist(), index=selected.index)
        selected = selected.drop(columns='value')
    else:
        raise NotImplementedError("Implement only selecting mean or median")
    # division by zero save guard
    selected[selected["val2"] == 0] = 1
    return selected