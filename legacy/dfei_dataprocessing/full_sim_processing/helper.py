import os
import re

import pandas as pd
import numpy as np


def get_file_information(file: str) -> tuple:
    base_name = os.path.basename(file).split(".")[0]
    identifier = base_name.split('_')[2]
    if base_name.split('_')[3] == 'inclusive' or base_name.split('_')[3] == 'test':
        sample = base_name.split('_')[3]
    else:
        sample = base_name.split('_')[3] + '_' + base_name.split('_')[4]

    return identifier, sample


def get_evt_df(arrays, track_types: list):
    dfs = []
    for ttype in track_types:
        fields = [f for f in arrays.fields if f.startswith(ttype)]
        event_data = {f: arrays[f] for f in fields}
        df = pd.DataFrame(event_data)
        df = df.rename(columns=dict(zip(df.keys(), [key.replace(f"{ttype}_", "") for key in df.keys()])))
        if ttype == 'long':
            df["TrackType"] = 3
        elif ttype == 'up':
            df["TrackType"] = 4
        elif ttype == "down":
            df["TrackType"] = 5
        else:
            raise NotImplementedError
        dfs.append(df)

    evt_df = pd.concat(dfs)
    # this can be empty due to having no tracks, weird case but events can have 0 long tracks
    # We just skip the event
    if len(evt_df) == 0:
        return False, False, False

    # ghost tracks dont have a mc particle and such have mc id of -1 this will be changed to a unique number
    ghost_selbool = evt_df["ParticleIndex"] == -1
    nghost = np.sum(ghost_selbool)
    max_indx = evt_df["ParticleIndex"].max()
    ghost_indx = np.arange(max_indx + 1, max_indx + 1 + nghost)
    evt_df.loc[ghost_selbool, "ParticleIndex"] = ghost_indx

    """
    # remove the ghost tracks
    ghost_selbool = evt_df['Prob_ghost'] < 0.6
    evt_df = evt_df[ghost_selbool].copy()
    evt_df["CounterParticleIndex"] = np.arange(len(evt_df))

    # Create a relation df
    idx = np.triu_indices(len(evt_df), k=1)
    relations_buffer = pd.DataFrame()
    relations_buffer['FirstParticleIndex'] = idx[0]
    relations_buffer['SecondParticleIndex'] = idx[1]
    data = {"p1": evt_df.iloc[idx[0]], "p2": evt_df.iloc[idx[1]]}
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
            to_false.append(idx[np.argmax(evt_df[evt_df["CounterParticleIndex"].isin(idx)]["Prob_ghost"].values)])
        evt_df = evt_df[~evt_df["CounterParticleIndex"].isin(to_false)]

    del evt_df["CounterParticleIndex"]"""

    # Split the DataFrame into a final particle and a mother DataFrame
    mother_keys = [s for s in evt_df.keys() if re.search(r'\d', s)]
    particle_keys = [s for s in evt_df.keys() if not re.search(r'\d', s)]
    particle_df = evt_df[particle_keys + ["1MotherIndex"]]
    particle_df = particle_df.rename(columns={"1MotherIndex": "MotherIndex", "Charge": "charge"})
    particle_df["isFinal"] = 1
    particle_df["inGeomAcc"] = 1
    merged_mother_df = evt_df[mother_keys]

    # We still have multiple generational mothers together. Each mother now get its own row :)
    tmp_df = []
    cols = merged_mother_df.columns
    generations = [int(re.match(r"(\d+)", col).group(1)) for col in cols if re.match(r"(\d+)", col)]
    for gen in np.unique(generations):
        keys = [key for key in merged_mother_df.keys() if str(gen) in key]
        split_df = merged_mother_df[keys]
        split_df = split_df.rename(columns=dict(zip(keys, [key.replace(str(gen), "") for key in keys])))
        tmp_df.append(split_df)
    mother_df = pd.concat(tmp_df)

    # Each row contain a MC truth mother of some real particles. Those are necessary to obtain the LCA chain
    mother_df = mother_df.drop_duplicates(subset='MotherIndex', keep='first')
    mother_df = mother_df[mother_df["MotherIndex"] != -1]  # motherindex -1 correspond to not existing particle
    mother_df["gammactau"] = np.linalg.norm(mother_df[["MotherFD_X", "MotherFD_Y", "MotherFD_Z"]], axis=1)

    # Creating placeholder for Primary Heavy Hadron index
    mother_df["PrimaryHeavyHadronIndex"] = -1
    particle_df["PrimaryHeavyHadronIndex"] = -1
    mother_df["NumVisDaugh"] = -1
    mother_df["isFinal"] = 0

    return particle_df, mother_df, True


def get_all_heads(mother_df):
    while True:
        # Recompute isHead
        mother_df["isHead"] = mother_df["MotherIndex"].isin(mother_df["MotherMotherIndex"]) & \
                              ~mother_df["MotherMotherIndex"].isin(mother_df["MotherIndex"])

        bad_heads = (mother_df["isHead"] == True) & (mother_df["MotherIshadron"] == 0)

        if not bad_heads.any():
            break

        mother_df = mother_df[~bad_heads].copy()

    all_heads = mother_df[mother_df["isHead"] == True]["MotherIndex"].to_numpy(int)
    return mother_df, all_heads


def get_heavy_heads(mother_df, hadrons="b"):
    # get the head of heavy hadron currently only bottom
    b_sel = mother_df["MotherIsbhadron"] == 1 if "b" in hadrons else False
    c_sel = mother_df["MotherIschadron"] == 1 if "c" in hadrons else False

    if hadrons == "b":
        selbool = b_sel
    elif hadrons == "c":
        selbool = c_sel
    elif hadrons == "bc":
        selbool = b_sel | c_sel
    else:
        raise NotImplementedError
    heavy_had_df = mother_df[selbool].copy()

    # B hadrons can decay from another B hadron (excited to no excited).
    heavy_had_df["isHead"] = heavy_had_df["MotherIndex"].isin(heavy_had_df["MotherMotherIndex"]) & ~heavy_had_df[
        "MotherMotherIndex"].isin(heavy_had_df["MotherIndex"])
    heavy_heads = heavy_had_df[heavy_had_df["isHead"] == True]["MotherIndex"].to_numpy(int)
    return heavy_heads


def get_graph_df(df1, df2):
    df1 = df1[["ParticleIndex", "MotherIndex"]]
    df2 = df2[["MotherIndex", "MotherMotherIndex"]].rename(
        columns=dict(zip(["MotherIndex", "MotherMotherIndex"], ["ParticleIndex", "MotherIndex"])))
    graph_df = pd.concat([df1, df2])
    return graph_df


def get_decay_graph(df, heads):
    decay_heads = heads
    filtered_df = df.copy()

    graph = {}

    def add_children_recursive(parent_idx):
        children = filtered_df[filtered_df['MotherIndex'] == parent_idx]

        if len(children) > 0:
            child_indices = children['ParticleIndex'].values
            graph[int(parent_idx)] = [int(i) for i in child_indices]

            # Recursively process each child
            for child_idx in child_indices:
                add_children_recursive(child_idx)

    for start_idx in decay_heads:
        add_children_recursive(start_idx)

    return graph


def flatten_dict(d):
    seen = set()

    def visit(x):
        if x not in seen:
            seen.add(x)
            if x in d:
                for y in d[x]:
                    visit(y)

    for key in d:
        visit(key)
        for val in d[key]:
            visit(val)
    return list(seen)


from collections import deque, defaultdict


def layer_transitions(graph, root):
    # BFS to get layer of each node
    layer = {}
    queue = deque([(root, 0)])
    layer[root] = 0

    while queue:
        node, depth = queue.popleft()
        for child in graph.get(node, []):
            if child not in layer:
                layer[child] = depth + 1
                queue.append((child, depth + 1))

    # group edges by transition (layer → layer+1)
    transitions = defaultdict(dict)

    for parent, children in graph.items():
        l = layer[parent]
        transitions[l][parent] = children

    return transitions


def add_primary_hh_index(particle_df, mother_df, graph_df, heavy_hadron_heads):
    for head in heavy_hadron_heads:
        graph = get_decay_graph(graph_df, [head])
        if np.sum(particle_df["ParticleIndex"].isin(flatten_dict(graph)).values) < 2:
            # we do not mark B decays where there is only a single visible B as reconstructable
            continue
        mother_df.loc[mother_df["MotherIndex"].isin(flatten_dict(graph)), "PrimaryHeavyHadronIndex"] = head
        particle_df.loc[particle_df["ParticleIndex"].isin(flatten_dict(graph)), "PrimaryHeavyHadronIndex"] = head
    # Weird thing the head does not have itself as the PrimaryHeavyHadronIndex (needed for create particles + relations)
    mask = (mother_df["MotherIsbhadron"] == 1) & (mother_df["isHead"] == 1)
    mother_df.loc[mask, "PrimaryHeavyHadronIndex"] = -1
    return particle_df, mother_df


def add_num_vis_daugh(particle_df, mother_df, graph_df, heavy_hadron_heads):
    carry_visble = {}
    num_vis_daughters = {}
    for head in heavy_hadron_heads:
        graph = get_decay_graph(graph_df, [head])
        # Add the carried visibility of the stable particles
        for i in (particle_df[particle_df["ParticleIndex"].isin(flatten_dict(graph))]["ParticleIndex"]):
            carry_visble[int(i)] = 1
            num_vis_daughters[int(i)] = -1  # Reconstructed tracks carry visibility of 1

        # Create an inverse layer transition of the graph and adjust the visibility
        transitions = layer_transitions(graph, head)
        for i in sorted(transitions, reverse=True):
            for mother, daughters in transitions[i].items():
                num_vis = 0
                for leaf in daughters:
                    num_vis += carry_visble[leaf]
                num_vis_daughters[mother] = num_vis
                if num_vis > 0:
                    carry_visble[mother] = 1
                else:
                    carry_visble[mother] = 0
    mask = mother_df["MotherIndex"].isin(num_vis_daughters)
    mother_df.loc[mask, "NumVisDaugh"] = mother_df.loc[mask, "MotherIndex"].map(num_vis_daughters)
    return particle_df, mother_df


def adjust_df(particle_df, mother_df):
    # LHCb sim stores in MeV while Pythia in GeV. For later stability GeV is chosen
    for var in ["px", "py", "pz"]:
        particle_df[var] = particle_df[var] / 1000
        particle_df[f"{var}_reco"] = particle_df[f"{var}_reco"] / 1000

    # adjust some naming in the mother_df
    del mother_df["isHead"], mother_df["MotherFD_X"], mother_df["MotherFD_Y"], mother_df["MotherFD_Z"], mother_df[
        "MotherIshadron"]
    mother_nameing = {"MotherID": "id", "MotherIndex": "ParticleIndex", "MotherMotherIndex": "MotherIndex",
                      "MotherIsbhadron": "isbhadron", "MotherIschadron": "ischadron"}
    mother_df.rename(columns=mother_nameing, inplace=True)

    return pd.concat([particle_df, mother_df]).fillna(-1)


def save_root_helper(result):
    output_data_particles = pd.concat(result)

    new_branches_particles = list(output_data_particles.columns)
    tree_data = {
        name: output_data_particles[name].values
        for name in new_branches_particles
    }
    branch_types = {
        name: output_data_particles[name].dtype
        for name in new_branches_particles
    }
    return tree_data, branch_types


def calculate_angle(relations_buffer, reco):
    if len(relations_buffer) == 0:
        return relations_buffer

    if reco:
        suffix = '_reco'
    else:
        suffix = ''

    relations_buffer['p1_norm'] = np.sqrt(
        relations_buffer['p1_px' + suffix] ** 2 + relations_buffer['p1_py' + suffix] ** 2 + relations_buffer[
            'p1_pz' + suffix] ** 2)
    relations_buffer['p2_norm'] = np.sqrt(
        relations_buffer['p2_px' + suffix] ** 2 + relations_buffer['p2_py' + suffix] ** 2 + relations_buffer[
            'p2_pz' + suffix] ** 2)

    relations_buffer['cos_theta'] = (relations_buffer['p1_px' + suffix] * relations_buffer['p2_px' + suffix] +
                                     relations_buffer['p1_py' + suffix] * relations_buffer['p2_py' + suffix] +
                                     relations_buffer['p1_pz' + suffix] * relations_buffer['p2_pz' + suffix]) / (
                                            relations_buffer['p1_norm'] * relations_buffer['p2_norm'])

    relations_buffer['theta' + suffix] = np.arccos(np.float32(relations_buffer['cos_theta']))

    relations_buffer.drop(
        columns=['p1_norm', 'p2_norm', 'cos_theta'], inplace=True)

    return relations_buffer
