import numpy as np

from helper import *
from create_events_helper import *
from tqdm import tqdm

from typing import List


def create_event(i: int,
                 stable_charged_df: pd.DataFrame, stable_photon_df, stable_pion_df,
                 heavy_hadron_df, unstable_from_hh_df,
                 evt_dict: dict = None):

    decay_dict = {"ancestors_per_particle": {}, "topo_ancestors_per_particle": {},
                  "ancestors_ids_per_particle": {}, "topo_ancestors_ids_per_particle": {},
                  "max_decay_chain_lengths": {-1: 0}, "topo_max_decay_chain_lengths": {-1: 0}}
    output_data_particles_list = []



    if evt_dict is not None:
        col_idx_list = evt_dict["CollisionIndexList"]
        idx_offset = evt_dict["IndexOffsetList"]
        col_offset_map = dict(zip(col_idx_list, idx_offset))


        true_pvs = np.array(evt_dict["TruePVCoords"])
        reco_pvs = np.array(evt_dict["RecoPVCoords"])
        inf_mask = np.isinf(reco_pvs)
        reco_pvs[inf_mask] = true_pvs[inf_mask]

        temp = get_evt_df(col_idx_list, col_offset_map, i,
                          stable_charged_df, stable_photon_df, stable_pion_df,
                          heavy_hadron_df, unstable_from_hh_df)
        stable_charged_df, stable_photon_df, stable_pion_df, heavy_hadron_df, unstable_from_hh_df = temp

    else:
        true_pvs, reco_pvs, temp = get_evt_df_fullsim(i ,
                                                      stable_charged_df, stable_photon_df, stable_pion_df,
                                                      heavy_hadron_df, unstable_from_hh_df)
        stable_charged_df, stable_photon_df, stable_pion_df, heavy_hadron_df, unstable_from_hh_df = temp


    list_of_df = [stable_charged_df, stable_photon_df, stable_pion_df, heavy_hadron_df, unstable_from_hh_df]
    zero_remap = np.max(pd.concat(list_of_df)["ExpandedParticleIndex"]) + 1
    for df in [stable_charged_df, stable_photon_df, stable_pion_df, heavy_hadron_df, unstable_from_hh_df]:
        df['ExpandedParticleIndex'] = df['ExpandedParticleIndex'].replace(0, zero_remap)
        df['ExpandedMotherIndex'] = df['ExpandedMotherIndex'].replace(0, zero_remap)
        df['ExpandedPrimaryHeavyHadronIndex'] = df['ExpandedPrimaryHeavyHadronIndex'].replace(0, zero_remap)
        
    """Process heavy hadrons"""
    if heavy_hadron_df.shape[0] > 0:
        particle_df = process_heavy_hadrons(heavy_hadron_df)
        output_data_particles_list.append(particle_df)
        epi_values = particle_df[f'ExpandedParticleIndex'].values
        decay_dict["max_decay_chain_lengths"].update(dict.fromkeys(epi_values, 0))
        decay_dict["topo_max_decay_chain_lengths"].update(dict.fromkeys(epi_values, 0))

        """Process unstable particles"""
        heavy_hadron_indices = np.unique(particle_df[f"ExpandedParticleIndex"]).tolist()
        if evt_dict is not None:
            PV_offset = [col_offset_map[i] for i in unstable_from_hh_df["EventNumber"]]
            for key in ["ParticleIndex", "MotherIndex", "fragPartnerIndex", "PrimaryHeavyHadronIndex"]:
                unstable_from_hh_df[f"Expanded{key}"] = unstable_from_hh_df[key] + PV_offset
        hh_daughter_selbool = unstable_from_hh_df[f'ExpandedPrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)
        unstable_from_hh_df = unstable_from_hh_df[hh_daughter_selbool].copy()
        cols = [f'ExpandedParticleIndex', f'ExpandedMotherIndex',
                'ParticleIndex', 'id', 'gammactau', 'NumVisDaugh']
        unstable_from_hh_df = unstable_from_hh_df[cols].drop_duplicates()
    else:
        return 0

    """Process charged stable particles"""
    if stable_charged_df.shape[0] > 0:
        stable_charged_df = process_finals(stable_charged_df, unstable_from_hh_df, heavy_hadron_df,
                                           heavy_hadron_indices,
                                           true_pvs, reco_pvs, decay_dict)
        if isinstance(stable_charged_df, int):
            return 0
        output_data_particles_list.append(stable_charged_df)


    """Process photon particles"""
    if stable_photon_df.shape[0] > 0 and False:
        stable_photon_df = process_finals(stable_photon_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices,
                                          true_pvs, reco_pvs, decay_dict)
        if isinstance(stable_photon_df, int):
            return 0
        output_data_particles_list.append(stable_photon_df)
    """Process neutral pion particles"""
    if stable_pion_df.shape[0] > 0 and False:
        stable_pion_df = process_finals(stable_pion_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices,
                                        true_pvs, reco_pvs, decay_dict)
        if isinstance(stable_pion_df, int):
            return 0
        output_data_particles_list.append(stable_pion_df)

    output_data_particles = pd.concat(output_data_particles_list, ignore_index=True, sort=True)
    if evt_dict is not None:
        mode = "onehot"
    else:
        mode= "LHCb"
    output_data_particles = compute_particle_PID(output_data_particles, mode=mode)
    output_data_particles = output_data_particles.drop_duplicates(subset=f'ExpandedParticleIndex')



    """Computing the relations"""
    relations_buffer = pd.DataFrame()
    relations_buffer = prcoess_relations(i, relations_buffer, output_data_particles, decay_dict)


    relations_buffer.drop(
        columns=['p1_xProd', 'p1_yProd', 'p1_zProd', 'p1_px', 'p1_py', 'p1_pz', 'p2_xProd', 'p2_yProd', 'p2_zProd',
                 'p2_px', 'p2_py', 'p2_pz', 'p1_xProd_reco', 'p1_yProd_reco', 'p1_zProd_reco', 'p1_px_reco',
                 'p1_py_reco', 'p1_pz_reco', 'p2_xProd_reco', 'p2_yProd_reco', 'p2_zProd_reco', 'p2_px_reco',
                 'p2_py_reco', 'p2_pz_reco', 'p1_MotherList', 'p2_MotherList', 'p1_TopoMotherList',
                 'p2_TopoMotherList',
                 'p1_MotherList_ids', 'p2_MotherList_ids', 'p1_TopoMotherList_ids', 'p2_TopoMotherList_ids'],
        inplace=True)
    output_data_particles.drop(columns=['MotherList', 'TopoMotherList', 'MotherList_ids', 'TopoMotherList_ids'],
                               inplace=True)
    return output_data_particles, relations_buffer


def create_multiple_events(infile: str, vars: List[str], bunch, outfile_suffix: str, dir: str, sim: str = None,
                           mapping=None, bkg_file: str = None, identifier=None):
    if sim == "pythia":
        col_in_bunch = np.array(list(set(
            col for i in bunch
            for col in mapping[i]['CollisionIndexList']
        )))

        sig_evt_mask = col_in_bunch < 0
        sig_col_in_bunch = col_in_bunch[sig_evt_mask] * -1 - 1
        bkg_col_in_bunch = col_in_bunch[~sig_evt_mask]

        print(f"Reading pythia events from {bunch[0]} to {bunch[-1]}")
        sig_mode, _, _ = get_signal(infile)
        sig_df = load_data(dir, infile, sig_mode, vars, pv_req=sig_col_in_bunch, sim=sim)
        sig_df["EventNumber"] = sig_df["EventNumber"] * -1 - 1
        bkg_df = load_data(dir, bkg_file, "background", vars, pv_req=bkg_col_in_bunch, sim=sim)
        df = pd.concat([sig_df, bkg_df], ignore_index=True, sort=True)
    elif sim == "LHCb":
        print(f"Reading LHCb events from {bunch[0]} to {bunch[-1]}")
        sig_mode, _, _ = get_signal(infile)
        df = load_data(dir, infile, sig_mode, vars, sim=sim, identifier=identifier)
    else:
        raise ValueError("Undefined simulation type")

    stable_charged_df = df[
        (df['isFinal'] == 1) & (df['isCharged'] == 1) & (df['zProd_reco'] > -500) & (df['zProd_reco'] < 500)]
    stable_photon_df = df[(df['isFinal'] == 1) & (df['id'] == 22) & (df['pz_reco'] > 0)]
    stable_pion_df = df[(df['isFinal'] == 1) & (df['id'] == 111) & (df['pz_reco'] > 0)]






    heavy_hadron_df = df[(df['PrimaryHeavyHadronIndex'] == -1) & (df['isbhadron'] == 1) &
                         (df['isFinal'] == 0)]

    unstable_from_hh_df = df[(df['isFinal'] == 0) & (df['PrimaryHeavyHadronIndex'] != -1)]

    results = []

    for i in tqdm(bunch, desc="Processing events"):
        if sim == "pythia":
            res = create_event(i,
                               stable_charged_df, stable_photon_df, stable_pion_df,
                               heavy_hadron_df, unstable_from_hh_df,
                               evt_dict=mapping[i])
        elif sim == "LHCb":
            res = create_event(i,
                               stable_charged_df, stable_photon_df, stable_pion_df,
                               heavy_hadron_df, unstable_from_hh_df)
        else:
            raise ValueError("Undefined simulation type")
        if res != 0:
            results.append(res)
    res0 = [result[0] for result in results]
    res1 = [result[1] for result in results]

    output_data_particles = pd.concat(res0, ignore_index=True, sort=True)
    output_data_relations = pd.concat(res1, ignore_index=True, sort=True)
    output_data_relations = format_relations(output_data_relations)
    if identifier is not None:
        outdir = f"{dir}/DFEI_{sim}_particles_relations/{identifier}_{sig_mode}"
    else:
        outdir = f"{dir}/DFEI_{sim}_particles_relations/{sig_mode}"
    os.makedirs(outdir, exist_ok=True)
    outfile = f"Event_data_particles_{outfile_suffix}_{str(bunch[0]).zfill(8)}_to_{str(bunch[-1]).zfill(8)}.root"

    """Saving particles file"""
    new_branches_particles = list(output_data_particles.columns)
    tree_data = {
        name: output_data_particles[name].values
        for name in new_branches_particles
    }
    branch_types = {
        name: output_data_particles[name].dtype
        for name in new_branches_particles
    }
    with uproot.recreate(f"{outdir}/{outfile}") as file_particles:
        file_particles.mktree("Particles", branch_types, title="Particles Tree")
        file_particles["Particles"].extend(tree_data)

    "Saving relations file"
    new_branches_relations = list(output_data_relations.columns)
    tree_data = {
        name: output_data_relations[name].values
        for name in new_branches_relations
    }
    branch_types = {
        name: output_data_relations[name].dtype
        for name in new_branches_relations
    }
    with uproot.recreate(f"{outdir}/{outfile.replace('particles', 'relations')}") as file_relations:
        file_relations.mktree("ParticleRelations", branch_types, title="Relations Tree")
        file_relations["ParticleRelations"].extend(tree_data)

    del results, res0, res1, output_data_particles, output_data_relations
