import os
import time
from tqdm import tqdm

import uproot
import numpy as np
import yaml

from helper import *


def get_collision_map(map_file: str, sig_file: str, bkg_file: str, indir: str, nu: int,
                      recreate_mapping: bool = False) -> dict:
    print("=" * 30)
    if recreate_mapping or not os.path.exists(map_file):
        print(f"Creating collision mapping")
        print(map_file)
        start = time.time()
        mapping = create_collision_map(sig_file, bkg_file, indir, nu)
        print(f"Time needed to create collision map: {(time.time() - start):.2f} s")
        with open(map_file, 'w') as f:
            yaml.dump(mapping, f)
    else:
        print("Loading collision map")
        print(map_file)
        with open(map_file) as f:
            mapping = yaml.load(f, Loader=yaml.FullLoader)
    return mapping


def create_collision_map(sig_file: str, bkg_file: str, indir: str, nu: int) -> dict:
    sig_mode, _, _ = get_signal(sig_file)
    with uproot.open(f"{indir}/DFEI_pythia_generator/{sig_mode}/{sig_file}") as file:
        sig_evts = np.unique(file["Particles"]["EventNumber"].array(library="np"))

    with uproot.open(f"{indir}/DFEI_pythia_generator/background/{bkg_file}") as file:
        bkg_evts = np.unique(file["Particles"]["EventNumber"].array(library="np").astype(np.uint32))


    col_dict = {}
    bkg_pvs_list = []
    num_pvs = np.maximum(1, np.random.poisson(nu, size=len(sig_evts)))
    mask_multi = num_pvs > 1
    sig_evts_multi = sig_evts[mask_multi]
    num_pvs_multi = num_pvs[mask_multi]
    for sig_collision in sig_evts[~mask_multi]:
        col_dict[int(sig_collision)] = np.array([sig_collision * -1 - 1])
    for sig_collision, num_pv in zip(sig_evts_multi, num_pvs_multi):
        sel_bkg_pv = np.random.randint(0, len(bkg_evts), size=num_pv - 1)
        selected = bkg_evts[sel_bkg_pv]
        col_dict[int(sig_collision)] = np.sort(np.append(selected, sig_collision * -1 - 1))
        bkg_pvs_list.append(selected)
    bkg_pvs = np.unique(np.concatenate(bkg_pvs_list) if bkg_pvs_list else np.array([]))


    vars  = ["EventNumber", "ParticleIndex",
             "xPV", "yPV", "zPV", "xPV_reco", "yPV_reco", "zPV_reco"]
    sig_df = load_data(indir, sig_file, sig_mode, vars)
    sig_grouped = sig_df[sig_df["EventNumber"].shift(-1) != sig_df["EventNumber"]].groupby("EventNumber")
    bkg_df = load_data(indir, bkg_file, "background", vars, pv_req=bkg_pvs)
    bkg_grouped = bkg_df[bkg_df["EventNumber"].shift(-1) != bkg_df["EventNumber"]].groupby("EventNumber")

    sorted_keys = np.sort(np.array(list(col_dict.keys())))
    for key in sorted_keys:
        sig_evt_df = sig_grouped.get_group(key)
        bkg_evt_dfs =[bkg_grouped.get_group(evt) for evt in col_dict[key] if evt in bkg_grouped.groups]

        if bkg_evt_dfs:
            bkg_evt_df = pd.concat(bkg_evt_dfs, ignore_index=True)
            df = pd.concat([sig_evt_df, bkg_evt_df], ignore_index=True)
        else:
            df = sig_evt_df.copy()

        idx_offset = np.concatenate([[0], np.cumsum(df["ParticleIndex"].values)[:-1]])
        pv_coords = df[["xPV", "yPV", "zPV"]].values
        reco_pv_coords = df[["xPV_reco", "yPV_reco", "zPV_reco"]].values

        col_dict[key] = {"CollisionIndexList":  col_dict[key].tolist(),
                         "IndexOffsetList": idx_offset.tolist(),
                         "TruePVCoords": pv_coords.tolist(),
                         "RecoPVCoords": reco_pv_coords.tolist()}

    return col_dict