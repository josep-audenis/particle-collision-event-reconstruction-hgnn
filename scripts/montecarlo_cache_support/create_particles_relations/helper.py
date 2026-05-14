from typing import Tuple
import re
import os

import uproot
import numpy as np
import pandas as pd


def get_signal(sig_file: str) -> Tuple[str, int, int]:
    match = re.search(r"GeneratedParticles_(.+?)_\d+", sig_file)
    sig_mode = match.group(1)
    numbers = re.findall(r'\d{8}', sig_file)

    return sig_mode, int(numbers[0]), int(numbers[1]) + 1


def get_rndm_seed(sig_file: str) -> int:
    if os.path.exists(sig_file):
        np_random_seed_dict = np.load("seed_dict.npy", allow_pickle=True).item()
        if sig_file not in np_random_seed_dict.keys():
            while True:
                rndm = np.random.randint(10000)
                if rndm not in np_random_seed_dict.values():
                    np_random_seed_dict[sig_file] = rndm
                    np.save("seed_dict.npy", np_random_seed_dict)
                    break
    else:
        rndm = np.random.randint(10000)
        np_random_seed_dict = {sig_file: rndm}
        np.save("seed_dict.npy", np_random_seed_dict)
    return np_random_seed_dict[sig_file]


def load_data(indir: str, file: str, mode: str, vars: list, sim: str = "pythia", pv_req: np.array = None,
              identifier=None) -> pd.DataFrame:
    if identifier is not None:
        root_file = f"{indir}/DFEI_{sim}_generator/{identifier}_{mode}/{file}"
    else:
        root_file = f"{indir}/DFEI_{sim}_generator/{mode}/{file}"
    with uproot.open(root_file) as file:
        particles = file["Particles"]

        df = particles.arrays(vars, library="pd")

        # In bkg sample not all pvs are used
        if pv_req is not None:
            df = df[df["EventNumber"].isin(set(pv_req))]

    return df
