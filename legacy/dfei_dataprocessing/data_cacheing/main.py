import argparse
from tqdm import tqdm
import os

from class_holder import *

import zstandard as zstd
import torch

def save_compressed(data, filepath, level=3):
    cctx = zstd.ZstdCompressor(level=level)
    with open(filepath + '.zst', 'wb') as f:
        with cctx.stream_writer(f) as compressor:
            torch.save(data, compressor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LHCb data to graphs')
    parser.add_argument("--path", type=str, required=True,
                        dest="PATH", help="path to root file")
    parser.add_argument("--branch", type=str, required=True,
                        dest="BRANCH", help="branch within the root tree")
    parser.add_argument("--sample", type=str, required=True,
                        dest="SAMPLE", help="Name of the decay mode (B_Something)")
    parser.add_argument("--identifier", type=str, required=True,
                        dest="IDENTIFIER", help="identifier to separate same decay mode but different condition")
    parser.add_argument("--start", type=int, required=True,
                        dest="START", help="starting number")
    args = parser.parse_args()

    features = {
        "node_features": [
            "px_reco", "py_reco", "pz_reco",
            "xProd_reco", "yProd_reco", "zProd_reco",
            "charge", "ghost_prob"
        ],
        "pid_features": [
            'Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi'
        ],
        "edge_features": [
            "FromSameAssociatedPV_reco", "theta_reco", "trdist_reco", "delta_z0_reco", "log_DOCA_reco"
        ]
    }
    print("=" * 30)
    print("Starting the cacheing script")
    outdir = "/eos/user/y/yukaiz/DFEI_IFT/data/DFEI_LHCb_data_normed_pt_data"
    outdir = f"{outdir}/{args.IDENTIFIER}_{args.SAMPLE}"
    os.makedirs(outdir, exist_ok=True)
    start = args.START
    end = args.START + 1000

    # Flag to use only long track
    long_only = True
    env = GraphCache(file=args.PATH, branch=args.BRANCH,
                     features=features,
                     use_long_only=long_only)

    tst_data = []
    for i in tqdm(range(env.nevts), desc="Cacheing"):
        data = env.process_data(i)
        data = env.normalize_graph(data)
        tst_data.append(data)
        if len(tst_data) == 1000:
            tst_file = f"{outdir}/tst_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
            val_file = f"{outdir}/val_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
            save_compressed(tst_data, tst_file, level=3)
            save_compressed(tst_data, val_file, level=3)
            start += 1000
            end += 1000
            tst_data = []


    # DFEI_LHCb_data_normed_pt_data
    tst_file = f"{outdir}/tst_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
    val_file = f"{outdir}/val_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
    save_compressed(tst_data, tst_file, level=3)
    save_compressed(tst_data, val_file, level=3)


# python main.py --path /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/DATA25.ROOT/00372678/0000/00372678_00000355_1.data25.root --branch BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree --sample Bu_JpsiK --identifier 23459876 --start 0
# python main.py --path /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/DATA25.ROOT/00372678/0000/00372678_00000042_1.data25.root --branch BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree --sample Bu_JpsiK --identifier 23459876 --start 24000
# python main.py --path /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/DATA25.ROOT/00372678/0000/00372678_00000043_1.data25.root --branch BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree --sample Bu_JpsiK --identifier 23459876 --start 45000