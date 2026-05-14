import argparse
import copy
import re
import yaml
import torch
import os
from tqdm import tqdm

from GraphConstruction import GraphConstruction
from helper import *

if __name__ == "__main__":
    # python main.py --indir /eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_pythia_particles_relations/ --outdir /eos/user/y/yukaiz/DFEI_IFT/cached_data/ --file Event_data_particles_inclusive_nu7p6_00079000_to_00079999.root --ft --normed
    # python main.py --indir /eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_LHCb_particles_relations/ --outdir /eos/user/y/yukaiz/DFEI_IFT/cached_data/ --file Event_data_particles_inclusive_LHCbcollision_00000000_to_00000999.root  --normed --identifier 00258329
    parser = argparse.ArgumentParser(description='Process signal and simulation files')
    parser.add_argument('--indir', type=str, default=None,
                        help='in directory')
    parser.add_argument('--outdir', type=str, default=None,
                        help='out directory')
    parser.add_argument('--file', type=str, default=None,
                        help='signal mode file')
    parser.add_argument('--ft', action='store_true',
                        help='Saving information for FT inference')
    parser.add_argument('--edgeprune', action='store_true',
                        help='Apply early edge prune which requires theta < 0.26 rad or same PV from minIP')
    parser.add_argument('--normed', action='store_true',
                        help='Normalize graph')
    parser.add_argument('--config', type=str, default=None,
                        help='Alternative config file holding all information')
    parser.add_argument('--identifier', type=str, default=None,
                        help='LHCb sim sim type identifier')
    parser.add_argument('--mode', type=str, default='default', choices=['tst_only', 'trn_only', 'default'],
                        help='Caching mode: tst_only saves full particle info, trn_only saves minimal graph data')
    args = parser.parse_args()

    node_features = ["px_reco", "py_reco", "pz_reco",
                     "xProd_reco", "yProd_reco", "zProd_reco",
                     "charge", "ghost_prob"
                     ]
    pythia_pid_features = ['Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi'
                           # real prob kinda exist but not check
                           # 'real_prob_k', 'real_prob_e', 'real_prob_mu', 'real_prob_p', 'real_prob_pi'
                           ]
    # we change the definition of Prob_ghost to ghost_prob to be not selected to a pid container
    lhcb_pid_features = ['Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi']

    if args.identifier is None:
        node_features = node_features + pythia_pid_features
    else:
        node_features = node_features + lhcb_pid_features

    edge_features = ["FromSameAssociatedPV_reco", "theta_reco", "trdist_reco", "delta_z0_reco", "log_DOCA_reco"]

    topo_classes = ['TopoLCA_class0', 'TopoLCA_class1', 'TopoLCA_class2', 'TopoLCA_class3']

    print("=" * 30)
    print("Starting the cacheing script")
    env = GraphConstruction(dir=args.indir, file=args.file, node_features=node_features, edge_features=edge_features,
                            topo_classes=topo_classes, identifier=args.identifier,
                            use_ft=args.ft, edge_prune=args.edgeprune)

    # Creating outdir syntax
    pattern = rf"{re.escape(env.decay)}_([^_]+)"
    match = re.search(pattern, args.file)
    conditions = match.group(1)
    if args.normed:
        conditions += '_normed'
    if args.identifier is None:
        outdir = args.outdir + f"DFEI_{env.simulation_type}_{conditions}_pt_data/{env.decay}/"
    else:
        outdir = args.outdir + f"DFEI_{env.simulation_type}_{conditions}_pt_data/{args.identifier}_{env.decay}/"
    os.makedirs(outdir, exist_ok=True)

    # just copy the old one and apply on new caching one
    start, end = env.start, env.end + 1
    thr = 0.8
    mode = args.mode
    print("Using mode", mode)

    print("=" * 30)
    print("Starting cacheing graphs")
    complete_fail = 0
    trn_data, val_data, tst_data = [], [], []
    for i in tqdm(range(start, end), desc="Cacheing"):
        env.get_event(i)
        data = env.get_graph()
        # Check if the graph has sufficient amount of nodes
        if data == 0:
            complete_fail += 1
            print("not enough tracks", i)
            continue

        # apply normalization
        if args.normed:
            data = env.normalize_graph(data)

        if (mode == "default" and i < int(start + thr * (end - start))) or mode == "trn_only":
            trn_data.append(data)
        else:
            if mode != "tst_only":
                val_data.append(copy.deepcopy(data))
            particle_keys = {"final_keys": "ExpandedParticleIndex", "part_ids": "id"}
            data = env.add_particle_info(data, particle_keys)
            data = env.add_particle_info(data, {"asso_hh": "ExpandedPrimaryHeavyHadronIndex"}, node="tracks")
            if env.simulation_type == "pythia":  # currently only exists in pythia :c
                data = env.add_particle_info(data, {"nfinals": "nFinals"}, node="tracks")
            relation_keys = {"moth_ids": "TopoLCA_ids"}
            data = env.add_relation_info(data, relation_keys)

            if env.simulation_type == "LHCb":
                data = env.add_evt_run_number(data)

            # Only information from signal particles
            env.use_only_signal_particles = True
            particle_keys = {"truth_part_keys": "ExpandedParticleIndex", "truth_part_ids": "id"}
            data = env.add_particle_info(data, particle_keys)
            relation_keys = {"truth_moth_ids": "TopoLCA_ids", "lca_chain": "LCA"}
            data = env.add_relation_info(data, relation_keys)
            # get lca reco of signal particles only
            data = env.get_true_graph(data)
            tst_data.append(data)
            '''
            # Checking the pv
            true_final_keys = data["truth_part_keys"].to(torch.int)
            final_keys = data["final_keys"].to(torch.int)
            mask = torch.isin(final_keys, true_final_keys)
            print('=' * 15)
            print(data["truth_moth_ids"])
            print(data['truth_y'])
            index = torch.where(mask)[0]
            pv_edge_bool = torch.isin(data[("tracks", "pvs")].edge_index[0], index)
            y = data[("tracks", "pvs")].y[pv_edge_bool].squeeze() == 1
            print(data[("tracks", "pvs")].edge_index[:, pv_edge_bool][1][y])
            '''

    print("Completed cacheing graphs")
    print("Number of events without any tracks", complete_fail)
    print("=" * 30)
    print("Writing graphs to disc")
    # saving the files
    if mode == "default":
        trn_file = f"{outdir}/trn_data_{str(start).zfill(8)}_{str(int(start + thr * (end - start - 1))).zfill(8)}.pt"
        val_file = f"{outdir}/val_data_{str(int(start + thr * (end - start))).zfill(8)}_{str(end - 1).zfill(8)}.pt"
        tst_file = f"{outdir}/tst_data_{str(int(start + thr * (end - start))).zfill(8)}_{str(end - 1).zfill(8)}.pt"
        save_compressed(trn_data, trn_file, level=3)
        save_compressed(val_data, val_file, level=3)
        save_compressed(tst_data, tst_file, level=3)
    elif mode == "trn_only":
        trn_file = f"{outdir}/trn_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
        save_compressed(trn_data, trn_file, level=3)
    elif mode == "tst_only":
        tst_file = f"{outdir}/tst_data_{str(start).zfill(8)}_{str(end - 1).zfill(8)}.pt"
        save_compressed(tst_data, tst_file, level=3)

    with open(f"{outdir}/config.yaml", "w") as f:
        yaml.safe_dump(env.configs, f, sort_keys=False)
    print("Done, exiting")
    print("=" * 30)
