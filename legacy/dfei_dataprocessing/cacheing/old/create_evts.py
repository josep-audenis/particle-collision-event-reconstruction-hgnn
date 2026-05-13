import numpy as np
import os

import torch

from create_pid_response import create_maps
from pid_helper import load_pid_data
from cache_pt import hgnn_caching


def create_evts(env, start, threshold, outdir, norm_graph=False, ft_bool=False, real_pid=False, save=None):
    train_data = []
    val_data = []
    tst_data = []

    failed = 0
    trn_file = f"{outdir}/trn_data_{str(start).zfill(6)}_{str(start + threshold).zfill(6)}.pt"
    val_file = f"{outdir}/val_data_{str(start + threshold).zfill(6)}_{str(start + env.nevents).zfill(6)}.pt"
    tst_file = f"{outdir}/tst_data_{str(start + threshold).zfill(6)}_{str(start + env.nevents).zfill(6)}.pt"

    # check if maps exists if not create them
    pid_maps = None
    if real_pid:
        if not os.path.exists("data/e_corr.npz"):
            create_maps()
        pid_maps = load_pid_data()

    for i in range(env.nevents):
        env.reset()
        env.get_event(start + i)

        graph = env.get_graph_LCA()
        graph_target = env.get_truth_graph_LCA()  # get the LCA score


        # Get information
        if ft_bool:
            graph["isfromb"] = env.get_truth_isfromb()
            graph["fragmentation"] = env.get_fragmentation_new()

        graph['part_ids'] = np.array(env.get_truth_particle_ids())
        graph["primariy_idx"] = env.get_primariyhh_indx()
        graph["nfinals"] = env.get_nfinals()
        graph['keys'] = np.array(env.get_particle_keys())
        graph['ids'] = np.array(env.get_truth_TopoLCA_ids())

        # Getting the truth information
        # set_use_only_signal_particles enables only consideration to signal true final states
        env.set_use_only_signal_particles(1)
        graph['lca_chain'] = np.array(env.get_truth_LCA())
        truth_graph = env.get_truth_graph_LCA()
        graph['truth_senders'] = truth_graph['senders']
        graph['truth_receivers'] = truth_graph['receivers']
        graph['truth_y'] = truth_graph['edges']
        graph['truth_ids'] = np.array(env.get_truth_TopoLCA_ids())
        graph['truth_part_ids'] = np.array(env.get_truth_particle_ids())
        graph['truth_part_keys'] = np.array(env.get_particle_keys())
        env.set_use_only_signal_particles(0)

        graph_target["true_b_id"] = env.get_b_to_final()

        if len(graph["nodes"]) < 4:
            print("not enough tracks", start + i)
            failed += 1
        elif i < threshold:
            data = hgnn_caching(graph, graph_target, norm_graph=norm_graph, ft_bool=ft_bool, real_pid=real_pid,
                                pid_map=pid_maps, perf_mode=True)
            train_data.append(data)
        else:
            data = hgnn_caching(graph, graph_target, norm_graph=norm_graph, ft_bool=ft_bool, real_pid=real_pid,
                                pid_map=pid_maps)
            tst_pt = hgnn_caching(graph, graph_target, perf_mode=True, norm_graph=norm_graph, ft_bool=ft_bool,
                                  real_pid=real_pid, pid_map=pid_maps)
            val_data.append(data)
            tst_data.append(tst_pt)

    print("Number of failed events:", failed)
    if save is None:
        torch.save(train_data, trn_file)
        torch.save(val_data, val_file)
        torch.save(tst_data, tst_file)
    else:
        if "trn" in save:
            torch.save(train_data, trn_file)
        if "val" in save:
            torch.save(val_data, val_file)
        if "tst" in save:
            torch.save(tst_data, tst_file)

