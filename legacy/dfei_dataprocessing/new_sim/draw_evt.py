import pdb

import uproot
import awkward as ak
import numpy as np
import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx

from particle import PDGID, Particle


def safe_get_particle_name(pdg_id, withID=True):
    """Get the name of a particle from its PDG ID, or return 'noID' if the ID is not found."""

    try:
        if withID:
            return f"{Particle.from_pdgid(pdg_id).name} [{int(pdg_id)}]" if pdg_id != 0 else "noName"
        else:
            return f"{Particle.from_pdgid(pdg_id).name}" if pdg_id != 0 else "noName"
    except:
        return "noID"


def get_var(_path, _files, _var, _tree):
    """Getting the variables from the root files"""
    trees = [uproot.open(_path + "/" + _file)[_tree] for _file in _files]
    _data = [tree.arrays(_var) for tree in trees]
    params = ak.concatenate(_data, axis=0)
    res_dict = {}
    for var in _var:
        res_dict[var] = np.array(params[var])
    return res_dict


def add_edges(G, parent, child_dict):
    for child, grandchild in child_dict.items():
        G.add_edge(parent, child)
        add_edges(G, child, grandchild)


def dfs_dict(node):
    children = decay_graph.get(node, [])
    return {node: {child: dfs_dict(child)[child] for child in children}}


if __name__ == "__main__":
    path = "/afs/cern.ch/work/y/yukaiz/public/new_sim/"
    file = ["statur.root"]

    var = ["EventNumber", "ParticleIndex", "id",
           "MotherIndex_1", "MotherIndex_2", "MotherId_1", "MotherId_2", "isFrag", "inGeomAcc", "isFinal"]
    var = get_var(path, file, var, "Particles")

    full_df = pd.DataFrame(var)
    events = np.unique(full_df["EventNumber"])

    with PdfPages("graph.pdf") as pdf:
        for evt in events:
            df = full_df[full_df["EventNumber"] == evt]
            df = df[df["id"] != 21]

            particles = np.concatenate([df["ParticleIndex"], df["ParticleIndex"]])
            mothers = np.concatenate([df["MotherIndex_1"], df["MotherIndex_2"]])
            is_frag = np.concatenate([df["isFrag"], df["isFrag"]])
            in_acc = np.concatenate([df["inGeomAcc"], df["inGeomAcc"]])
            is_final = np.concatenate([df["isFinal"], df["isFinal"]])

            G = nx.DiGraph()
            res = []
            for m, p in zip(mothers, particles):
                if m == 0:
                    continue
                G.add_edge(m, p)

            id = []
            for part_id in df["id"]:
                id.append(safe_get_particle_name(part_id))
            lookup = dict(zip(df["ParticleIndex"], id))

            frag_lookup = dict(zip(particles, is_frag))

            acc_lookup = dict(zip(particles, in_acc))
            final_lookup = dict(zip(particles, is_final))

            labels = {}
            node_colors = []
            font_color = []
            for node in G.nodes:
                labels[node] = lookup.get(node, "-999")

                color = "lightblue"

                if  final_lookup.get(node, 0) == 0:
                    # intermediate are grey
                    color="grey"

                name = lookup.get(node, "")
                if name.startswith("b"):
                    color = "red"
                elif name.startswith("s"):
                    color = "green"

                if frag_lookup.get(node, 0) == 1:
                    # only new
                    color = "purple"
                elif frag_lookup.get(node, 0) == 2:
                    # only old
                    color = "yellow"
                elif frag_lookup.get(node, 0) == 3:
                    # only both // take both with high confidence
                    color = "blue"



                if acc_lookup.get(node, 0) != 1:
                    # out of acceptance
                    color= "white"



                node_colors.append(color)
            plt.figure(figsize=(100, 30))  # much bigger canvas
            pos = graphviz_layout(G, prog="dot")

            nx.draw(
                G, pos,
                labels=labels,
                arrows=True,
                node_size=300,
                node_color=node_colors,
                font_size=3,
                arrowsize=8,
                width=0.5
            )
            plt.title("Decay Graph (excluding mother=90)")
            pdf.savefig()
            plt.close()
