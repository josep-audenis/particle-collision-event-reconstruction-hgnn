import numpy as np
import uproot
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.LHCb2)

from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from matplotlib.colors import LogNorm

import os
import glob


def plot_prob_corr(df, prob_vars, file=None):
    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    corr_matrix = df[prob_vars].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax[0], cbar_kws={'label': 'Correlation'})
    cov_matrix = df[prob_vars].cov()
    sns.heatmap(cov_matrix, annot=True, fmt='.4f', cmap='viridis',
                square=True, ax=ax[1], cbar_kws={'label': 'Covariance'})
    plt.tight_layout()
    if file is not None:
        plt.savefig(f"data/plots/{file}.pdf")
        np.savez(f"data/{file}.npz", cov_matrix)

    return corr_matrix, cov_matrix


def create_prob_interpolators(df, prob_vars, file=None):
    interpolators = {}

    for var in prob_vars:
        interpolators[var] = interp1d(
            df["p"],
            df[var],
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )

    if file is not None:
        raw_inter_data = {key: {'x': obj.x, 'y': obj.y} for key, obj in interpolators.items()}
        np.savez(f'data/{file}.npz', **raw_inter_data)

    return interpolators


def get_prob_at_p(p_val, prob_vars, interpolators, cov_matrix, noise_std=0.05):
    p_val = np.asarray(p_val)

    mean_values = np.column_stack([interpolators[var](p_val) for var in prob_vars]).astype(float)

    # Adding noise
    if noise_std > 0:
        scaled_cov = (noise_std ** 2) * cov_matrix
        try:
            noisy_values = multivariate_normal.rvs(mean=np.zeros(5), cov=scaled_cov, size=len(p_val))
            noisy_values = mean_values + noisy_values
        except np.linalg.LinAlgError:
            noisy_values = mean_values + np.random.normal(0, noise_std * mean_values, size=mean_values.shape)
        noisy_values = np.clip(noisy_values, 1e-5, 1)
    else:
        noisy_values = mean_values

    noisy_values /= noisy_values.sum(axis=1, keepdims=True)


    result_df = pd.DataFrame({
        "p": p_val,
        **{var: noisy_values[:, i] for i, var in enumerate(prob_vars)}
    })

    return result_df


def plot_pid_sampling(old_df, new_df, prob_vars, file=None):
    fig, axs = plt.subplots(2, 5, figsize=(24, 8))

    p_bins = 50
    prob_bins = 50

    for idx, var in enumerate(prob_vars):
        # Top row - old data
        h1 = axs[0][idx].hist2d(
            old_df["p"], old_df[var],
            bins=[p_bins, prob_bins],
            norm=LogNorm(vmin=1, vmax=1e4),
            cmap='viridis'
        )
        axs[0][idx].set_xlabel(r'$p$ [MeV/$c$]', fontsize=14)
        axs[0][idx].set_ylabel(var, fontsize=14)

        cbar1 = fig.colorbar(h1[3], ax=axs[0][idx], pad=0.02)
        cbar1.ax.tick_params(labelsize=11)

        h2 = axs[1][idx].hist2d(
            new_df["p"], new_df[var],
            bins=[p_bins, prob_bins],
            norm=LogNorm(vmin=1, vmax=1e4),
            cmap='viridis'
        )
        axs[1][idx].set_xlabel(r'$p$ [MeV/$c$]', fontsize=14)
        axs[1][idx].set_ylabel(var, fontsize=14)

        cbar2 = fig.colorbar(h2[3], ax=axs[1][idx], pad=0.02)
        cbar2.ax.tick_params(labelsize=11)

        axs[0][idx].tick_params(labelsize=11)
        axs[1][idx].tick_params(labelsize=11)

    plt.tight_layout()

    if file is not None:
        plt.savefig(f"data/plots/{file}.pdf")
        plt.savefig(f"data/plots/{file}.png")

    plt.show()


def get_pid_data(keys, recalculate_maps=False):
    corr_data = {}
    interpolator = {}

    calculate_maps = []

    for key in keys:
        corr_file = f"data/{key}_corr.npz"
        inter_file = f"data/{key}_interpolators.npz"
        if not os.path.exists(corr_file) and not os.path.exists(inter_file):
            calculate_maps.append(f"Prob_{key}")
        if recalculate_maps:
            calculate_maps.append(f"Prob_{key}")

    if len(calculate_maps) > 0:
        # need to calculate the maps again
        create_maps(calculate_maps)

    for key in keys:
        corr_data[key] = np.load(f"data/{key}_corr.npz", allow_pickle=True)["arr_0"]

        inter_data = np.load(f"data/{key}_interpolators.npz", allow_pickle=True)
        interpolator[key] = {
            k: interp1d(
                inter_data[k].item()['x'],
                inter_data[k].item()['y'],
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            for k in inter_data.files
        }

    return corr_data, interpolator


def create_maps(calculate_maps):
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)

    files = glob.glob(
        "/eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_LHCb_particles_relations/00258329_inclusive/Event_data_particles*")
    vars = ["id", "px_reco", "py_reco", "pz_reco", "isFinal"] + calculate_maps
    dfs = []
    for file in files:
        with uproot.open(file) as f:
            particles = f["Particles"]
            df = particles.arrays(vars, library="pd")
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df["p"] = np.sqrt(df["px_reco"] ** 2 + df["py_reco"] ** 2 + df["pz_reco"] ** 2)
    df = df.drop(['px_reco', 'py_reco', 'pz_reco'], axis=1)
    df = df[df["isFinal"] == 1]

    # Create different df based on true ids
    abs_id = np.abs(df["id"])
    id_maps = {"e": 11, "mu": 13, "pi": 211, "k": 321, "p": 2212}
    part_data = {}
    for key, id in id_maps.items():
        part_data[key] = df[abs_id == id]

    for key in part_data.keys():
        print(f"{key}: {len(part_data[key])}")
    print("=" * 30)

    # Create correlation and cov matrix
    corr_matrix = {}
    for key in part_data.keys():
        corr_matrix[key] = plot_prob_corr(part_data[key], calculate_maps, file=f"{key}_corr")

    # Create interpolation maps currently using lin
    inter_dict = {}
    for key in part_data.keys():
        inter_dict[key] = create_prob_interpolators(part_data[key], calculate_maps, file=f"{key}_interpolators")

    for key in part_data.keys():
        new_df = get_prob_at_p(part_data[key]["p"].values, calculate_maps,
                               inter_dict[key], corr_matrix[key][1])
        plot_pid_sampling(part_data[key], new_df, calculate_maps, file=f"{key}_response_sim")


if __name__ == "__main__":
    create_maps(["Prob_e", "Prob_mu", "Prob_pi", "Prob_k", "Prob_p"])
