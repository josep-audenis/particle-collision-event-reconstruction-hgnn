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


def plot_prob_corr(df, file=None):
    prob_vars = ['Prob_p', 'Prob_e', 'Prob_mu', 'Prob_k', 'Prob_pi']

    f, ax = plt.subplots(1, 2, figsize=(16, 6))
    corr_matrix = df[prob_vars].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=ax[0], cbar_kws={'label': 'Correlation'})
    cov_matrix = df[prob_vars].cov()
    sns.heatmap(cov_matrix, annot=True, fmt='.4f', cmap='viridis',
                square=True, ax=ax[1], cbar_kws={'label': 'Covariance'})
    plt.tight_layout()
    if file != None:
        plt.savefig(f"plots/{file}.pdf")
        np.savez(f"data/{file}.npz", cov_matrix)

    return corr_matrix, cov_matrix


def create_prob_interpolators(df, file=None):
    interpolators = {}
    prob_vars = ['Prob_p', 'Prob_e', 'Prob_mu', 'Prob_k', 'Prob_pi']

    for var in prob_vars:
        interpolators[var] = interp1d(
            df["p"],
            df[var],
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )

    if file != None:
        raw_inter_data = {key: {'x': obj.x, 'y': obj.y} for key, obj in interpolators.items()}
        np.savez(f'data/{file}.npz', **raw_inter_data)

    return interpolators


def get_prob_at_p(p_val, interpolators, cov_matrix, noise_std=0.05):
    prob_vars = ['Prob_p', 'Prob_e', 'Prob_mu', 'Prob_k', 'Prob_pi']
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
        noisy_values = np.clip(noisy_values, 0, 1)
    else:
        noisy_values = mean_values

    noisy_values /= noisy_values.sum(axis=1, keepdims=True)

    result_df = pd.DataFrame({
        'p': p_val,
        prob_vars[0]: noisy_values[:, 0],
        prob_vars[1]: noisy_values[:, 1],
        prob_vars[2]: noisy_values[:, 2],
        prob_vars[3]: noisy_values[:, 3],
        prob_vars[4]: noisy_values[:, 4],
    })

    return result_df


def plot_pid_sampling(old_df, new_df, file=None):
    prob_vars = ['Prob_p', 'Prob_e', 'Prob_mu', 'Prob_k', 'Prob_pi']

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
        plt.savefig(f"plots/{file}.pdf")
        plt.savefig(f"plots/{file}.png")

    plt.show()

def create_maps():
    file = "/eos/lhcb/user/s/sicapell/DFEI/publictemp/dfei_tuple_magup_b_onlyLong.root"
    vars = ["id", "Prob_e", "Prob_mu", "Prob_k", "Prob_p", "Prob_pi", "px_reco", "py_reco", "pz_reco"]

    # Load df
    with uproot.open(file) as f:
        particles = f["Particles"]
        df = particles.arrays(vars, library="pd")

    df["p"] = np.sqrt(df["px_reco"] ** 2 + df["py_reco"] ** 2 + df["pz_reco"] ** 2)
    df = df.drop(['px_reco', 'py_reco', 'pz_reco'], axis=1)
    df = df[df["Prob_e"] > 0]

    # Create different df based on true ids
    id = np.abs(df["id"])
    part_data = {"e": df[id == 11], "mu": df[id == 13], "pi": df[id == 211], "k": df[id == 321], "p": df[id == 2212]}
    for key in part_data.keys():
        print(f"{key}: {len(part_data[key])}")
    print("=" * 30)

    # Create correlation and cov matrix
    corr_matrix = {}
    for key in part_data.keys():
        corr_matrix[key] = plot_prob_corr(part_data[key], f"{key}_corr")

    # Create interpolation maps currently using lin
    inter_dict = {}
    for key in part_data.keys():
        inter_dict[key] = create_prob_interpolators(part_data[key], f"{key}_interpolators")


    for key in part_data.keys():
        new_df = get_prob_at_p(part_data[key]["p"].values,
                               inter_dict[key], corr_matrix[key][1])
        plot_pid_sampling(part_data[key], new_df, f"{key}_response_sim")


if __name__ == "__main__":
    create_maps()