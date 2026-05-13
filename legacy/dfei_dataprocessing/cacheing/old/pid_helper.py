import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import pandas as pd

def load_pid_data():
    keys = ["e", "mu", "pi", "k", "p"]

    corr_data = {}
    interpolator = {}
    for key in keys:
        corr_data[key] = np.load(f"data/{key}_corr.npz", allow_pickle=True)["arr_0"]

        inter_data = np.load(f"data/{key}_interpolators.npz", allow_pickle=True)
        interpolator[key] = {
            key: interp1d(inter_data[key].item()['x'], inter_data[key].item()['y'])
            for key in inter_data.files
        }

    return corr_data, interpolator


def get_prob_at_p(p_val, interpolators, cov_matrix, noise_std=0.05):
    prob_vars = ['Prob_k', 'Prob_e', 'Prob_mu', 'Prob_p', 'Prob_pi']
    p_val = np.asarray(p_val)

    # Interpolate mean values at all p values at once
    mean_values = np.column_stack([
        interpolators[var](
            np.clip(p_val, interpolators[var].x.min(), interpolators[var].x.max())
        )
        for var in prob_vars
    ]).astype(float)

    # Add multivariate Gaussian noise if needed
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

    # Normalize rows to sum to 1
    noisy_values /= noisy_values.sum(axis=1, keepdims=True)

    # Create DataFrame
    result_df = pd.DataFrame({
        'p': p_val,
        prob_vars[0]: noisy_values[:, 0],
        prob_vars[1]: noisy_values[:, 1],
        prob_vars[2]: noisy_values[:, 2],
        prob_vars[3]: noisy_values[:, 3],
        prob_vars[4]: noisy_values[:, 4],
    })

    return result_df
