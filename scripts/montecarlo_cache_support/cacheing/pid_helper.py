import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import pandas as pd

def load_pid_data(keys):

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


def get_prob_at_p(p_val, pid_feature_list, interpolators, cov_matrix, noise_std=0.05):
    p_val = np.asarray(p_val)


    mean_values = np.column_stack([
        interpolators[var](
            np.clip(p_val, interpolators[var].x.min(), interpolators[var].x.max())
        )
        for var in pid_feature_list
    ]).astype(float)

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
        pid_feature_list[0]: noisy_values[:, 0],
        pid_feature_list[1]: noisy_values[:, 1],
        pid_feature_list[2]: noisy_values[:, 2],
        pid_feature_list[3]: noisy_values[:, 3],
        pid_feature_list[4]: noisy_values[:, 4],
    })

    return result_df

