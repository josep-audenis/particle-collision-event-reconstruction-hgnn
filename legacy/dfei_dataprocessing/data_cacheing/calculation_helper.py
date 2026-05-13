import numpy as np
import pandas as pd
import torch


feature_translation = {
    "px_reco": "B_Tr_T_PX", "py_reco": "B_Tr_T_PY", "pz_reco": "B_Tr_T_PZ",
    "xProd_reco": "B_Tr_T_ProdX", "yProd_reco": "B_Tr_T_ProdY", "zProd_reco": "B_Tr_T_ProdZ",
    "charge": "B_Tr_T_Charge",
    "Prob_p": "B_Tr_T_PROBNN_P", "Prob_pi": "B_Tr_T_PROBNN_PI", "Prob_k": "B_Tr_T_PROBNN_K",
    "Prob_e": "B_Tr_T_PROBNN_E","Prob_mu": "B_Tr_T_PROBNN_MU",
    "ghost_prob": "B_Tr_T_PROBNN_GHOST",

    "associated_zPV_reco":"B_Tr_T_OWNPV_Z",

    "xPV_reco": "ALLPV_X", "yPV_reco": "ALLPV_Y", "zPV_reco": "ALLPV_Z",

    "RUNNUMBER": "RUNNUMBER", "EVENTNUMBER": "EVENTNUMBER",
}




def get_base_quant(features):
    to_load_quant = []

    if "theta_reco" in features:
        req = ["px_reco", "py_reco" , "pz_reco"]
        to_load_quant.append(req)
        features = [x for x in features if "theta_reco" != x]
    if "trdist_reco" or "delta_z0_reco" or "log_DOCA_reco" in features:
        req = ["xProd_reco", "yProd_reco", "zProd_reco" ,"px_reco", "py_reco" , "pz_reco"]
        to_load_quant.append(req)
        features = [x for x in features if x not in ["trdist_reco","delta_z0_reco", "log_DOCA_reco"]]
    if "FromSameAssociatedPV_reco" in features:
        req = ["associated_zPV_reco"]
        to_load_quant.append(req)
        features = [x for x in features if "FromSameAssociatedPV_reco" != x]
    to_load_quant.append(features)
    return [feature_translation[x] for x in sum(to_load_quant, [])]


def calculate_doca(relations_buffer, reco):
    '''
    Calculate DOCA between 2 particles. Define particle's traj. of the form
    vec{r} = vec{r0} + vec{u} * t, from this DeltaR = deltaR0 + u1 * t1 - u2 * t2,
    and minimize for t1 and t2
    '''

    if len(relations_buffer) == 0:
        return relations_buffer

    if reco:
        suffix = '_reco'
    else:
        suffix = ''

    # vector distance at t1=0, t2=0
    relations_buffer['delta_x0' + suffix] = relations_buffer['p1_xProd' + suffix] - relations_buffer[
        'p2_xProd' + suffix]
    relations_buffer['delta_y0' + suffix] = relations_buffer['p1_yProd' + suffix] - relations_buffer[
        'p2_yProd' + suffix]
    relations_buffer['delta_z0' + suffix] = relations_buffer['p1_zProd' + suffix] - relations_buffer[
        'p2_zProd' + suffix]

    # Define particle direction as unit vector
    p1_p = np.sqrt(relations_buffer['p1_px' + suffix] ** 2 + relations_buffer['p1_py' + suffix] ** 2 + relations_buffer[
        'p1_pz' + suffix] ** 2)
    p2_p = np.sqrt(relations_buffer['p2_px' + suffix] ** 2 + relations_buffer['p2_py' + suffix] ** 2 + relations_buffer[
        'p2_pz' + suffix] ** 2)
    relations_buffer['u1_x' + suffix] = relations_buffer['p1_px' + suffix] / p1_p
    relations_buffer['u1_y' + suffix] = relations_buffer['p1_py' + suffix] / p1_p
    relations_buffer['u1_z' + suffix] = relations_buffer['p1_pz' + suffix] / p1_p
    relations_buffer['u2_x' + suffix] = relations_buffer['p2_px' + suffix] / p2_p
    relations_buffer['u2_y' + suffix] = relations_buffer['p2_py' + suffix] / p2_p
    relations_buffer['u2_z' + suffix] = relations_buffer['p2_pz' + suffix] / p2_p

    # Solution of differential eq. d2|DeltaR|/dt1 dt2
    relations_buffer['delta_r0_dot_u1' + suffix] = relations_buffer['delta_x0' + suffix] * relations_buffer[
        'u1_x' + suffix] + relations_buffer['delta_y0' + suffix] * relations_buffer['u1_y' + suffix] + relations_buffer[
                                                       'delta_z0' + suffix] * relations_buffer['u1_z' + suffix]
    relations_buffer['delta_r0_dot_u2' + suffix] = relations_buffer['delta_x0' + suffix] * relations_buffer[
        'u2_x' + suffix] + relations_buffer['delta_y0' + suffix] * relations_buffer['u2_y' + suffix] + relations_buffer[
                                                       'delta_z0' + suffix] * relations_buffer['u2_z' + suffix]
    relations_buffer['u1_dot_u2' + suffix] = relations_buffer['u1_x' + suffix] * relations_buffer['u2_x' + suffix] + \
                                             relations_buffer['u1_y' + suffix] * relations_buffer['u2_y' + suffix] + \
                                             relations_buffer['u1_z' + suffix] * relations_buffer['u2_z' + suffix]

    # Computation of the DOCA for the cases in which the momenta of the particles are parallel
    relations_case1 = relations_buffer[relations_buffer['u1_dot_u2' + suffix] == 1.].copy()
    relations_case1['DOCA' + suffix] = np.sqrt(
        relations_case1['delta_x0' + suffix] ** 2 + relations_case1['delta_y0' + suffix] ** 2 + relations_case1[
            'delta_z0' + suffix] ** 2)
    relations_case1.drop(
        columns=['delta_x0' + suffix, 'delta_y0' + suffix, 'u1_x' + suffix, 'u1_y' + suffix,
                 'u1_z' + suffix, 'u2_x' + suffix, 'u2_y' + suffix, 'u2_z' + suffix, 'delta_r0_dot_u1' + suffix,
                 'delta_r0_dot_u2' + suffix, 'u1_dot_u2' + suffix], inplace=True)

    # Computation of the doca in the rest of the cases
    relations_case2 = relations_buffer[relations_buffer['u1_dot_u2' + suffix] != 1.].copy()
    relations_case2['t1' + suffix] = - ((relations_case2['delta_r0_dot_u1' + suffix] + relations_case2[
        'u1_dot_u2' + suffix] * relations_case2['delta_r0_dot_u2' + suffix])
                                        / (1. - relations_case2['u1_dot_u2' + suffix] ** 2))
    relations_case2['t2' + suffix] = ((relations_case2['delta_r0_dot_u2' + suffix] - relations_case2[
        'u1_dot_u2' + suffix] * relations_case2['delta_r0_dot_u1' + suffix])
                                      / (1. - relations_case2['u1_dot_u2' + suffix] ** 2))

    # Vector distance at minimum
    relations_case2['delta_x' + suffix] = relations_case2['delta_x0' + suffix] + relations_case2['u1_x' + suffix] * \
                                          relations_case2['t1' + suffix] - relations_case2['u2_x' + suffix] * \
                                          relations_case2['t2' + suffix]
    relations_case2['delta_y' + suffix] = relations_case2['delta_y0' + suffix] + relations_case2['u1_y' + suffix] * \
                                          relations_case2['t1' + suffix] - relations_case2['u2_y' + suffix] * \
                                          relations_case2['t2' + suffix]
    relations_case2['delta_z' + suffix] = relations_case2['delta_z0' + suffix] + relations_case2['u1_z' + suffix] * \
                                          relations_case2['t1' + suffix] - relations_case2['u2_z' + suffix] * \
                                          relations_case2['t2' + suffix]
    relations_case2['DOCA' + suffix] = np.sqrt(
        relations_case2['delta_x' + suffix] ** 2 + relations_case2['delta_y' + suffix] ** 2 + relations_case2[
            'delta_z' + suffix] ** 2)
    relations_case2.drop(
        columns=['delta_x0' + suffix, 'delta_y0' + suffix, 'u1_x' + suffix, 'u1_y' + suffix, 'u1_z' + suffix,
                 'u2_x' + suffix, 'u2_y' + suffix, 'u2_z' + suffix, 'delta_r0_dot_u1' + suffix,
                 'delta_r0_dot_u2' + suffix, 'u1_dot_u2' + suffix, 't1' + suffix, 't2' + suffix, 'delta_x' + suffix,
                 'delta_y' + suffix, 'delta_z' + suffix], inplace=True)

    # Update the relations array by combining the two different cases
    df = pd.concat([relations_case1, relations_case2], ignore_index=True, sort=True)
    df["log_DOCA_reco"] = np.log(df["DOCA_reco"].values + 1e-5)
    return  df


def calculate_transversedist(relations_buffer, reco):
    if len(relations_buffer) == 0:
        return relations_buffer

    if reco:
        suffix = '_reco'
    else:
        suffix = ''

    relations_buffer['z_axis_prime_x'] = relations_buffer['p1_px' + suffix] + relations_buffer['p2_px' + suffix]
    relations_buffer['z_axis_prime_y'] = relations_buffer['p1_py' + suffix] + relations_buffer['p2_py' + suffix]
    relations_buffer['z_axis_prime_z'] = relations_buffer['p1_pz' + suffix] + relations_buffer['p2_pz' + suffix]

    relations_buffer['z_axis_prime_norm'] = np.vstack(np.linalg.norm(np.vstack(
        (relations_buffer['z_axis_prime_x'], relations_buffer['z_axis_prime_y'], relations_buffer['z_axis_prime_z'])).T,
                                                                     axis=1))
    relations_buffer['z_axis_prime_x'] = relations_buffer['z_axis_prime_x'] / relations_buffer['z_axis_prime_norm']
    relations_buffer['z_axis_prime_y'] = relations_buffer['z_axis_prime_y'] / relations_buffer['z_axis_prime_norm']
    relations_buffer['z_axis_prime_z'] = relations_buffer['z_axis_prime_z'] / relations_buffer['z_axis_prime_norm']

    relations_buffer['x_axis_prime_x'] = np.random.randn(relations_buffer.shape[0])
    relations_buffer['x_axis_prime_y'] = np.random.randn(relations_buffer.shape[0])
    relations_buffer['x_axis_prime_z'] = np.random.randn(relations_buffer.shape[0])

    relations_buffer['x_axis_prime_cdot_z_axis_prime'] = relations_buffer['x_axis_prime_x'] * relations_buffer[
        'z_axis_prime_x'] + relations_buffer['x_axis_prime_y'] * relations_buffer['z_axis_prime_y'] + relations_buffer[
                                                             'x_axis_prime_z'] * relations_buffer['z_axis_prime_z']
    relations_buffer['x_axis_prime_x'] -= relations_buffer['x_axis_prime_cdot_z_axis_prime'] * relations_buffer[
        'z_axis_prime_x']
    relations_buffer['x_axis_prime_y'] -= relations_buffer['x_axis_prime_cdot_z_axis_prime'] * relations_buffer[
        'z_axis_prime_y']
    relations_buffer['x_axis_prime_z'] -= relations_buffer['x_axis_prime_cdot_z_axis_prime'] * relations_buffer[
        'z_axis_prime_z']

    relations_buffer['x_axis_prime_norm'] = np.vstack(np.linalg.norm(np.vstack(
        (relations_buffer['x_axis_prime_x'], relations_buffer['x_axis_prime_y'], relations_buffer['x_axis_prime_z'])).T,
                                                                     axis=1))
    relations_buffer['x_axis_prime_x'] = relations_buffer['x_axis_prime_x'] / relations_buffer['x_axis_prime_norm']
    relations_buffer['x_axis_prime_y'] = relations_buffer['x_axis_prime_y'] / relations_buffer['x_axis_prime_norm']
    relations_buffer['x_axis_prime_z'] = relations_buffer['x_axis_prime_z'] / relations_buffer['x_axis_prime_norm']

    relations_buffer['y_axis_prime_x'] = relations_buffer['z_axis_prime_y'] * relations_buffer['x_axis_prime_z'] - \
                                         relations_buffer['z_axis_prime_z'] * relations_buffer['x_axis_prime_y']
    relations_buffer['y_axis_prime_y'] = relations_buffer['z_axis_prime_z'] * relations_buffer['x_axis_prime_x'] - \
                                         relations_buffer['z_axis_prime_x'] * relations_buffer['x_axis_prime_z']
    relations_buffer['y_axis_prime_z'] = relations_buffer['z_axis_prime_x'] * relations_buffer['x_axis_prime_y'] - \
                                         relations_buffer['z_axis_prime_y'] * relations_buffer['x_axis_prime_x']

    relations_buffer['x1_prime_x'] = relations_buffer['p1_xProd' + suffix] * relations_buffer['x_axis_prime_x']
    relations_buffer['x1_prime_y'] = relations_buffer['p1_yProd' + suffix] * relations_buffer['x_axis_prime_y']
    relations_buffer['x1_prime_z'] = relations_buffer['p1_zProd' + suffix] * relations_buffer['x_axis_prime_z']

    relations_buffer['y1_prime_x'] = relations_buffer['p1_xProd' + suffix] * relations_buffer['y_axis_prime_x']
    relations_buffer['y1_prime_y'] = relations_buffer['p1_yProd' + suffix] * relations_buffer['y_axis_prime_y']
    relations_buffer['y1_prime_z'] = relations_buffer['p1_zProd' + suffix] * relations_buffer['y_axis_prime_z']

    relations_buffer['x2_prime_x'] = relations_buffer['p2_xProd' + suffix] * relations_buffer['x_axis_prime_x']
    relations_buffer['x2_prime_y'] = relations_buffer['p2_yProd' + suffix] * relations_buffer['x_axis_prime_y']
    relations_buffer['x2_prime_z'] = relations_buffer['p2_zProd' + suffix] * relations_buffer['x_axis_prime_z']

    relations_buffer['y2_prime_x'] = relations_buffer['p2_xProd' + suffix] * relations_buffer['y_axis_prime_x']
    relations_buffer['y2_prime_y'] = relations_buffer['p2_yProd' + suffix] * relations_buffer['y_axis_prime_y']
    relations_buffer['y2_prime_z'] = relations_buffer['p2_zProd' + suffix] * relations_buffer['y_axis_prime_z']

    relations_buffer['trdist' + suffix] = np.sqrt(
        (relations_buffer['x1_prime_x'] - relations_buffer['x2_prime_x']) ** 2 + (
                relations_buffer['x1_prime_y'] - relations_buffer['x2_prime_y']) ** 2 + (
                relations_buffer['x1_prime_z'] - relations_buffer['x2_prime_z']) ** 2 + (
                relations_buffer['y1_prime_x'] - relations_buffer['y2_prime_x']) ** 2 + (
                relations_buffer['y1_prime_y'] - relations_buffer['y2_prime_y']) ** 2 + (
                relations_buffer['y1_prime_z'] - relations_buffer['y2_prime_z']) ** 2)

    relations_buffer.drop(
        columns=['z_axis_prime_x', 'z_axis_prime_y', 'z_axis_prime_z', 'z_axis_prime_norm',
                 'x_axis_prime_x', 'x_axis_prime_y', 'x_axis_prime_z', 'x_axis_prime_norm',
                 'x_axis_prime_cdot_z_axis_prime',
                 'y_axis_prime_x', 'y_axis_prime_y', 'y_axis_prime_z',
                 'x1_prime_x', 'x1_prime_y', 'x1_prime_z',
                 'y1_prime_x', 'y1_prime_y', 'y1_prime_z',
                 'x2_prime_x', 'x2_prime_y', 'x2_prime_z',
                 'y2_prime_x', 'y2_prime_y', 'y2_prime_z'], inplace=True)

    return relations_buffer


def calculate_angle(relations_buffer, reco):
    if len(relations_buffer) == 0:
        return relations_buffer

    if reco:
        suffix = '_reco'
    else:
        suffix = ''

    relations_buffer['p1_norm'] = np.sqrt(
        relations_buffer['p1_px' + suffix] ** 2 + relations_buffer['p1_py' + suffix] ** 2 + relations_buffer[
            'p1_pz' + suffix] ** 2)
    relations_buffer['p2_norm'] = np.sqrt(
        relations_buffer['p2_px' + suffix] ** 2 + relations_buffer['p2_py' + suffix] ** 2 + relations_buffer[
            'p2_pz' + suffix] ** 2)

    relations_buffer['cos_theta'] = (relations_buffer['p1_px' + suffix] * relations_buffer['p2_px' + suffix] +
                                     relations_buffer['p1_py' + suffix] * relations_buffer['p2_py' + suffix] +
                                     relations_buffer['p1_pz' + suffix] * relations_buffer['p2_pz' + suffix]) / (
                                            relations_buffer['p1_norm'] * relations_buffer['p2_norm'])

    relations_buffer['theta' + suffix] = np.arccos(np.float32(relations_buffer['cos_theta']))

    relations_buffer.drop(
        columns=['p1_norm', 'p2_norm', 'cos_theta'], inplace=True)

    return relations_buffer

def calculate_ip(stable_df, pv_df, reco_suffix: str = "", log_ip: bool = True):
    # Getting the unique reco pvs inside of the event
    pvs = torch.from_numpy(pv_df[[f'xPV{reco_suffix}', f'yPV{reco_suffix}', f'zPV{reco_suffix}']].to_numpy())
    pvs = torch.unique(pvs, dim=0).to(torch.float32)

    # Get the tracks prod position and momentum
    tracks_prod = stable_df[[f'xProd{reco_suffix}', f'yProd{reco_suffix}', f'zProd{reco_suffix}']]
    tracks_p = stable_df[[f'px{reco_suffix}', f'py{reco_suffix}', f'pz{reco_suffix}']]

    # Expand to match the shape of pvs
    tracks_prod = torch.from_numpy(tracks_prod.to_numpy()).unsqueeze(1).repeat(1, pvs.shape[0], 1)
    tracks_p = torch.from_numpy(tracks_p.to_numpy()).unsqueeze(1).repeat(1, pvs.shape[0], 1)

    r = tracks_prod - pvs
    ips = torch.sqrt(
        torch.sum(r ** 2, dim=-1) - torch.sum(tracks_p * r, dim=-1) ** 2 / torch.sum(tracks_p ** 2, dim=-1))
    ips[torch.isnan(ips)] = 0  # if something is negative in the sqrt it results in nan replacing it with 0

    if log_ip:
        ips = torch.log(ips + 1.e-6)
    return ips