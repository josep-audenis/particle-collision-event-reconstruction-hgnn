import numpy as np
import pandas as pd


def compute_pv_asso(particle: pd.DataFrame, pvs: np.array([]), mode: str) -> pd.DataFrame:
    suffix = ""
    if mode == "reco":
        suffix = "_reco"

    p = np.sqrt(particle['px' + suffix] ** 2 + particle['py' + suffix] ** 2 + particle['pz' + suffix] ** 2)
    px_norm = particle['px' + suffix] / p
    py_norm = particle['py' + suffix] / p
    pz_norm = particle['pz' + suffix] / p

    xpv, ypv, zpv = pvs.T
    xpv = np.resize(xpv, (particle.shape[0], xpv.shape[0]))
    ypv = np.resize(ypv, (particle.shape[0], ypv.shape[0]))
    zpv = np.resize(zpv, (particle.shape[0], zpv.shape[0]))

    A = ((particle['xProd' + suffix].values[:, np.newaxis] - xpv) ** 2 +
         (particle['yProd' + suffix].values[:, np.newaxis] - ypv) ** 2 +
         (particle['zProd' + suffix].values[:, np.newaxis] - zpv) ** 2)

    B = ((particle['xProd' + suffix].values[:, np.newaxis] - xpv) * px_norm.values[:, np.newaxis] +
         (particle['yProd' + suffix].values[:, np.newaxis] - ypv) * py_norm.values[:, np.newaxis] +
         (particle['zProd' + suffix].values[:, np.newaxis] - zpv) * pz_norm.values[:, np.newaxis])

    C = px_norm ** 2 + py_norm ** 2 + pz_norm ** 2

    ip = A - B * B / C.values[:, np.newaxis]
    ip[ip < 0] = 0
    ip = np.sqrt(ip)

    asso_xpv, asso_ypv, asso_zpv = pvs[np.argmin(ip, axis=1)].T
    asso_ip = np.min(ip, axis=1)

    particle['associated_xPV' + suffix] = asso_xpv
    particle['associated_yPV' + suffix] = asso_ypv
    particle['associated_zPV' + suffix] = asso_zpv
    particle['associated_ip' + suffix] = asso_ip
    return particle


def compute_particle_PID(particles: pd.DataFrame, mode: str):
    if mode == "onehot":
        particles['ParticleRecoType'] = np.array(particles['isCharged'] == 0, dtype=np.int32)
        particles['Prob_pi'] = np.array(np.abs(particles['id']) == 211, dtype=np.int32)
        particles['Prob_k'] = np.array(np.abs(particles['id']) == 321, dtype=np.int32)
        particles['Prob_p'] = np.array(np.abs(particles['id']) == 2212, dtype=np.int32)
        particles['Prob_e'] = np.array(np.abs(particles['id']) == 11, dtype=np.int32)
        particles['Prob_mu'] = np.array(np.abs(particles['id']) == 13, dtype=np.int32)
        particles['Prob_gamma'] = np.array(np.abs(particles['id']) == 22, dtype=np.int32)
        particles['Prob_pi0'] = np.array(np.abs(particles['id']) == 111, dtype=np.int32)
    if mode == "LHCb":
        particles['ParticleRecoType'] = np.array(particles['isCharged'] == 0, dtype=np.int32)
        particles['Prob_pi'] = particles["Prob_pi"].values
        particles['Prob_k'] = particles["Prob_k"].values
        particles['Prob_p'] = particles["Prob_p"].values
        particles['Prob_e'] = particles["Prob_e"].values
        particles['Prob_mu'] = particles["Prob_mu"].values
        particles['Prob_ghost'] = particles["Prob_ghost"].values
        particles['Prob_gamma'] = particles["Prob_pi"].values * 0
        particles['Prob_pi0'] = particles[
                                    "Prob_pi"].values * 0  # currently dont hold that information  since no neutral

    return particles


def calculate_LCA(relations_buffer, prefix=''):
    def LCA(anc1, anc2, max_depth, anc1_ids, anc2_ids):
        if (anc1 == []) or (anc2 == []):
            return [0, 0]

        _, _, indexes = np.intersect1d(anc1, anc2, return_indices=True)
        common_ancestors = [anc2[i] for i in indexes]

        if (common_ancestors == []):
            return [0, 0]

        lowest_common_ancestor = common_ancestors[-1]
        if (len(anc1) >= len(anc2)):
            max_length = anc1
            max_length_ids = anc1_ids
        else:
            max_length = anc2
            max_length_ids = anc2_ids
        lowest_common_ancestor_generation = max_length.index(lowest_common_ancestor)

        return [max_depth - lowest_common_ancestor_generation, max_length_ids[lowest_common_ancestor_generation]]

    if len(relations_buffer) == 0:
        return relations_buffer

    # Maximum expected value for the LCA components, used to translate the integer LCA values into a one-hot encoder
    max_LVA_value = 10

    # Select the pairs of particles that come from the same primary heavy hadron
    signal_pairs = relations_buffer[relations_buffer['FromSameHeavyHadron'] == 1].copy()
    # Compute the LCA values for the signal pairs
    if len(signal_pairs) > 0:
        LCA_output = np.array(signal_pairs.apply(
            lambda x: LCA(x['p1_' + prefix + 'MotherList'], x['p2_' + prefix + 'MotherList'],
                          x[prefix + 'MaxDecayChainLength'], x['p1_' + prefix + 'MotherList_ids'],
                          x['p2_' + prefix + 'MotherList_ids']), axis=1).values.tolist())
        signal_pairs[prefix + 'LCA_values'] = LCA_output[:, 0]
        signal_pairs[prefix + 'LCA_ids'] = LCA_output[:, 1]
        for i in range(max_LVA_value + 1):
            signal_pairs[prefix + 'LCA_class' + str(i)] = np.array(
                (signal_pairs[prefix + 'LCA_values'].to_numpy(dtype=np.int32) == i), dtype=np.int32)

    # Port the previous information to the whole set of pairs, zero-padding where needed
    relations_buffer[prefix + 'LCA'] = np.array(0, dtype=np.int32)
    relations_buffer[prefix + 'LCA_ids'] = np.array(0, dtype=np.int32)
    if len(signal_pairs) > 0:
        relations_buffer.loc[(relations_buffer['FromSameHeavyHadron'] == 1), prefix + 'LCA'] = signal_pairs[
            prefix + 'LCA_values'].to_numpy(dtype=np.int32)
        relations_buffer.loc[(relations_buffer['FromSameHeavyHadron'] == 1), prefix + 'LCA_ids'] = signal_pairs[
            prefix + 'LCA_ids'].to_numpy(dtype=np.int32)
    for i in range(max_LVA_value + 1):
        relations_buffer[prefix + 'LCA_class' + str(i)] = np.array(0, dtype=np.int32)
        if len(signal_pairs) > 0:
            relations_buffer.loc[(relations_buffer['FromSameHeavyHadron'] == 1), prefix + 'LCA_class' + str(i)] = \
                signal_pairs[prefix + 'LCA_class' + str(i)].to_numpy(dtype=np.int32)

    # All the background-like pairs will have a LVA value equal to 0
    relations_buffer.loc[(relations_buffer['FromSameHeavyHadron'] == 0), prefix + 'LCA_class0'] = np.array(1,
                                                                                                           dtype=np.int32)

    return relations_buffer


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
    return pd.concat([relations_case1, relations_case2], ignore_index=True, sort=True)


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
