import pandas as pd
import numpy as np


def ensure_decay_chain_key(decay_dict: dict, head_index: float) -> None:
    decay_dict["max_decay_chain_lengths"].setdefault(head_index, 0)
    decay_dict["topo_max_decay_chain_lengths"].setdefault(head_index, 0)


def lookup_scalar(df: pd.DataFrame, index_col: str, index_value: float, value_col: str):
    match = df.loc[df[index_col] == index_value, value_col]
    if match.empty:
        return None
    return match.iloc[0]


def remove_short_lived_ancestors(ancestors: list, unstable_df: pd.DataFrame, hh_df: pd.DataFrame,
                                 num_vis_daugh_map: pd.DataFrame) -> list:



    min_gammactau = 0.005
    if not ancestors:
        return []

    reduced_ancestors = []

    for anc in ancestors[:-1]:
        gammactau = lookup_scalar(unstable_df, 'ExpandedParticleIndex', anc, 'gammactau')
        if gammactau is not None and gammactau >= min_gammactau:
            reduced_ancestors.append(anc)

    gammactau = lookup_scalar(hh_df, 'ExpandedParticleIndex', ancestors[-1], 'gammactau')
    if gammactau is not None and gammactau >= min_gammactau:
        reduced_ancestors.append(ancestors[-1])










    for anc in ancestors[:-1]:
        if anc not in reduced_ancestors:
            mother = lookup_scalar(unstable_df, 'ExpandedParticleIndex', anc, 'ExpandedMotherIndex')
            if mother in num_vis_daugh_map.index and anc in num_vis_daugh_map.index:
                num_vis_daugh_map.loc[mother, 'NumVisDaugh'] = num_vis_daugh_map.loc[mother, 'NumVisDaugh'] + \
                                                               num_vis_daugh_map.loc[anc, 'NumVisDaugh']
    return reduced_ancestors


def remove_ancestors_without_vertex(ancestors: list, num_vis_daugh_map: pd.DataFrame) -> list:




    min_numvisdaugh = 2
    if ancestors == []:
        return []
    reduced_ancestors = []
    for anc in ancestors:
        if num_vis_daugh_map.loc[anc, 'NumVisDaugh'] >= min_numvisdaugh:
            reduced_ancestors.append(anc)
    return reduced_ancestors


def compute_lists_of_mothers(particle_df: pd.DataFrame, unstable_df: pd.DataFrame, hh_df: pd.DataFrame,
                             hh_indices: list, decay_dict: dict) -> dict:

    ids_df = pd.DataFrame(pd.concat([pd.DataFrame(unstable_df, columns=['ExpandedParticleIndex', 'id']),
                                     pd.DataFrame(hh_df, columns=['ExpandedParticleIndex', 'id'])]))
    ids_df = ids_df.set_index('ExpandedParticleIndex')
    known_ancestor_indices = set(ids_df.index)


    sig_df = particle_df[particle_df["ExpandedPrimaryHeavyHadronIndex"].isin(hh_indices)].copy()


    mother_map = pd.DataFrame(sig_df, columns=['ExpandedParticleIndex', 'ExpandedMotherIndex'])
    mother_map["ExpandedMother1"] = mother_map["ExpandedMotherIndex"]
    mother_map.drop(columns=['ExpandedMotherIndex'], inplace=True)



    current_list_of_mothers = unstable_df[unstable_df['ExpandedParticleIndex'].isin(set(mother_map['ExpandedMother1']))]


    counter = 2
    while current_list_of_mothers.shape[0] > 0:
        if counter > 20:
            raise RuntimeError(
                "Mother chain depth exceeded 20 generations. "
                "This usually indicates a cycle or malformed ancestry in the input event."
            )
        prev_gen = f'ExpandedMother{counter - 1}'
        curr_gen = f'ExpandedMother{counter}'


        mothers_to_grandmothers = pd.Series(current_list_of_mothers.ExpandedMotherIndex.values,
                                            index=current_list_of_mothers.ExpandedParticleIndex.values).to_dict()

        mothers_to_grandmothers.update({id_: 0 for id_ in hh_indices})


        mother_map[curr_gen] = mother_map[prev_gen].map(mothers_to_grandmothers)


        mother_map['ExpandedMother' + str(counter)] = mother_map['ExpandedMother' + str(counter - 1)].map(
            mothers_to_grandmothers)


        unique_mothers = mother_map[curr_gen].unique()
        current_list_of_mothers = unstable_df[
            unstable_df['ExpandedParticleIndex'].isin(unique_mothers)
        ]

        counter += 1




    num_vis_daugh_map = pd.DataFrame(pd.concat(
        [pd.DataFrame(unstable_df, columns=['ExpandedParticleIndex', 'NumVisDaugh']),
         pd.DataFrame(hh_df, columns=['ExpandedParticleIndex', 'NumVisDaugh'])]))
    num_vis_daugh_map = num_vis_daugh_map.set_index('ExpandedParticleIndex')

    for irow in range(mother_map.shape[0]):
        pruned_list = [x for x in list(mother_map.iloc[irow]) if x > 0]
        if not pruned_list:
            continue

        chain_head = pruned_list[-1]
        ensure_decay_chain_key(decay_dict, chain_head)
        decay_dict["max_decay_chain_lengths"][chain_head] = max(
            decay_dict["max_decay_chain_lengths"][chain_head], len(pruned_list) - 1)

        ancestor_list = [ancestor for ancestor in pruned_list[1:] if ancestor in known_ancestor_indices]





        if sig_df[sig_df['ExpandedParticleIndex'] == pruned_list[0]]['isCharged'].iloc[0] == 0:
            topo_ancestor_list = []
        else:
            ancestor_pruned = remove_short_lived_ancestors(ancestor_list, unstable_df, hh_df, num_vis_daugh_map)
            topo_ancestor_list = remove_ancestors_without_vertex(ancestor_pruned, num_vis_daugh_map)
        decay_dict["topo_max_decay_chain_lengths"][chain_head] = max(
            decay_dict["topo_max_decay_chain_lengths"][chain_head], len(topo_ancestor_list))


        ancestor_list.reverse()
        decay_dict["ancestors_per_particle"][pruned_list[0]] = ancestor_list
        topo_ancestor_list.reverse()
        decay_dict["topo_ancestors_per_particle"][pruned_list[0]] = topo_ancestor_list
        decay_dict["ancestors_ids_per_particle"][pruned_list[0]] = ids_df.loc[ancestor_list]['id'].to_list()
        decay_dict["topo_ancestors_ids_per_particle"][pruned_list[0]] = ids_df.loc[topo_ancestor_list]['id'].to_list()

    return decay_dict


def compute_mother_PID(stable_charged_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices):

    signal_particles = pd.DataFrame(
        stable_charged_df[stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)],
        columns=['ExpandedMotherIndex'])


    mother_particles = pd.concat(
        [unstable_from_hh_df[["id", "ExpandedParticleIndex"]], heavy_hadron_df[["id", "ExpandedParticleIndex"]]],
        ignore_index=True, sort=True)
    mother_index_to_id = pd.Series(mother_particles.id.values,
                                   index=mother_particles.ExpandedParticleIndex.values).to_dict()


    signal_particles['mother_id'] = signal_particles['ExpandedMotherIndex'].map(mother_index_to_id)


    stable_charged_df['MotherId'] = np.array(0, dtype=np.int32)
    stable_charged_df.loc[
        (stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)), 'MotherId'] = \
        signal_particles['mother_id'].to_numpy(dtype=np.int32)



    b_hadron_dict = dict(zip(
        heavy_hadron_df.loc[heavy_hadron_df["isbhadron"] == 1, "ParticleIndex"],
        np.sign(heavy_hadron_df.loc[heavy_hadron_df["isbhadron"] == 1, "id"])
    ))

    stable_charged_df['isfromb'] = stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].map(b_hadron_dict).fillna(
        0).astype(np.int32)

    return stable_charged_df
