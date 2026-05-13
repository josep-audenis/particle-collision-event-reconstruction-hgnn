import pandas as pd
import numpy as np


def remove_short_lived_ancestors(ancestors: list, unstable_df: pd.DataFrame, hh_df: pd.DataFrame,
                                 num_vis_daugh_map: pd.DataFrame) -> list:
    # Function used to remove short-lived ancestors from a list, updating the visible-daughters mapping accordingly
    # This is the first step to identify the "topo" ancestors
    # Thresholds used to define the topo ancestors
    min_gammactau = 0.005  # mm
    reduced_ancestors = []
    # Inspect the intermediate particles
    for anc in ancestors[:-1]:
        if unstable_df[unstable_df['ExpandedParticleIndex'] == anc]['gammactau'].iloc[0] >= min_gammactau:
            reduced_ancestors.append(anc)
    # Inspect the primary heavy hadron (always the last ancestor at this stage)
    if hh_df[hh_df['ExpandedParticleIndex'] == ancestors[-1]]['gammactau'].iloc[0] >= min_gammactau:
        reduced_ancestors.append(ancestors[-1])
    # This line above is quite interesting. Looking at the case 515 -> 521 + 211
    # The 515 has gammactau of 0 thus the 211 will be removed from the decay chain and the head becomes 521
    # However 211 is still part of the 515 decay chain and is hence marked as a signal particle
    # In the end we only care about the topological information for the GNN
    # the PrimaryHeavyHadronIndex will be renamed to LegacyPrimaryHeavyHadronIndex
    # and the PrimaryHeavyHadronIndex based on topological information will be added
    # During the cacheing signal information is obtained from ExpandedPrimaryHeavyHadronIndex key and thus this change is needed

    # Update the effective number of visible daughters of the other mothers, to reflect the collapse of particles
    # (Note: if the primary heavy hadron gets removed, don't update the dictionary)
    for anc in ancestors[:-1]:
        if anc not in reduced_ancestors:
            mother = unstable_df[unstable_df['ExpandedParticleIndex'] == anc]['ExpandedMotherIndex'].iloc[
                0]
            num_vis_daugh_map.loc[mother, 'NumVisDaugh'] = num_vis_daugh_map.loc[mother, 'NumVisDaugh'] + \
                                                           num_vis_daugh_map.loc[anc, 'NumVisDaugh']
    return reduced_ancestors


def remove_ancestors_without_vertex(ancestors: list, num_vis_daugh_map: pd.DataFrame) -> list:
    # Function used to remove ancestors that don't have a reconstruction vertex from a list
    # This is the second step to identify the "topo" ancestors,
    # running on a list from which the short-lived ancestors have already been removed
    # Thresholds used to define the topo ancestors
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
    #
    ids_df = pd.DataFrame(pd.concat([pd.DataFrame(unstable_df, columns=['ExpandedParticleIndex', 'id']),
                                     pd.DataFrame(hh_df, columns=['ExpandedParticleIndex', 'id'])]))
    ids_df = ids_df.set_index('ExpandedParticleIndex')

    # Obtain signal stable particles
    sig_df = particle_df[particle_df["ExpandedPrimaryHeavyHadronIndex"].isin(hh_indices)].copy()

    # Map with chain of ancestors IDs to stable particles
    mother_map = pd.DataFrame(sig_df, columns=['ExpandedParticleIndex', 'ExpandedMotherIndex'])
    mother_map["ExpandedMother1"] = mother_map["ExpandedMotherIndex"]
    mother_map.drop(columns=['ExpandedMotherIndex'], inplace=True)

    # First level of ancestors of signal which are not heavy hadrons
    # Identify the first level of mother particles (which are not primary heavy hadrons)
    current_list_of_mothers = unstable_df[unstable_df['ExpandedParticleIndex'].isin(set(mother_map['ExpandedMother1']))]

    # Create a df containing all the mother generation lists
    counter = 2
    while current_list_of_mothers.shape[0] > 0:
        if counter > 20:
            import pdb; pdb.set_trace()
        prev_gen = f'ExpandedMother{counter - 1}'
        curr_gen = f'ExpandedMother{counter}'

        # Create a mapping between the mother index and the grandmother index, for the current level
        mothers_to_grandmothers = pd.Series(current_list_of_mothers.ExpandedMotherIndex.values,
                                            index=current_list_of_mothers.ExpandedParticleIndex.values).to_dict()
        # Expand the mapping to include (grand ...) mothers of primary heavy hadrons, that will be assigned a value of 0
        mothers_to_grandmothers.update({id_: 0 for id_ in hh_indices})

        # Identify the next level of mother particles, using the previous mapping
        mother_map[curr_gen] = mother_map[prev_gen].map(mothers_to_grandmothers)
        #mother_map = mother_map.copy()

        mother_map['ExpandedMother' + str(counter)] = mother_map['ExpandedMother' + str(counter - 1)].map(
            mothers_to_grandmothers)

        # Read the next level of mother particles, and update the counter
        unique_mothers = mother_map[curr_gen].unique()
        current_list_of_mothers = unstable_df[
            unstable_df['ExpandedParticleIndex'].isin(unique_mothers)
        ]

        counter += 1

    # Create a mapping to store the number of "visible" (not stable neutral) daughters of a given particle
    # (this mapping will be updated if any of the intermediate particles is "collapsed",
    # to reflect the effective number of visible daughters at the given level)
    num_vis_daugh_map = pd.DataFrame(pd.concat(
        [pd.DataFrame(unstable_df, columns=['ExpandedParticleIndex', 'NumVisDaugh']),
         pd.DataFrame(hh_df, columns=['ExpandedParticleIndex', 'NumVisDaugh'])]))
    num_vis_daugh_map = num_vis_daugh_map.set_index('ExpandedParticleIndex')

    for irow in range(mother_map.shape[0]):
        pruned_list = [x for x in list(mother_map.iloc[irow]) if x > 0]

        try:
            decay_dict["max_decay_chain_lengths"][pruned_list[-1]] = max(
                decay_dict["max_decay_chain_lengths"][pruned_list[-1]], len(pruned_list) - 1)
        except Exception as e:
            print(f'Failed: {type(e).__name__}: {e}')
            print(f'pruned_list = {pruned_list}')
            print(f'mother_map row = {list(mother_map.iloc[irow])}')
            import pdb; pdb.set_trace()
            return 0

        ancestor_list = list(pruned_list[1:])

        # Obtain the list of topo ancestors and update the maximum length of the associated decay chain
        # If the particle is neutral, the list of topo ancestors is manually set to be empty
        # Note: even in the case in which the primary hadron ancestor has been removed,
        # its index is used for bookkeeping (identifier of the chain)
        if sig_df[sig_df['ExpandedParticleIndex'] == pruned_list[0]]['isCharged'].iloc[0] == 0:
            topo_ancestor_list = []
        else:
            ancestor_pruned = remove_short_lived_ancestors(ancestor_list, unstable_df, hh_df, num_vis_daugh_map)
            topo_ancestor_list = remove_ancestors_without_vertex(ancestor_pruned, num_vis_daugh_map)
        decay_dict["topo_max_decay_chain_lengths"][pruned_list[-1]] = max(
            decay_dict["topo_max_decay_chain_lengths"][pruned_list[-1]], len(topo_ancestor_list))

        # Fill the dictionaries with the list of ancestors per particle
        ancestor_list.reverse()
        decay_dict["ancestors_per_particle"][pruned_list[0]] = ancestor_list
        topo_ancestor_list.reverse()
        decay_dict["topo_ancestors_per_particle"][pruned_list[0]] = topo_ancestor_list
        decay_dict["ancestors_ids_per_particle"][pruned_list[0]] = ids_df.loc[ancestor_list]['id'].to_list()
        decay_dict["topo_ancestors_ids_per_particle"][pruned_list[0]] = ids_df.loc[topo_ancestor_list]['id'].to_list()

    return decay_dict


def compute_mother_PID(stable_charged_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices):
    # Select the signal stable particles
    signal_particles = pd.DataFrame(
        stable_charged_df[stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)],
        columns=['ExpandedMotherIndex'])

    # Select the mother particles and create a dictionary to access their ids
    mother_particles = pd.concat(
        [unstable_from_hh_df[["id", "ExpandedParticleIndex"]], heavy_hadron_df[["id", "ExpandedParticleIndex"]]],
        ignore_index=True, sort=True)
    mother_index_to_id = pd.Series(mother_particles.id.values,
                                   index=mother_particles.ExpandedParticleIndex.values).to_dict()

    # Assign the mother ids to the signal particles
    signal_particles['mother_id'] = signal_particles['ExpandedMotherIndex'].map(mother_index_to_id)

    # Port the previous information to all the stable particles, zero-padding where needed
    stable_charged_df['MotherId'] = np.array(0, dtype=np.int32)
    stable_charged_df.loc[
        (stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)), 'MotherId'] = \
        signal_particles['mother_id'].to_numpy(dtype=np.int32)

    # Add the particle is from b bbar hadron
    # Create dictionary of B-hadron indices to +1/-1
    b_hadron_dict = dict(zip(
        heavy_hadron_df.loc[heavy_hadron_df["isbhadron"] == 1, "ParticleIndex"],
        np.sign(heavy_hadron_df.loc[heavy_hadron_df["isbhadron"] == 1, "id"])
    ))
    # Map values
    stable_charged_df['isfromb'] = stable_charged_df['ExpandedPrimaryHeavyHadronIndex'].map(b_hadron_dict).fillna(
        0).astype(np.int32)

    return stable_charged_df
