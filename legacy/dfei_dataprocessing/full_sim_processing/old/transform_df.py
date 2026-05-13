import numpy as np
import pandas as pd

import variables_definitions as v

# neutral: gamma, pi0, K_L, nu_e, nu_mu, nu_tau
neutrals_list = [22, 111, 130, 12, 14, 16]

# stable particles:
stable_list = [211, 321, 2212, 11, 13]


class DfCreator:
    def __init__(self, events, evt_counter):
        self.events = events
        self.df = None
        self.event = None
        self.evt_counter = evt_counter
        self.internal_counter = 0

    def create_df(self):
        type_keys = v.part_keys
        df_keys, B_keys, extra_keys = np.array(v.df_keys).T

        df_dict = {}
        for key in df_keys:
            df_dict[key] = []

        for key in type_keys:
            for i, feature in enumerate(df_keys):
                if "B" in key:
                    df_dict[feature].append(self.event[key][f"{key}_{B_keys[i]}"])
                else:
                    if extra_keys[i] == "":
                        df_dict[feature].append(
                            self.event[key][f"{key}_{extra_keys[0]}"] * 0)  # default value to 0, should be fine
                    else:
                        df_dict[feature].append(self.event[key][f"{key}_{extra_keys[i]}"])

        for key in df_keys:
            df_dict[key] = np.nan_to_num(np.concatenate(df_dict[key]), nan=-999)

        df = pd.DataFrame(df_dict)
        df = df.drop_duplicates(subset='ParticleIndex', keep='last')
        self.df = df

    @staticmethod
    def get_decay_graph(df, mode="full"):
        decay_heads = df[df['decay_head'] == True]['ParticleIndex'].values
        if len(decay_heads) == 0:
            raise ValueError("No decay_head found in DataFrame")
        # print(f"Decay heads: {decay_heads}")

        filtered_df = df.copy()

        if "no_neutrals" in mode:
            # remove nu, gamma, pi, KL
            neutral_mask = filtered_df['id'].isin(neutrals_list)
            filtered_df = filtered_df[~neutral_mask]

        graph = {}

        def add_children_recursive(parent_idx):
            children = filtered_df[filtered_df['MotherIndex'] == parent_idx]

            if len(children) > 0:
                child_indices = children['ParticleIndex'].values
                graph[int(parent_idx)] = [int(i) for i in child_indices]

                if "no_stable_decay" in mode:
                    # Filter out pi+ (211) and K+ (321)
                    selbool = children["id"].abs().isin(stable_list)
                    children = children[~selbool]

                    # remove the stable one from the indices to be passed on
                    child_indices = children['ParticleIndex'].values

                # Recursively process each child
                for child_idx in child_indices:
                    add_children_recursive(child_idx)

        for start_idx in decay_heads:
            add_children_recursive(start_idx)

        return graph

    @staticmethod
    def get_decay_chain_particles(graph, decay_head_idx):

        particles = [decay_head_idx]

        def collect_descendants(parent_idx):
            """Recursively collect all descendant particles."""
            if parent_idx in graph:
                for child_idx in graph[parent_idx]:
                    particles.append(child_idx)
                    collect_descendants(child_idx)

        collect_descendants(decay_head_idx)
        return particles

    def df_cleanup(self):
        # removing particles which come or are of id -1 or mother is nan
        removal_bool = (self.df["ParticleIndex"] == -1) | (self.df["MotherIndex"] == -1) | (
                self.df["MotherIndex"] == -999)

        removed_df = self.df[removal_bool]
        part_df = self.df[~removal_bool].copy()

        # Checking is a particle is part of a decay
        part_df["in_decay"] = part_df["MotherIndex"].isin(part_df["ParticleIndex"]) | part_df["ParticleIndex"].isin(
            part_df["MotherIndex"])

        finals_bkg_df = part_df[~part_df["in_decay"]].copy()
        charge_selbool = finals_bkg_df["isCharged"] == 1
        # print("Number of neutral in bkg final df:", np.sum(~charge_selbool))

        # Create a decay df and decay heads
        decay_df = part_df[part_df["in_decay"]].copy()
        decay_df["decay_head"] = ~decay_df["MotherIndex"].isin(decay_df["ParticleIndex"])

        graph = self.get_decay_graph(decay_df, mode="no_neutrals_no_stable_decay")

        b_df = decay_df[decay_df["decay_head"] == True]

        mothers = np.array(list(graph.keys()))
        daughters = np.concatenate(list(graph.values()))
        in_graph = np.unique(np.concatenate([mothers, daughters]))

        filtered_decay_df = decay_df[decay_df["ParticleIndex"].isin(in_graph)]

        b_final_df = filtered_decay_df[~filtered_decay_df["ParticleIndex"].isin(filtered_decay_df["MotherIndex"])]
        b_inter_df = filtered_decay_df[filtered_decay_df["ParticleIndex"].isin(filtered_decay_df["MotherIndex"])]
        b_inter_df = b_inter_df[b_inter_df["decay_head"] == False]

        # list of df
        list_of_df = [b_df, b_inter_df, b_final_df, finals_bkg_df]
        final_df = pd.concat(list_of_df).reset_index()

        final_df["EventNumber"] = self.evt_counter
        final_df["PrimaryHeavyHadronIndex"] = -1
        # here we need to chang it that the head does not have itself
        for head in final_df[final_df['decay_head'] == True]['ParticleIndex'].values:
            part_of_decay = self.get_decay_chain_particles(graph, head)
            part_of_decay.remove(head)  # removing the head as it shouldnt be classified with index
            final_df.loc[
                final_df['ParticleIndex'].isin(part_of_decay),
                'PrimaryHeavyHadronIndex'
            ] = head

        final_df = final_df.drop(columns=["in_decay", "decay_head"])

        return final_df

    def get_df(self, event_number):
        try:
            self.event = self.events[self.internal_counter]
        except IndexError:
            import pdb; pdb.set_trace()
        self.create_df()
        final_df = self.df_cleanup()
        self.evt_counter += event_number
        self.internal_counter += event_number
        return final_df
