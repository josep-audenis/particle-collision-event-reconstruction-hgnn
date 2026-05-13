from skhep.math import Vector3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import yaml
import uproot
#from workspace_configuration.profile_configuration import PYTHIA_tuples_path


#---Create my custom env.
class DecayChainReconstructionEnv():
    def __init__(self, name="my_env", use_reco_vars = 0, validation_fraction = -1, simulation_type_ = 0, use_edge_prefiltering_ = 0, sim_file_name_ = "", ft_bool=False, evt_max_ = None):
        self.name = name

        self.current_evt = -1

        # Configurables
        self.num_evs_in_buffer = 1000
        self.why = -1
        # if evt_max_ is not None and self.num_evs_in_buffer > evt_max_:
        #     self.num_evs_in_buffer = int(evt_max_/2)
        self.validation_dataset_fraction = validation_fraction
        self.simulation_type = simulation_type_
        self.use_edge_prefiltering = use_edge_prefiltering_

        if self.use_edge_prefiltering:
            print('Edge pre-filtering being applied.')
        else:
            print('Edge pre-filtering not being applied.')

        if sim_file_name_ != "":
            self.sim_file_name = sim_file_name_
        else:
            if self.simulation_type == 0: # PYTHIA-based MC
                self.sim_file_name = "Event_data_particles_bcquark_nu7_6_49405events_allbkgtracks_events_47500_to_47999.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_49399events_allbkgtracks_events_0_to_49398.root" # training
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_49194events_allbkgtracks_events_0_to_49193.root" # evaluation
                #self.sim_file_name = "Event_data_particles_bquark_Bd_Kpimumu_nu7_6_5000events_allbkgtracks_events_0_to_4999.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_49399events_0bkgtracks_events_0_to_49398.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_6153events_allbkgtracks_events_0_to_6152.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_6162events_allbkgtracks_events_0_to_6161.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_6162events_0bkgtracks_events_0_to_6161.root"
                #self.sim_file_name = "Event_data_particles_bquark_nu7_6_100events_0bkgtracks_events_0_to_99.root"
            else: # full MC
                self.sim_file_name = "Event_data_particles_bquark_Run2_FullMC_10000000_195965events_allbkgtracks_events_0_to_195964.root"
        print('Using file', self.sim_file_name)

        self.input_particles_columns = ['ExpandedEventNumber','ExpandedParticleIndex','ExpandedPrimaryHeavyHadronIndex','ParticleType','charge','Prob_k','Prob_e','Prob_gamma','Prob_mu','Prob_p','Prob_pi','Prob_pi0','associated_ip_reco','associated_xPV_reco','associated_yPV_reco','associated_zPV_reco', 'xPV_reco','yPV_reco','zPV_reco','inGeomAcc','isCharged', 'isbhadron', 'pt_reco','px_reco','py_reco','pz_reco','xProd_reco','yProd_reco','zProd_reco','id', 'nFinals']
        self.input_particles_column_types = {'ExpandedParticleIndex': int, 'ExpandedPrimaryHeavyHadronIndex': int, 'ParticleType':int, 'Prob_k': bool, 'Prob_e': bool, 'Prob_gamma': bool, 'Prob_mu': bool, 'Prob_p': bool, 'Prob_pi': bool, 'Prob_pi0': bool, 'inGeomAcc': bool, 'isCharged': bool, 'isbhadron': int, 'charge': int, 'id': int, 'nFinals': int}
        if ft_bool:
            self.input_particles_columns += ["isfromb", "isFrag", 'fragType', 'fragPartnerIndex']
            self.input_particles_column_types['isfromb'] = int
            self.input_particles_column_types['isFrag'] = int
            self.input_particles_column_types['fragType'] = int
            self.input_particles_column_types['fragPartnerIndex'] = int
        if self.simulation_type == 1:
            self.input_particles_columns += ['ParticleRecoType', 'cand_B0_ISOLATION_BDT', 'Part_ISO_InSignalCandidate']
            self.input_particles_column_types['ParticleRecoType'] = int
            self.input_particles_column_types['cand_B0_ISOLATION_BDT'] = float
            self.input_particles_column_types['Part_ISO_InSignalCandidate'] = int

        self.input_relations_columns = ['FromSameAssociatedPV_reco','BothInGeomAcc','DOCA_reco','ExpandedEventNumber','FirstParticleIndex','FromSameHeavyHadron','SecondParticleIndex','delta_z0_reco','p1_isCharged','p2_isCharged','theta_reco','trdist_reco','TopoLCA_class0', 'TopoLCA_class1','TopoLCA_class2','TopoLCA_class3','LCA_ids','TopoLCA_ids','LCA']
        self.input_relations_column_types = {'FromSameAssociatedPV_reco': bool, 'BothInGeomAcc': bool, 'ExpandedEventNumber': int, 'FirstParticleIndex': int, 'FromSameHeavyHadron': bool, 'SecondParticleIndex': int, 'p1_isCharged': bool, 'p2_isCharged': bool, 'TopoLCA_class0': bool, 'TopoLCA_class1': bool,'TopoLCA_class2': bool,'TopoLCA_class3': bool, 'LCA_ids': int, 'TopoLCA_ids': int, 'LCA': int}
        if self.simulation_type == 1:
            self.input_relations_columns += ['p1_ParticleRecoType', 'p2_ParticleRecoType']
            self.input_relations_column_types['p1_ParticleRecoType'] = int
            self.input_relations_column_types['p2_ParticleRecoType'] = int

        #self.trigger_mapping_file_name = 'Trigger_mapping_bquark_nu7_6_6162events.yaml'
        self.include_neutrals = 0
        if use_reco_vars:
            self.reco_suffix = '_reco'
        else:
            self.reco_suffix = ''
        self.current_event_number = -1
        self.current_validation_event_number = -1
        self.current_event_is_validation = 0
        self.min_ev_in_buffer = -1
        self.max_ev_in_buffer = -1
        self.first_validation_event_in_buffer = -1
        self.total_num_events = -1
        self.nevents =-1
        self.last_particles_entry = -1
        self.last_relations_entry = -1
        self.first_entry_dict_particles, self.last_entry_dict_particles, self.first_entry_dict_relations, self.last_entry_dict_relations = self.input_file_splitting()
        self.data_stable_particles_in_buffer = pd.DataFrame()
        self.data_b_particles_in_buffer = pd.DataFrame()
        self.data_relations_in_buffer = pd.DataFrame()
        self.signal_data_relations_in_buffer = pd.DataFrame()
        self.trigger_type = 0 # 0 to accept everything, 1 to require 2-body candidates, 2 to require 2-body candidates and >0 isolation tracks
        self.node_features_filtering = ['pt','associated_ip','eta']
        self.node_features_filtering = [f + self.reco_suffix for f in self.node_features_filtering]
        self.node_features_filtering += ['charge']
        if self.include_neutrals:
            self.node_features_filtering += ['Prob_gamma', 'Prob_pi0', 'isCharged']
        #self.node_features_filtering += ['isCharged']
        #self.node_features_coarsening = ['pt','associated_ip','xProd','yProd','zProd','px','py','pz','associated_xPV','associated_yPV','associated_zPV']
        self.node_features_coarsening = ['pt','associated_ip','eta']
        self.node_features_coarsening = [f + self.reco_suffix for f in self.node_features_coarsening]
        self.node_features_coarsening += ['charge']
        if self.include_neutrals:
            self.node_features_coarsening += ['Prob_gamma', 'Prob_pi0', 'isCharged']
        if self.simulation_type == 0:
            self.node_features_LCA = ['xProd','yProd','zProd','px','py','pz']
            self.node_features_LCA = [f + self.reco_suffix for f in self.node_features_LCA]
            self.node_features_LCA += ['charge', 'Prob_k','Prob_e','Prob_mu','Prob_p','Prob_pi', 'xPV_reco', 'yPV_reco', 'zPV_reco'] # the last three features are the reco PV which are used for PV node creation
        elif self.simulation_type == 1:
            self.node_features_LCA = ['xProd','yProd','zProd','px','py','pz']
            self.node_features_LCA = [f + self.reco_suffix for f in self.node_features_LCA]
            self.node_features_LCA += ['charge', 'ParticleRecoType']
        self.BDT_NP_features = ['pt_reco', 'log_associated_ip_reco', 'eta_reco', 'ZDist_reco', 'RadialDist_reco', 'pz_reco', 'NumTracksInEvent']
        if self.simulation_type == 1:
            self.BDT_NP_features += ['ParticleRecoType']
        self.BDT_EP_features = ['NumTracksInEvent', 'FromSameAssociatedPV_reco', 'theta_reco', 'abs_delta_z0_reco', 'DOCA_reco', 'delta_RadialDist_reco', 'delta_phi_reco', 'delta_eta_reco', 'min_pt_reco']
        if self.simulation_type == 1:
            self.BDT_EP_features += ['p1_ParticleRecoType', 'p2_ParticleRecoType']

        # Initial reset of the environment
        self.reset()


    def reset(self):

        self.final_state_particles = pd.DataFrame()  # list containing all final state particles
        self.final_state_particle_relations = pd.DataFrame()
        self.b_particles = pd.DataFrame()
        self.current_event_is_validation = 0
        self.num_HeavyHadrons_in_event = 0
        self.num_HeavyHadrons_daughters_in_event = 0
        self.true_particle_blocks = {} # dictionary of blocks of stable particles produced in each heavy hadron decay in the event
        self.node_filtering = 0
        self.edge_filtering = 0
        self.nodefiltered_final_state_particles = pd.DataFrame() # list with the stable particles selected by the node filtering algorithm
        self.nodefiltered_final_state_particle_relations = pd.DataFrame()
        self.edgefiltered_nodefiltered_final_state_particle_relations = pd.DataFrame()
        self.use_only_signal_particles = 0
        self.signal_final_state_particles = pd.DataFrame() # list with the signal stable particles
        self.signal_final_state_particle_relations = pd.DataFrame() # list with all the relations between the signal stable particles
        self.reco_particle_blocks = [] # list of reconstructed blocks of stable particles


    def input_file_splitting(self):

        # Obtain the total number of events
        with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
            for input_data in branch.iterate(['ExpandedEventNumber'], library='pd', entry_start=-1):
                self.total_num_events = input_data['ExpandedEventNumber'].iloc[0]+1
                break
        with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
            for input_data in branch.iterate(['ExpandedEventNumber'], library='pd', entry_start=-1):
                end = input_data['ExpandedEventNumber'].iloc[0] + 1
                break
        with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
            for input_data in branch.iterate(['ExpandedEventNumber'], library='pd', entry_start=0):
                start = input_data['ExpandedEventNumber'].iloc[0]
                break
        self.nevents = end - start
        # Find the first and last event number per batch
        num_batches = int(self.total_num_events/self.num_evs_in_buffer)
        if num_batches < 1:
            num_batches = 1
        first_event_in_batch_list = []
        last_event_in_batch_list = []
        for i in range(num_batches):
            first_event_in_batch_list.append(i*self.num_evs_in_buffer)
            last_event_in_batch_list.append((i+1)*self.num_evs_in_buffer-1)
        if self.total_num_events%self.num_evs_in_buffer > 0:
            first_event_in_batch_list.append(last_event_in_batch_list[-1]+1)
            last_event_in_batch_list.append(self.total_num_events-1)

        # Create dictionaries to store the first and last entry in the files for each batch
        first_entry_dict_particles = {}
        last_entry_dict_particles = {}
        first_entry_dict_relations = {}
        last_entry_dict_relations = {}
        for ev_num in first_event_in_batch_list:
            first_entry_dict_particles[ev_num] = -1
            last_entry_dict_particles[ev_num] = -1
            first_entry_dict_relations[ev_num] = -1
            last_entry_dict_relations[ev_num] = -1

        print('Obtaining the splitting scheme for the input files ...')

        with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
            for input_data in branch.iterate(['ExpandedEventNumber'], library='pd', entry_start=-1):
                self.last_particles_entry = input_data.index.values[0]

        with uproot.open(self.sim_file_name.replace('data_particles', 'data_relations'), compression=None)['ParticleRelations'] as branch:
            for input_data in branch.iterate(['ExpandedEventNumber'], library='pd', entry_start=-1):
                self.last_relations_entry = input_data.index.values[0]

        # Explore the particles file
        with tqdm(total=100, desc="Processing particles", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
                for input_data, report in branch.iterate(['ExpandedEventNumber'], library='pd', step_size='10MB', report=True):
                    for i in range(len(first_event_in_batch_list)):

                        # Update the entry of the first event per batch
                        min_ev_num = first_event_in_batch_list[i]
                        data_in_event = input_data[input_data['ExpandedEventNumber'] == min_ev_num]
                        if data_in_event.size > 0:
                            if first_entry_dict_particles[min_ev_num] == -1:
                                first_entry_dict_particles[min_ev_num] = data_in_event.index.values[0]
                            else:
                                first_entry_dict_particles[min_ev_num] = min(data_in_event.index.values[0], first_entry_dict_particles[min_ev_num])

                        # Update the entry of the last event per batch
                        max_ev_num = last_event_in_batch_list[i]
                        data_in_event = input_data[input_data['ExpandedEventNumber'] == max_ev_num]
                        if data_in_event.size > 0:
                            if last_entry_dict_particles[min_ev_num] == -1:
                                last_entry_dict_particles[min_ev_num] = data_in_event.index.values[-1]
                            else:
                                last_entry_dict_particles[min_ev_num] = max(data_in_event.index.values[-1], last_entry_dict_particles[min_ev_num])

                    pbar.update(int(report.stop/self.last_particles_entry*100.)-pbar.last_print_n)

        # Explore the relations file
        with tqdm(total=100, desc="Processing relations", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            with uproot.open(self.sim_file_name.replace('data_particles', 'data_relations'), compression=None)['ParticleRelations'] as branch:
                for input_data, report in branch.iterate(['ExpandedEventNumber'], library='pd', step_size='20KB', report=True):
                    for i in range(len(first_event_in_batch_list)):

                        # Update the entry of the first event per batch
                        min_ev_num = first_event_in_batch_list[i]
                        data_in_event = input_data[input_data['ExpandedEventNumber'] == min_ev_num]
                        if data_in_event.size > 0:
                            if first_entry_dict_relations[min_ev_num] == -1:
                                first_entry_dict_relations[min_ev_num] = data_in_event.index.values[0]
                            else:
                                first_entry_dict_relations[min_ev_num] = min(data_in_event.index.values[0], first_entry_dict_relations[min_ev_num])

                        # Update the entry of the last event per batch
                        max_ev_num = last_event_in_batch_list[i]
                        data_in_event = input_data[input_data['ExpandedEventNumber'] == max_ev_num]
                        if data_in_event.size > 0:
                            if last_entry_dict_relations[min_ev_num] == -1:
                                last_entry_dict_relations[min_ev_num] = data_in_event.index.values[-1]
                            else:
                                last_entry_dict_relations[min_ev_num] = max(data_in_event.index.values[-1], last_entry_dict_relations[min_ev_num])

                    pbar.update(int(report.stop/self.last_relations_entry*100.)-pbar.last_print_n)

        print('Splitting scheme obtained.')

        """last_key, last_value = list(first_entry_dict_particles.items())[-1]
        first_entry_dict_particles = {0: last_value}
        last_key, last_value = list(last_entry_dict_particles.items())[-1]
        last_entry_dict_particles = {0: last_value}
        last_key, last_value = list(first_entry_dict_relations.items())[-1]
        first_entry_dict_relations = {0: last_value}
        last_key, last_value = list(last_entry_dict_relations.items())[-1]
        last_entry_dict_relations = {0: last_value}"""

        return first_entry_dict_particles, last_entry_dict_particles, first_entry_dict_relations, last_entry_dict_relations


    def read_simulation(self, first_ev_number, last_ev_number):
        '''
        Return a pandas df and the number of events in the sample
        '''


        print('Loading new buffer file (event numbers from '+str(first_ev_number)+' to '+str(last_ev_number)+')')
        # Load the pandas df for the particles
        data_list = []
        with tqdm(total=100, desc="Processing particles", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            with uproot.open(self.sim_file_name, compression=None)['Particles'] as branch:
                for input_data, report in branch.iterate(self.input_particles_columns, library='pd', step_size='10MB', report=True): #, entry_start=self.first_entry_dict_particles[first_ev_number], entry_stop=self.last_entry_dict_particles[first_ev_number], report=True):
                    data_list.append(input_data.astype(dtype = self.input_particles_column_types))
                    #pbar.update(int((report.stop-self.first_entry_dict_particles[first_ev_number])/(self.last_entry_dict_particles[first_ev_number]-self.first_entry_dict_particles[first_ev_number])*100.)-pbar.last_print_n)
        data = pd.concat(data_list, ignore_index=True, sort=True)

        del data_list
        print("New particles dataframe buffer, of size: {:.1f} MB".format(data.memory_usage(deep=True).sum() / 1048576.))

        # Load the pandas df for the particle relations
        data_relations_list = []
        with tqdm(total=100, desc="Processing relations", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            with uproot.open(self.sim_file_name.replace('data_particles', 'data_relations'), compression=None)['ParticleRelations'] as branch:
                for input_data, report in branch.iterate(self.input_relations_columns, library='pd', step_size='250KB', report=True): #, entry_start=self.first_entry_dict_relations[first_ev_number], entry_stop=self.last_entry_dict_relations[first_ev_number], report=True):
                    data_relations_list.append(input_data.astype(dtype = self.input_relations_column_types))
                    #pbar.update(int((report.stop-self.first_entry_dict_relations[first_ev_number])/(self.last_entry_dict_relations[first_ev_number]-self.first_entry_dict_relations[first_ev_number])*100.)-pbar.last_print_n)
        data_relations = pd.concat(data_relations_list, ignore_index=True, sort=True)
        del data_relations_list

        print("New relations dataframe buffer, of size: {:.1f} MB".format(data_relations.memory_usage(deep=True).sum() / 1048576.))

        # Addition of extra single-particle-level variables to the relations table, to allow applying fiducial cuts
        print('Adding extra variables to the relations, to apply fiducial cuts ...')
        data = data.drop_duplicates(subset=['ExpandedEventNumber','ExpandedParticleIndex'], keep='first')
        data.set_index(['ExpandedEventNumber','ExpandedParticleIndex'], inplace=True)
        p1 = data.reindex(pd.MultiIndex.from_arrays([data_relations['ExpandedEventNumber'].values, data_relations['FirstParticleIndex'].values]))[['pt_reco', 'associated_ip_reco']]
        p2 = data.reindex(pd.MultiIndex.from_arrays([data_relations['ExpandedEventNumber'].values, data_relations['SecondParticleIndex'].values]))[['pt_reco', 'associated_ip_reco']]
        data_relations.loc[:, 'p1_pt_reco'] = p1['pt_reco'].to_numpy(np.float32)
        data_relations.loc[:, 'p2_pt_reco'] = p2['pt_reco'].to_numpy(np.float32)
        data_relations.loc[:, 'p1_associated_ip_reco'] = p1['associated_ip_reco'].to_numpy(np.float32)
        data_relations.loc[:, 'p2_associated_ip_reco'] = p2['associated_ip_reco'].to_numpy(np.float32)
        data.reset_index(inplace=True)
        del p1, p2
        print('Done.')

        # Filtering of the particles and the relations
        print('Filtering the particles and relations ...')
        if self.include_neutrals:
            data_stable_particles = data[(data['ParticleType'] == 0) & (data['inGeomAcc'] == 1)].reset_index(drop=True)
            signal_data_relations = data_relations[((data_relations['FromSameHeavyHadron'] == 1) & (data_relations['BothInGeomAcc'] == 1))].copy().reset_index(drop=True)
            if self.use_edge_prefiltering:
                data_relations = data_relations[(data_relations['BothInGeomAcc'] == 1) & (((data_relations['FromSameAssociatedPV'+self.reco_suffix] == 1) | (data_relations['theta'+self.reco_suffix] <= 0.26)) | (data_relations['p1_isCharged'] == 0) | (data_relations['p2_isCharged'] == 0))]
            else:
                data_relations = data_relations[(data_relations['BothInGeomAcc'] == 1)]
        else:
            # adjust here to include all particles
            allBfinal = False
            selbool = (data['ParticleType'] == 0) & (data['isCharged'] == 1) & (data['pt_reco'] > 0) & (data['associated_ip_reco'] > 0)
            #selbool = (data['ParticleType'] == 0) & (data['isCharged'] == 1)
            if allBfinal:
                geo_selbool = (data['inGeomAcc'] == 1) | (data["ExpandedPrimaryHeavyHadronIndex"] != -1)
            else:
                geo_selbool = (data['inGeomAcc'] == 1)
            selbool = selbool & geo_selbool
            data_stable_particles = data[selbool].reset_index(drop=True)

            selbool = (data_relations['FromSameHeavyHadron'] == 1) & (data_relations['p1_isCharged'] == 1) & (data_relations['p2_isCharged'] == 1)
            if not allBfinal:
                geo_selbool = data_relations['BothInGeomAcc'] == 1
                selbool = selbool & geo_selbool
            signal_data_relations = data_relations[selbool].copy().reset_index(drop=True)

            if allBfinal:
                B_finalparticles = data_stable_particles["ExpandedParticleIndex"][data_stable_particles["ExpandedPrimaryHeavyHadronIndex"]!=-1]
                stable_in_acc = data_stable_particles["ExpandedParticleIndex"][data_stable_particles['inGeomAcc'] == 1]

            selbool = (data_relations['p1_isCharged'] == 1) & (data_relations['p2_isCharged'] == 1) & (data_relations['p1_pt_reco'] > 0) & (data_relations['p2_pt_reco'] > 0) & (data_relations['p1_associated_ip_reco'] > 0) & (data_relations['p2_associated_ip_reco'] > 0)
            if allBfinal:
                # either first or second in B final particles index or both in geometric acceptance
                sel_1 = (data_relations["FirstParticleIndex"].isin(B_finalparticles)) & (
                    data_relations["SecondParticleIndex"].isin(stable_in_acc))
                sel_2 = (data_relations["SecondParticleIndex"].isin(B_finalparticles)) & (
                    data_relations["FirstParticleIndex"].isin(stable_in_acc))
                geo_selbool = (data_relations['BothInGeomAcc'] == 1)|sel_1|sel_2
            else:
                geo_selbool = data_relations['BothInGeomAcc'] == 1
            selbool = selbool & geo_selbool
            if self.use_edge_prefiltering:
                edge_pre_selbool = (data_relations['FromSameAssociatedPV'+self.reco_suffix] == 1) | (data_relations['theta'+self.reco_suffix] <= 0.26)
                data_relations = data_relations[selbool*edge_pre_selbool]
            else:
                data_relations = data_relations[selbool]
        self.data_b_particles_in_buffer = data[(data['isbhadron']==1)]
        del data
        print('Done.')

        # Addition of derived quantities to the particles
        print('Adding derived quantities to the particles ...')
        data_stable_particles.loc[:, 'eta_reco'] = 0.5 * np.log((np.sqrt(data_stable_particles['pt_reco'].to_numpy(np.float32)*data_stable_particles['pt_reco'].to_numpy(np.float32) + data_stable_particles['pz_reco'].to_numpy(np.float32)*data_stable_particles['pz_reco'].to_numpy(np.float32)) + data_stable_particles['pz_reco'].to_numpy(np.float32)) / (np.sqrt(data_stable_particles['pt_reco'].to_numpy(np.float32)*data_stable_particles['pt_reco'].to_numpy(np.float32) + data_stable_particles['pz_reco'].to_numpy(np.float32)*data_stable_particles['pz_reco'].to_numpy(np.float32)) - data_stable_particles['pz_reco'].to_numpy(np.float32)))
        data_stable_particles.loc[:, 'phi_reco'] = np.arctan2(data_stable_particles['px_reco'].to_numpy(np.float32), data_stable_particles['py_reco'].to_numpy(np.float32))
        data_stable_particles.loc[:, 'ZDist_reco'] = data_stable_particles['zProd_reco'].to_numpy(np.float32) - data_stable_particles['associated_zPV_reco'].to_numpy(np.float32)
        data_stable_particles.loc[:, 'RadialDist_reco'] = np.sqrt( (data_stable_particles['xProd_reco'].to_numpy(np.float32) - data_stable_particles['associated_xPV_reco'].to_numpy(np.float32)) * (data_stable_particles['xProd_reco'].to_numpy(np.float32) - data_stable_particles['associated_xPV_reco'].to_numpy(np.float32)) + (data_stable_particles['yProd_reco'].to_numpy(np.float32) - data_stable_particles['associated_yPV_reco'].to_numpy(np.float32)) * (data_stable_particles['yProd_reco'].to_numpy(np.float32) - data_stable_particles['associated_yPV_reco'].to_numpy(np.float32)) )
        data_stable_particles.loc[:, 'log_associated_ip_reco'] = np.log(data_stable_particles['associated_ip_reco'].to_numpy(np.float32) + 1.e-6)
        print('Done.')

        # Addition of derived quantities to the relations
        print('Adding derived quantities to the relations ...')
        #data_relations['distance_variable'] = data_relations['theta'+self.reco_suffix]/0.1 - data_relations['trdist'+self.reco_suffix]/2.
        data_relations['zero'] = 0.
        data_relations['one'] = 1.
        data_relations['abs_delta_z0_reco'] = np.abs(data_relations['delta_z0_reco'])
        data_relations['NotFromSameHeavyHadron'] = (data_relations['FromSameHeavyHadron'] == 0)
        data_stable_particles.set_index(['ExpandedEventNumber','ExpandedParticleIndex'], inplace=True)
        p1 = data_stable_particles.reindex(pd.MultiIndex.from_arrays([data_relations['ExpandedEventNumber'].values, data_relations['FirstParticleIndex'].values]))[['pt_reco', 'eta_reco', 'phi_reco', 'RadialDist_reco']]
        p2 = data_stable_particles.reindex(pd.MultiIndex.from_arrays([data_relations['ExpandedEventNumber'].values, data_relations['SecondParticleIndex'].values]))[['pt_reco', 'eta_reco', 'phi_reco', 'RadialDist_reco']]
        data_relations.loc[:, 'min_pt_reco'] = np.minimum(p1['pt_reco'].to_numpy(np.float32), p2['pt_reco'].to_numpy(np.float32))
        data_relations.loc[:, 'delta_eta_reco'] = np.abs(p1['eta_reco'].to_numpy(np.float32) - p2['eta_reco'].to_numpy(np.float32))
        data_relations.loc[:, 'delta_phi_reco'] = np.abs(p1['phi_reco'].to_numpy(np.float32) - p2['phi_reco'].to_numpy(np.float32))
        data_relations.loc[:, 'delta_RadialDist_reco'] = np.abs(p1['RadialDist_reco'].to_numpy(np.float32) - p2['RadialDist_reco'].to_numpy(np.float32))
        data_stable_particles.reset_index(inplace=True)
        del p1, p2
        print('Done.')

        #if self.current_event_number == -1:
        #    with open('../PYTHIA_gen_files/' + self.trigger_mapping_file_name) as file:
        #        self.trigger_mapping = yaml.load(file, Loader=yaml.CLoader)

        # Add information on the particle multiplicity
        print('Computing the particle multiplicity per event ...')
        if self.include_neutrals == 0:
            num_tracks_map = data_stable_particles['ExpandedEventNumber'].value_counts()
            data_stable_particles.loc[:, 'NumTracksInEvent'] = data_stable_particles['ExpandedEventNumber'].map(num_tracks_map).to_numpy(np.int32)
            data_relations.loc[:, 'NumTracksInEvent'] = data_relations['ExpandedEventNumber'].map(num_tracks_map).to_numpy(np.int32)
            del num_tracks_map
        print('Done.')

        # Total number of events stored in the file
        #max_num_events = data_stable_particles['ExpandedEventNumber'].max() + 1

        print('Input files read.')
        return data_stable_particles, data_relations, signal_data_relations


    def get_event(self, ev_number = -1, from_validation_dataset = 0):
        '''
        From the simulated sample (pandas df) return one event (potentially made out of several collisions)
        '''
        self.current_evt = ev_number

        # Check for invalid configurations
        if from_validation_dataset:
            if self.validation_dataset_fraction == -1:
                print('ERROR: the size of the validation dataset has not been set. An event could not be retrieved.')
                return
            if ev_number != -1:
                print('ERROR: the choice of an event number is incompatible with the request of a validation event.')
                return

        self.current_event_is_validation = from_validation_dataset

        """if self.trigger_type == 0:
            self.current_event_number = np.random.randint(self.max_num_events)
        elif self.trigger_type == 1:
            self.current_event_number = random.choice(self.trigger_mapping['has_candidate'])
        elif self.trigger_type == 2:
            self.current_event_number = random.choice(self.trigger_mapping['has_isolation_tracks'])"""

        if ev_number != -1:
            # Case of user's choice of event number
            if ev_number >= self.total_num_events:
                import pdb; pdb.set_trace()
                print('ERROR: the chosen event number is larger than the maximum value found in the dataset.')
                return

            if self.validation_dataset_fraction != -1:
                print('ERROR: choice of specific event number disabled when a validation-data fraction has been set.')
                return

            if self.min_ev_in_buffer == -1 or ev_number < self.min_ev_in_buffer or ev_number > self.max_ev_in_buffer:
                del self.data_stable_particles_in_buffer, self.data_relations_in_buffer, self.signal_data_relations_in_buffer
                self.min_ev_in_buffer = self.num_evs_in_buffer * int(ev_number/self.num_evs_in_buffer)
                self.max_ev_in_buffer = min(self.min_ev_in_buffer + self.num_evs_in_buffer -1, self.total_num_events - 1)
                self.data_stable_particles_in_buffer, self.data_relations_in_buffer, self.signal_data_relations_in_buffer = self.read_simulation(self.min_ev_in_buffer, self.max_ev_in_buffer)
                if self.validation_dataset_fraction != -1:
                    self.first_validation_event_in_buffer = self.min_ev_in_buffer + int((1. - self.validation_dataset_fraction) * (self.max_ev_in_buffer - self.min_ev_in_buffer + 1))

            self.current_event_number = ev_number

        else:
            # Case of random choice of event number (that can be from the validation dataset or not)

            if from_validation_dataset == 0:
                # Draw an event from the not-validation sample

                ev_number = self.current_event_number + 1

                if self.validation_dataset_fraction != -1:
                    if self.first_validation_event_in_buffer != -1:
                        if ev_number >= self.first_validation_event_in_buffer:
                            ev_number = self.max_ev_in_buffer + 1

                if ev_number >= self.total_num_events:
                    ev_number = 0

                if self.min_ev_in_buffer == -1 or ev_number < self.min_ev_in_buffer or ev_number > self.max_ev_in_buffer:
                    del self.data_stable_particles_in_buffer, self.data_relations_in_buffer, self.signal_data_relations_in_buffer
                    self.min_ev_in_buffer = self.num_evs_in_buffer * int(ev_number/self.num_evs_in_buffer)
                    self.max_ev_in_buffer = min(self.min_ev_in_buffer + self.num_evs_in_buffer -1, self.total_num_events - 1)
                    self.data_stable_particles_in_buffer, self.data_relations_in_buffer, self.signal_data_relations_in_buffer = self.read_simulation(self.min_ev_in_buffer, self.max_ev_in_buffer)
                    if self.validation_dataset_fraction != -1:
                        self.first_validation_event_in_buffer = self.min_ev_in_buffer + int((1. - self.validation_dataset_fraction) * (self.max_ev_in_buffer - self.min_ev_in_buffer + 1))

                self.current_event_number = ev_number

            else:
                # Draw an event from the validation sample

                if self.first_validation_event_in_buffer == -1:
                    self.first_validation_event_in_buffer = int((1. - self.validation_dataset_fraction) * min(self.num_evs_in_buffer, self.total_num_events))

                if self.current_validation_event_number < self.first_validation_event_in_buffer:
                    self.current_validation_event_number = self.first_validation_event_in_buffer - 1

                ev_number = self.current_validation_event_number + 1

                if ev_number > self.max_ev_in_buffer:
                    ev_number = self.first_validation_event_in_buffer

                self.current_validation_event_number = ev_number

        del self.final_state_particles, self.final_state_particle_relations
        self.final_state_particles = self.data_stable_particles_in_buffer[self.data_stable_particles_in_buffer['ExpandedEventNumber'] == ev_number].copy().reset_index(drop=True)
        self.b_particles = self.data_b_particles_in_buffer[self.data_b_particles_in_buffer['ExpandedEventNumber'] == ev_number] # continue here
        signal_particles = self.final_state_particles[self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] > 0]
        self.num_HeavyHadrons_daughters_in_event = signal_particles.shape[0]
        primary_heavy_hadron_indices = list(set(signal_particles['ExpandedPrimaryHeavyHadronIndex']))
        self.num_HeavyHadrons_in_event = len(primary_heavy_hadron_indices)

        for index in primary_heavy_hadron_indices:
            self.true_particle_blocks[int(index)] = [int(x) for x in signal_particles[signal_particles['ExpandedPrimaryHeavyHadronIndex'] == index]['ExpandedParticleIndex']]

        self.final_state_particle_relations = self.data_relations_in_buffer[(self.data_relations_in_buffer['ExpandedEventNumber'] == ev_number) & (self.data_relations_in_buffer['FirstParticleIndex'].isin(self.final_state_particles.ExpandedParticleIndex.values)) & (self.data_relations_in_buffer['SecondParticleIndex'].isin(self.final_state_particles.ExpandedParticleIndex.values))].copy().reset_index(drop=True)
        particle_key_to_list_index = pd.Series(self.final_state_particles.index,index=self.final_state_particles.ExpandedParticleIndex.values).to_dict()
        self.final_state_particle_relations['list_index_1'] = self.final_state_particle_relations['FirstParticleIndex'].map(particle_key_to_list_index)
        self.final_state_particle_relations['list_index_2'] = self.final_state_particle_relations['SecondParticleIndex'].map(particle_key_to_list_index)
        self.final_state_particle_relations['both_charged'] = (self.final_state_particle_relations['p1_isCharged'] & self.final_state_particle_relations['p2_isCharged'])
        self.final_state_particle_relations['one_charged'] = (((self.final_state_particle_relations['p1_isCharged'] == 1) & (self.final_state_particle_relations['p2_isCharged'] == 0)) | ((self.final_state_particle_relations['p1_isCharged'] == 0) & (self.final_state_particle_relations['p2_isCharged'] == 1)))


    def particles_in_block(self, ancestor_index):
        ''' 
        Return a list of booleans for the final state particles, indicating whether or not each particle cames from the decay of the given ancestor
        '''
        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        node_selection = list(list_of_particles['ExpandedPrimaryHeavyHadronIndex'] == ancestor_index)
        return np.array(node_selection)


    def trigger_candidate_indices(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        node_selection = list(list_of_particles['TriggerKey'] == 1)
        return np.array(node_selection)


    def isolation_track_indices(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        node_selection = list(list_of_particles['TriggerKey'] == 2)
        return np.array(node_selection)


    def set_node_filtering(self, node_booleans):
        self.node_filtering = 1
        node_list = list(np.array(range(len(node_booleans)))[np.array(node_booleans)])
        self.nodefiltered_final_state_particles = self.final_state_particles[self.final_state_particles.index.isin(node_list)].copy().reset_index(drop=True)
        self.nodefiltered_final_state_particle_relations = self.final_state_particle_relations[((self.final_state_particle_relations['list_index_1'].isin(node_list)) & (self.final_state_particle_relations['list_index_2'].isin(node_list)))].copy().reset_index(drop=True)
        self.nodefiltered_final_state_particle_relations.drop(columns = ['list_index_1','list_index_2'])
        particle_key_to_list_index = pd.Series(self.nodefiltered_final_state_particles.index,index=self.nodefiltered_final_state_particles.ExpandedParticleIndex.values).to_dict()
        self.nodefiltered_final_state_particle_relations['list_index_1'] = self.nodefiltered_final_state_particle_relations['FirstParticleIndex'].map(particle_key_to_list_index)
        self.nodefiltered_final_state_particle_relations['list_index_2'] = self.nodefiltered_final_state_particle_relations['SecondParticleIndex'].map(particle_key_to_list_index)


    def set_use_only_signal_particles(self, x):
        self.use_only_signal_particles = x
        if (self.use_only_signal_particles):
            if self.current_event_is_validation == 0:
                ev_number = self.current_event_number
            else:
                ev_number = self.current_validation_event_number
            self.signal_final_state_particles = self.final_state_particles[self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] > 0].copy().reset_index(drop=True)
            self.signal_final_state_particle_relations = self.signal_data_relations_in_buffer[(self.signal_data_relations_in_buffer['ExpandedEventNumber'] == ev_number) & (self.signal_data_relations_in_buffer['FirstParticleIndex'].isin(self.signal_final_state_particles.ExpandedParticleIndex.values)) & (self.signal_data_relations_in_buffer['SecondParticleIndex'].isin(self.signal_final_state_particles.ExpandedParticleIndex.values))].copy().reset_index(drop=True)
            particle_key_to_list_index = pd.Series(self.signal_final_state_particles.index,index=self.signal_final_state_particles.ExpandedParticleIndex.values).to_dict()
            self.signal_final_state_particle_relations['list_index_1'] = self.signal_final_state_particle_relations['FirstParticleIndex'].map(particle_key_to_list_index)
            self.signal_final_state_particle_relations['list_index_2'] = self.signal_final_state_particle_relations['SecondParticleIndex'].map(particle_key_to_list_index)
            self.signal_final_state_particle_relations['both_charged'] = (self.signal_final_state_particle_relations['p1_isCharged'] & self.signal_final_state_particle_relations['p2_isCharged'])
            self.signal_final_state_particle_relations['one_charged'] = (((self.signal_final_state_particle_relations['p1_isCharged'] == 1) & (self.signal_final_state_particle_relations['p2_isCharged'] == 0)) | ((self.signal_final_state_particle_relations['p1_isCharged'] == 0) & (self.signal_final_state_particle_relations['p2_isCharged'] == 1)))


    def set_edge_filtering(self, edge_booleans):
        self.edge_filtering = 1
        edge_list = list(np.array(range(len(edge_booleans)))[np.array(edge_booleans)])
        self.edgefiltered_nodefiltered_final_state_particle_relations = self.nodefiltered_final_state_particle_relations[self.nodefiltered_final_state_particle_relations.index.isin(edge_list)].copy().reset_index(drop=True)


    def set_reco_blocks(self, reco_blocks):

        """if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        for block in reco_blocks:
            self.reco_particle_blocks.append(list(list_of_particles.iloc[block]['ExpandedParticleIndex']))"""

        self.reco_particle_blocks = reco_blocks


    def get_BDT_NP_input_per_event(self):

        return self.final_state_particles[self.BDT_NP_features].to_numpy(np.float32)


    def get_BDT_NP_target_per_event(self):

        return (self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] > 0).to_numpy(np.float32)


    def get_BDT_EP_input_per_event(self):

        if self.node_filtering==0:
            particle_connections =  self.final_state_particle_relations
        else:
            particle_connections =  self.nodefiltered_final_state_particle_relations

        return particle_connections[self.BDT_EP_features].to_numpy(np.float32)


    def get_BDT_EP_target_per_event(self):

        if self.node_filtering==0:
            particle_connections =  self.final_state_particle_relations
        else:
            particle_connections =  self.nodefiltered_final_state_particle_relations

        return particle_connections['FromSameHeavyHadron'].to_numpy(np.float32)


    def get_graph(self, process_id):
        '''
        Get the graph from the list of final_stat_particles
        '''

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
            particle_connections =  self.final_state_particle_relations
        else:
            list_of_particles = self.nodefiltered_final_state_particles
            if self.edge_filtering==0:
                particle_connections =  self.nodefiltered_final_state_particle_relations
            else:
                particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        if process_id == 0: # Node filtering
            nodes = list_of_particles[self.node_features_filtering].to_numpy(np.float32)
            particle_connections = particle_connections[particle_connections['both_charged'] == 1]
            single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['FromSameAssociatedPV_reco', 'theta'+self.reco_suffix, 'trdist'+self.reco_suffix, 'delta_z0'+self.reco_suffix, 'one']).to_numpy(np.float32)
            edges = np.concatenate((np.tile(np.array([1.,0.,0.,0.,0.]), (list_of_particles.shape[0], 1)).astype(np.float32), single_direction_edge_features, single_direction_edge_features), axis=0)
            senders = np.concatenate((list_of_particles.index.to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
            receivers = np.concatenate((list_of_particles.index.to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))
        elif process_id == 1: # Coarse clustering
            nodes = list_of_particles[self.node_features_coarsening].to_numpy(np.float32)
            #single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['DOCA'+self.reco_suffix, 'FromSameAssociatedPV_reco', 'theta'+self.reco_suffix, 'trdist'+self.reco_suffix]).to_numpy(np.float32)
            if self.include_neutrals:
                single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['FromSameAssociatedPV_reco', 'theta'+self.reco_suffix, 'trdist'+self.reco_suffix, 'delta_z0'+self.reco_suffix, 'both_charged', 'one_charged']).to_numpy(np.float32)
            else:
                single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['FromSameAssociatedPV_reco', 'theta'+self.reco_suffix, 'trdist'+self.reco_suffix, 'delta_z0'+self.reco_suffix]).to_numpy(np.float32)
            edges = np.concatenate((single_direction_edge_features, single_direction_edge_features), axis=0)
            senders = np.concatenate((particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
            receivers = np.concatenate((particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))
        elif process_id == 2: # LCA matrix reconstruction
            #list_of_particles = list_of_particles[list_of_particles['isCharged'] == 1]
            #particle_connections = particle_connections[particle_connections['both_charged'] == 1]
            nodes = list_of_particles[self.node_features_LCA].to_numpy(np.float32)
            if self.simulation_type == 0:
                single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['FromSameAssociatedPV_reco', 'theta'+self.reco_suffix, 'trdist'+self.reco_suffix, 'delta_z0'+self.reco_suffix]).to_numpy(np.float32)
            elif self.simulation_type == 1:
                single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['abs_delta_z0_reco', 'delta_RadialDist_reco', 'delta_phi_reco', 'delta_eta_reco']).to_numpy(np.float32)
            edges = np.concatenate((single_direction_edge_features, single_direction_edge_features), axis=0)
            senders = np.concatenate((particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
            receivers = np.concatenate((particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))

        #Put everything in a dictionary
        graph_dict = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([len(self.final_state_particles)]).astype(np.float32)
            }

        return graph_dict


    def get_graph_filtering(self):
        return self.get_graph(0)


    def get_graph_coarsening(self):
        return self.get_graph(1)


    def get_graph_LCA(self):
        return self.get_graph(2)


    def get_truth_graph_filtering(self):
        '''
        Get the truth level graph from the list of final state particles
        '''

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        nodes_sig = (list_of_particles['ExpandedPrimaryHeavyHadronIndex'] > 0).to_numpy(np.float32)
        nodes_bkg = (list_of_particles['ExpandedPrimaryHeavyHadronIndex'] < 0).to_numpy(np.float32)
        nodes = np.vstack((nodes_sig, nodes_bkg)).T

        #Put everything in a dictionary
        graph_dict = {
            "nodes": nodes,
            "edges": np.array([]).astype(np.float32),
            "senders": np.array([]).astype(np.int32),
            "receivers": np.array([]).astype(np.int32),
            "globals": np.array([]).astype(np.float32)
            }

        return graph_dict


    def get_truth_graph_coarsening(self):
        '''
        Get the truth level graph from the list of final state particles
        '''

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
            particle_connections =  self.final_state_particle_relations
        else:
            list_of_particles = self.nodefiltered_final_state_particles
            particle_connections =  self.nodefiltered_final_state_particle_relations

        nodes = np.zeros((list_of_particles.shape[0],1))

        single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['FromSameHeavyHadron', 'NotFromSameHeavyHadron']).to_numpy(np.float32)
        #edges = np.concatenate((np.vstack((np.zeros(list_of_particles.shape[0]), np.ones(list_of_particles.shape[0]))).T.astype(np.float32), single_direction_edge_features, single_direction_edge_features), axis=0)
        #senders = np.concatenate((list_of_particles.index.to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
        #receivers = np.concatenate((list_of_particles.index.to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))
        edges = np.concatenate((single_direction_edge_features, single_direction_edge_features), axis=0)
        senders = np.concatenate((particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
        receivers = np.concatenate((particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))

        #Put everything in a dictionary
        graph_dict = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([]).astype(np.float32)
            }

        return graph_dict


    def get_truth_graph_LCA(self):
        '''
        Get the truth level graph from the list of final state particles
        '''
        if self.use_only_signal_particles:
            list_of_particles = self.signal_final_state_particles
            particle_connections =  self.signal_final_state_particle_relations
        else:
            if self.node_filtering==0:
                list_of_particles = self.final_state_particles
                particle_connections =  self.final_state_particle_relations
            else:
                list_of_particles = self.nodefiltered_final_state_particles
                if self.edge_filtering==0:
                    particle_connections =  self.nodefiltered_final_state_particle_relations
                else:
                    particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        #list_of_particles = list_of_particles[list_of_particles['isCharged'] == 1]
        #particle_connections = particle_connections[particle_connections['both_charged'] == 1]

        nodes = np.zeros((list_of_particles.shape[0],1))
        single_direction_edge_features = pd.DataFrame(particle_connections, columns = ['TopoLCA_class0', 'TopoLCA_class1','TopoLCA_class2','TopoLCA_class3']).to_numpy(np.float32)
        edges = np.concatenate((single_direction_edge_features, single_direction_edge_features), axis=0)
        senders = np.concatenate((particle_connections['list_index_1'].to_numpy(np.int32), particle_connections['list_index_2'].to_numpy(np.int32)))
        receivers = np.concatenate((particle_connections['list_index_2'].to_numpy(np.int32), particle_connections['list_index_1'].to_numpy(np.int32)))

        #Put everything in a dictionary
        graph_dict = {
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([]).astype(np.float32)
            }

        return graph_dict


    def get_particle_keys(self):

        if self.use_only_signal_particles:
            list_of_particles = self.signal_final_state_particles
        else:
            if self.node_filtering==0:
                list_of_particles = self.final_state_particles
            else:
                list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['ExpandedParticleIndex'].values.tolist()


    def get_signal_candidate_particle_keys(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles[list_of_particles['Part_ISO_InSignalCandidate'] == 1]['ExpandedParticleIndex'].values.tolist()


    def get_signal_candidate_ISO_BDT_output(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['cand_B0_ISOLATION_BDT'].values.tolist()[0]


    def argmax(self, lst):
        return lst.index(max(lst))


    def get_particle_predicted_ids(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        prob_pi = list_of_particles['Prob_pi'].values.tolist()
        prob_k = list_of_particles['Prob_k'].values.tolist()
        prob_p = list_of_particles['Prob_p'].values.tolist()
        prob_mu = list_of_particles['Prob_mu'].values.tolist()
        prob_e = list_of_particles['Prob_e'].values.tolist()

        predicted_ids = list(map(lambda p_pi, p_k, p_p, p_mu, p_e: self.argmax([p_pi, p_k, p_p, p_mu, p_e]), prob_pi, prob_k, prob_p, prob_mu, prob_e))

        return predicted_ids


    def get_particle_pxs(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['px_reco'].values.tolist()


    def get_particle_pys(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['py_reco'].values.tolist()


    def get_particle_pzs(self):

        if self.node_filtering==0:
            list_of_particles = self.final_state_particles
        else:
            list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['pz_reco'].values.tolist()


    def get_truth_particle_ids(self):

        if self.use_only_signal_particles:
            list_of_particles = self.signal_final_state_particles
        else:
            if self.node_filtering==0:
                list_of_particles = self.final_state_particles
            else:
                list_of_particles = self.nodefiltered_final_state_particles

        return list_of_particles['id'].values.tolist()


    def get_truth_LCA(self):

        if self.use_only_signal_particles:
            particle_connections =  self.signal_final_state_particle_relations
        else:
            if self.node_filtering==0:
                particle_connections =  self.final_state_particle_relations
            else:
                if self.edge_filtering==0:
                    particle_connections =  self.nodefiltered_final_state_particle_relations
                else:
                    particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        return particle_connections['LCA'].values.tolist()


    def get_truth_TopoLCA(self):

        if self.use_only_signal_particles:
            particle_connections =  self.signal_final_state_particle_relations
        else:
            if self.node_filtering==0:
                particle_connections =  self.final_state_particle_relations
            else:
                if self.edge_filtering==0:
                    particle_connections =  self.nodefiltered_final_state_particle_relations
                else:
                    particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        return particle_connections['TopoLCA'].values.tolist()


    def get_truth_LCA_ids(self):

        if self.use_only_signal_particles:
            particle_connections =  self.signal_final_state_particle_relations
        else:
            if self.node_filtering==0:
                particle_connections =  self.final_state_particle_relations
            else:
                if self.edge_filtering==0:
                    particle_connections =  self.nodefiltered_final_state_particle_relations
                else:
                    particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        return particle_connections['LCA_ids'].values.tolist()


    def get_truth_TopoLCA_ids(self):

        if self.use_only_signal_particles:
            particle_connections =  self.signal_final_state_particle_relations
        else:
            if self.node_filtering==0:
                particle_connections =  self.final_state_particle_relations
            else:
                if self.edge_filtering==0:
                    particle_connections =  self.nodefiltered_final_state_particle_relations
                else:
                    particle_connections =  self.edgefiltered_nodefiltered_final_state_particle_relations

        return particle_connections['TopoLCA_ids'].values.tolist()

    def get_truth_isfromb(self):
        list_of_particles = self.final_state_particles
        return list_of_particles["isfromb"].to_numpy()

    def get_primariyhh_indx(self):
        list_of_particles = self.final_state_particles
        return list_of_particles['ExpandedPrimaryHeavyHadronIndex'].to_numpy()


    def get_fragmentation(self):
        list_of_particles = self.final_state_particles
        return list_of_particles["FragmentationPartnerIndex"].to_numpy()

    def get_fragmentation_new(self):
        frag_flag = self.final_state_particles["isFrag"]
        frag_type = self.final_state_particles["fragType"]
        frag_indx = self.final_state_particles["fragPartnerIndex"]

        combined = [
            int(f"{f}{t}{i}") if f != -1 and t != -1 and i != -1 else -1
            for f, t, i in zip(frag_flag, frag_type, frag_indx)
        ]
        return np.array(combined)

    def get_nfinals(self):
        list_of_particles = self.final_state_particles
        return list_of_particles["nFinals"].to_numpy()

    def get_b_to_final(self):
        sf_ephi = self.signal_final_state_particles["ExpandedPrimaryHeavyHadronIndex"]
        sf_epi = self.signal_final_state_particles["ExpandedParticleIndex"]

        selbool = self.data_b_particles_in_buffer["ExpandedEventNumber"] == self.current_evt
        b_epi = self.data_b_particles_in_buffer["ExpandedParticleIndex"][selbool]
        b_id = self.data_b_particles_in_buffer["id"][selbool]

        f_i = self.final_state_particles["ExpandedParticleIndex"]

        b_id_to_sf_epi = {
            bid: sf_epi[(sf_ephi == b)].tolist()
            for b, bid in zip(b_epi, b_id)
            if b in sf_ephi.values
        }

        b_id_to_f_i = {
            k: f_i[f_i.isin(v)].index.tolist()
            for k, v in b_id_to_sf_epi.items()
        }
        return b_id_to_f_i

    def plot_projected_event(self, v_axis):
        '''
        Plot projection 
        '''
        if v_axis == "x":
            ind_v = 0
        if v_axis == "y":
            ind_v = 1
        h_axis = "z"
        ind_h = 2

        if self.include_neutrals:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
            ax2.set_xlabel('x_ECAL')
            ax2.set_ylabel('y_ECAL')
        else:
            fig, ax1 = plt.subplots()
        ax1.set_xlabel(h_axis+' [mm]')
        ax1.set_ylabel(v_axis+' [mm]')

        end_h_axis = 500. # (mm)
        dz_pp_to_ECAL = 13000. # (mm)

        linestyle = 'solid'

        prefiltered_particles = []
        for block in self.reco_particle_blocks:
            prefiltered_particles += block

        sig_particles = self.final_state_particles[self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] > 0]
        bkg_particles = self.final_state_particles[self.final_state_particles['ExpandedPrimaryHeavyHadronIndex'] < 0]

        # Plot all final state particles
        for particle_list in [bkg_particles, sig_particles]:
            for irow in range(particle_list.shape[0]):
                particle = particle_list.iloc[irow]
                if (particle['zProd'+self.reco_suffix] < end_h_axis):
                    three_momentum = Vector3D(particle['px'+self.reco_suffix],particle['py'+self.reco_suffix],particle['pz'+self.reco_suffix])
                    if (particle['isCharged'] == 1):
                        origin_vertex = Vector3D(particle['xProd'+self.reco_suffix],particle['yProd'+self.reco_suffix],particle['zProd'+self.reco_suffix])
                        u = three_momentum.unit()
                        x_init = origin_vertex[ind_h]
                        y_init = origin_vertex[ind_v]
                        x_end = end_h_axis
                        y_end = y_init + u[ind_v] / u[ind_h] * (x_end - x_init)
                        if (particle['ExpandedPrimaryHeavyHadronIndex'] > 0):
                            if (particle['ExpandedParticleIndex'] in prefiltered_particles):
                                ax1.plot( [x_init, x_end], [y_init, y_end], color='tab:red', linewidth=0.7, linestyle=linestyle)
                            else:
                                ax1.plot( [x_init, x_end], [y_init, y_end], color='tab:pink', linewidth=0.7, linestyle=linestyle)
                        else:
                            if (particle['ExpandedParticleIndex'] in prefiltered_particles):
                                ax1.plot( [x_init, x_end], [y_init, y_end], color='black', linewidth=0.7, linestyle=linestyle)
                            else:
                                ax1.plot( [x_init, x_end], [y_init, y_end], color='lightgray', linewidth=0.7, linestyle=linestyle)
                    else:
                        if self.include_neutrals:
                            x_ECAL = dz_pp_to_ECAL * three_momentum[0] / three_momentum[2];
                            y_ECAL = dz_pp_to_ECAL * three_momentum[1] / three_momentum[2];
                            if (particle['ExpandedPrimaryHeavyHadronIndex'] > 0):
                                if (particle['ExpandedParticleIndex'] in prefiltered_particles):
                                    ax2.plot( [x_ECAL], [y_ECAL], color='tab:red', marker="o", markersize=5)
                                else:
                                    ax2.plot( [x_ECAL], [y_ECAL], color='tab:pink', marker="o", markersize=5)
                            else:
                                if (particle['ExpandedParticleIndex'] in prefiltered_particles):
                                    ax2.plot( [x_ECAL], [y_ECAL], color='black', marker="o", markersize=5)
                                else:
                                    ax2.plot( [x_ECAL], [y_ECAL], color='lightgray', marker="o", markersize=5)


        # Save fig
        plt.tight_layout()
        if self.current_event_is_validation == 0:
            ev_number = self.current_event_number
        else:
            ev_number = self.current_validation_event_number
        plt.savefig("topoLCA_plots/plot_filtering_"+v_axis+h_axis+"_event{}.pdf".format(ev_number))
        plt.close()
