// ./main -d Bs_Jpsiphi -s 0 -e 500

#include "helper.h"
#include "root_helper.h"

#include <filesystem>

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>


int main(int argc, char **argv) {
    Options opts = parseOptions(argc, argv);
    if (!opts.valid) {
        return 1;
    }

    std::cout << std::string(30, '=') << std::endl;
    std::cout << "Decay: " << opts.decay << std::endl;
    std::cout << "Start: " << opts.start << std::endl;
    std::cout << "End: " << opts.end - 1 << std::endl;
    std::cout << "Mode: " << opts.mode << std::endl;

    bool use_true = false; // pass to ots arg
    if (opts.mode == "true") use_true = true;

    std::cout << "Bunch size: " << opts.bunch_size << std::endl;
    std::cout << "indir: " << opts.indir << std::endl;
    std::cout << std::string(30, '=') << std::endl;
    std::cout << "Reading in the data:" << std::endl;

    // File settings
    TChain part_chain("Particles");
    TChain rel_chain("ParticleRelations");
    std::ostringstream start, end;
    start << std::setw(8) << std::setfill('0') << opts.start;
    end << std::setw(8) << std::setfill('0') << (opts.start + opts.bunch_size - 1);

    std::string part_file, rel_file;
    //Event_data_particles_Bs_Jpsiphi_nu7p6_00000000_to_00000999.root
    part_file = opts.indir + "Event_data_particles_" + opts.decay + "_nu7p6_" +
                start.str() + "_to_" + end.str() + ".root";
    rel_file = opts.indir + "Event_data_relations_" + opts.decay + "_nu7p6_" +
               start.str() + "_to_" + end.str() + ".root";
    if (!std::filesystem::exists(part_file)) {
        std::cout << "File " << std::endl << part_file << std::endl << "does not exist." << std::endl;
        return 1;
    }
    part_chain.Add(part_file.c_str());
    rel_chain.Add(rel_file.c_str());

    std::cout << "Data read in" << std::endl;
    std::cout << std::string(30, '=') << std::endl;

    // Create a new file and tree for output
    std::string suffix;
    if (use_true){
        suffix = "_singlecollision_reasso_" + start.str() + "_to_" + end.str();
    }else{
        suffix = "_missassosinglecollision_" + start.str() + "_to_" + end.str();
    }
    std::string part_out_name = opts.indir + "Event_data_particles_" + opts.decay + suffix + ".root";
    //std::string part_out_name = "Event_data_particles_" + opts.decay + suffix + ".root";
    std::string rel_out_name = opts.indir + "Event_data_relations_" + opts.decay + suffix + ".root";
    //std::string rel_out_name = "Event_data_relations_" + opts.decay + suffix + ".root";
    TFile *part_out_file = new TFile(part_out_name.c_str(), "RECREATE");
    TTree *part_out_tree = part_chain.CloneTree(0);
    part_out_tree->SetDirectory(part_out_file);
    TFile *rel_out_file = new TFile(rel_out_name.c_str(), "RECREATE");
    TTree *rel_out_tree = rel_chain.CloneTree(0);
    rel_out_tree->SetDirectory(rel_out_file);


    // Associate branch to data struct
    Part_data part_data;
    Rel_data rel_data;
    get_var_part(part_data, part_chain);
    get_var_rel(rel_data, rel_chain);
    Long64_t nEntries_part = part_chain.GetEntries();
    Long64_t nEntries_rel = rel_chain.GetEntries();

    // looping over the events
    int evt_counter = opts.start;
    int particle_counter = 0;
    int relation_counter = 0;
    int passed = 0;
    while (evt_counter <= opts.end - 1) {
        bool found_next_event = false;

        std::vector<PV_data> pv_data;
        // Going through particle files
        for (Long64_t i = particle_counter; i < nEntries_part; ++i) {
            part_chain.GetEntry(i);

            if (part_data.EventNumber != evt_counter) {
                evt_counter++;
                particle_counter = i;
                found_next_event = true;
                break;
            }

            pv_separator(part_data, pv_data, i, use_true);
        }
        // If we reached the end of entries without finding next event, we're done
        if (!found_next_event) {
            std::cout << "Reached end of entries" << std::endl;
            break;
        }

        /*for (size_t j = 0; j < pv_data.size(); ++j) {
            std::cout << "========================" << std::endl;
            std::cout << pv_data[j].xPV << " " << pv_data[j].xPV << " " << pv_data[j].xPV << " " << std::endl;
            for (size_t k = 0; k < pv_data[j].is_sig.size(); ++k){
                std::cout << pv_data[j].is_sig[k] << " " << pv_data[j].particleindex[k] << std::endl;
            }
        }*/

        // Here we start identifying our signal PV after pv miss asso
        int pv_index = -1;
        if (opts.decay != "inclusive"){
            // for exclusive decays, we got is_sig as a marker
            for (size_t j = 0; j < pv_data.size(); ++j) {
                int nsig = 0;
                for (int is_sig: pv_data[j].is_sig) {
                    if (is_sig != -1) nsig++;
                }
                if (nsig >= 4) {
                    pv_index = j;
                    passed++;
                    break;
                }
            }
            if (pv_index == -1) {
                std::cout << evt_counter<< std::endl;
                continue;
            }
        } else {
            // for inclusive we can only use the fact that sig pv has neg evt number idx
            for (size_t j = 0; j < pv_data.size(); ++j) {
                part_chain.GetEntry(pv_data[j].index[0]);
                if (part_data.evtidx < 0){
                    pv_index = j;
                }
            }
        }


        // Event passed the selection, copying the events which passed
        for (int evt: pv_data[pv_index].index) {
            part_chain.GetEntry(evt);
            part_out_tree->Fill();
        }

        for (Long64_t i = relation_counter; i < nEntries_rel; ++i) {
            rel_chain.GetEntry(i);

            if (rel_data.EventNumber > evt_counter - 1) {
                // need the -1 because earlier we have ++ on evt_counter
                relation_counter = i;
                break;
            } else if (rel_data.EventNumber < evt_counter - 1) continue;

            bool first_found = std::find(pv_data[pv_index].particleindex.begin(),
                                         pv_data[pv_index].particleindex.end(),
                                         rel_data.FirstParticleIndex) != pv_data[pv_index].particleindex.end();
            bool second_found = std::find(pv_data[pv_index].particleindex.begin(),
                                          pv_data[pv_index].particleindex.end(),
                                          rel_data.SecondParticleIndex) != pv_data[pv_index].particleindex.end();

            if (first_found && second_found) {
                rel_out_tree->Fill();
            }
        }
    }

    part_out_file->cd();
    part_out_tree->Write();
    part_out_file->Close();

    rel_out_file->cd();
    rel_out_tree->Write();
    rel_out_file->Close();


    std::cout << passed << std::endl;
    return 0;
}