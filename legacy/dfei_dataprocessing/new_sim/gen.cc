#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/EvtGen.h"
#include "TFile.h"
#include "TTree.h"
#include <getopt.h>

#include "helpers.h"
#include "frag_helper.h"
#include "reco_helper.h"

using namespace Pythia8;

int main(int argc, char **argv) {
    // Settings
    std::string decayname;
    int nEvent = 100;
    int iEvent = 0;
    int opt;
    // here flags for some cases, if
    // use_evtgen false + gen_bb true  .-> inclusive
    // use_evtgen false + gen_bb false -> bkg
    bool use_evtgen = true;
    bool gen_bb = true;

    while (true) {
        static struct option long_options[] = {
            {"decay", no_argument, 0, 'D'},
            {"nevents", required_argument, 0, 'n'},
            {"firstEvt", required_argument, 0, 'f'},
            {"inclusive", no_argument, 0, 'i'},
            {"background", no_argument, 0, 'b'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        opt = getopt_long(argc, argv, "D:n:f:", long_options, &option_index);

        if (opt == -1)
            break;

        switch (opt) {
            case 'D':
                decayname = optarg;
                break;
            case 'n':
                nEvent = atoi(optarg);
                break;
            case 'f':
                iEvent = atoi(optarg);
                break;
            case 'i':
                use_evtgen = false;
                decayname = "inclusive";
                break;
            case 'b':
                use_evtgen = false;
                gen_bb = false;
                decayname = "background";
                break;
            default:
                std::cerr << "Usage: " << argv[0]
                        << " --decay <decayname> [--nevents <nEvent>] [--firstEvt <iEvent>]"
                        << std::endl;
                return EXIT_FAILURE;
        }
    }


    if (decayname.empty()) {
        std::cerr << "Need to specify decayname for the mode to be generated" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Decay: " << decayname << std::endl;
    std::cout << "Number of events: " << nEvent << std::endl;
    std::cout << "First event: " << iEvent << std::endl;
    int lastEvent = iEvent + nEvent - 1;


    // Pythia generator
    int randomSeed = getRandomSeed();
    Pythia pythia;
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(randomSeed));
    Event &event = pythia.event;
    std::cout << "Random seed: " << randomSeed << std::endl;
    // Pythia simulation process
    pythia.readString("Beams:eCM = 13000.");
    // Requiring B generation for frag studies
    std::cout << "Generating B" << std::endl;
    if (gen_bb) {
        pythia.readString("HardQCD:gg2bbbar = on");
        pythia.readString("HardQCD:qqbar2bbbar = on");
        pythia.readString("PhaseSpace:pTHatMin = 50.");
    } else {
        pythia.readString("SoftQCD:all = on");
    }


    pythia.readString("111:mayDecay = off"); // Deactivate the decay of the pi0

    // Configure the smearing of the PV position for each collision
    // (approximate values taken from https://indico.cern.ch/event/831165/contributions/3717129/attachments/2022791/3382986/ctd_2020_freiss.pdf)
    pythia.readString("Beams:allowVertexSpread = on");
    pythia.readString("Beams:sigmaVertexX = 0.05");
    pythia.readString("Beams:sigmaVertexY = 0.05");
    pythia.readString("Beams:sigmaVertexZ = 100.");

    double min_eta = 1.9;
    double max_eta = 4.9;

    pythia.init();


    // EvtGen initialization
    EvtGenDecays *evtgen = 0;
    if (use_evtgen) {
        evtgen = new EvtGenDecays(&pythia, "DECAY.DEC", "evt.pdl");
        std::string adjustedDecayName = adjustDecayFileName(decayname);
        evtgen->readDecayFile(adjustedDecayName.c_str());
    }


    // Output file name
    std::string cleanedDecayName = cleanDecayName(decayname);

    // Construct the output file name
    std::string outfile_name;
    std::string outdir = "/afs/cern.ch/work/y/yukaiz/public/dfei_repo/new_sim/PYTHIA_tuples/";
    std::ostringstream iEventStr, lastEventStr;
    std::cout << iEvent << std::endl;
    std::cout << lastEvent << std::endl;
    iEventStr << std::setw(8) << std::setfill('0') << iEvent;
    lastEventStr << std::setw(8) << std::setfill('0') << lastEvent;
    outfile_name = outdir + "GeneratedParticles_" + cleanedDecayName + "_" + iEventStr.str() + "_" +
                   lastEventStr.str() + ".root";

    TFile outfile(outfile_name.c_str(), "recreate");
    TTree outtree("Particles", "Particles");

    struct data_holder track_info;

    outtree.Branch("EventNumber", &track_info.evt_number, "EventNumber/I");
    outtree.Branch("ParticleIndex", &track_info.part_idx, "ParticleIndex/I");
    outtree.Branch("MotherIndex", &track_info.mother_idx, "MotherIndex/I");

    // legacy for drawing event
    outtree.Branch("MotherIndex_1", &track_info.mother_idx_1, "MotherIndex_1/I");
    outtree.Branch("MotherIndex_2", &track_info.mother_idx_2, "MotherIndex_2/I");
    outtree.Branch("MotherId_1", &track_info.mother_id_1, "MotherId_1/I");
    outtree.Branch("MotherId_2", &track_info.mother_id_2, "MotherId_2/I");

    outtree.Branch("PrimaryHeavyHadronIndex", &track_info.primariy_heavyhadron_idx, "PrimaryHeavyHadronIndex/I");
    outtree.Branch("is_sig", &track_info.is_sig, "is_sig/I");
    outtree.Branch("nFinals", &track_info.nFinals, "nFinals/I");

    outtree.Branch("id", &track_info.id, "id/I");
    outtree.Branch("m", &track_info.m, "m/D");
    outtree.Branch("px", &track_info.px, "px/D");
    outtree.Branch("py", &track_info.py, "py/D");
    outtree.Branch("pz", &track_info.pz, "pz/D");
    outtree.Branch("px_reco", &track_info.px_reco, "px_reco/D");
    outtree.Branch("py_reco", &track_info.py_reco, "py_reco/D");
    outtree.Branch("pz_reco", &track_info.pz_reco, "pz_reco/D");
    outtree.Branch("xProd", &track_info.xprod, "xProd/D");
    outtree.Branch("yProd", &track_info.yprod, "yProd/D");
    outtree.Branch("zProd", &track_info.zprod, "zProd/D");
    outtree.Branch("xProd_reco", &track_info.xprod_reco, "xProd_reco/D");
    outtree.Branch("yProd_reco", &track_info.yprod_reco, "yProd_reco/D");
    outtree.Branch("zProd_reco", &track_info.zprod_reco, "zProd_reco/D");
    outtree.Branch("ip", &track_info.ip, "ip/D");
    outtree.Branch("ip_reco", &track_info.ip_reco, "ip_reco/D");

    outtree.Branch("xPV", &track_info.xPV, "xPV/D");
    outtree.Branch("yPV", &track_info.yPV, "yPV/D");
    outtree.Branch("zPV", &track_info.zPV, "zPV/D");
    outtree.Branch("xPV_reco", &track_info.xPV_reco, "xPV_reco/D");
    outtree.Branch("yPV_reco", &track_info.zPV_reco, "yPV_reco/D");
    outtree.Branch("zPV_reco", &track_info.yPV_reco, "zPV_reco/D");

    outtree.Branch("isFinal", &track_info.is_final, "isFinal/I");
    outtree.Branch("isCharged", &track_info.is_charged, "isCharged/I");
    outtree.Branch("isNeutral", &track_info.is_neutral, "isNeutral/I");
    outtree.Branch("inGeomAcc", &track_info.in_geom_acc, "inGeomAcc/I");
    outtree.Branch("isbhadron", &track_info.is_b_hadron, "isbhadron/I");
    outtree.Branch("ischadron", &track_info.is_c_hadron, "ischadron/I");

    outtree.Branch("isFrag", &track_info.is_frag, "isFrag/I");
    outtree.Branch("fragType", &track_info.frag_type, "fragType/I");
    outtree.Branch("fragPartnerIndex", &track_info.frag_partner_idx, "fragPartnerIndex/I");

    outtree.Branch("ECAL_region", &track_info.ecal_region, "ECAL_region/I");
    outtree.Branch("ECAL_x", &track_info.ecal_x, "ECAL_x/D");
    outtree.Branch("ECAL_y", &track_info.ecal_y, "ECAL_y/D");
    outtree.Branch("ECAL_x_reco", &track_info.ecal_x_reco, "ECAL_x_reco/D");
    outtree.Branch("ECAL_y_reco", &track_info.ecal_y_reco, "ECAL_y_reco/D");

    outtree.Branch("charge", &track_info.charge, "charge/I");
    outtree.Branch("tau", &track_info.tau, "tau/D");
    outtree.Branch("gammactau", &track_info.gammactau, "gammactau/D");
    outtree.Branch("NumVisDaugh", &track_info.num_vis_daugh, "NumVisDaugh/I");

    int step = 0.01 * nEvent + 1;
    double ev_weight = 0.;
    while (iEvent <= lastEvent) {
        if (!pythia.next()) {
            continue;
        }

        if (use_evtgen) {
            ev_weight = evtgen->decay();
            if (ev_weight <= 0.) {
                continue;
            }
        }


        // Find heavy hadrons and fragmentation
        std::vector<int> fragmentation_index, fragmentation_flag, fragmentation_partner;
        std::vector<int> iSignal, iBHad, iBMother, iHeavyHad;
        for (int i = 0; i < event.size(); ++i) {
            // Check if signal is generated and within lhcb acceptance
            if (use_evtgen && is_in_acceptance(event, i, min_eta, max_eta)) {
                if (abs(event[i].status()) == 95 || abs(event[i].status()) == 96) iSignal.push_back(i);
            }
            // Find primary heavy hadron
            int idAbs = event[i].idAbs();
            if (gen_bb) {
                if (is_primarybhadron(idAbs, event[event[i].mother1()].status(), event[i].isHadron())) {
                    // vector of primary heavy hadron array
                    iBHad.push_back(i);
                    iHeavyHad.push_back(i);

                    // Getting mother to check for old frag identification
                    iBMother.push_back(event[i].mother1());
                    iBMother.push_back(event[i].mother2());

                    // Obtain vector of particle which can be fragmenation particle
                    get_fragmentation_idx(event, i, fragmentation_index, fragmentation_flag);
                    for (size_t j = 0; j < fragmentation_index.size(); ++j) {
                        fragmentation_partner.push_back(i);
                    }
                }
            } else {
                if (is_primaryheavyhadron(idAbs, event[event[i].mother1()].status(), event[i].isHadron())) {
                    iHeavyHad.push_back(i);
                }
            }
        }

        // Check if there is a heavy hadron
        int nBHad = iBHad.size();
        int nHeavyHad = iHeavyHad.size();

        if (use_evtgen) {
            if (iSignal.size() == 0) {
                continue;
            }
            bool any_daughter_outside = false;
            for (int i = 0; i < event.size(); ++i) {
                // Check first signal as it is sufficient if any daughters are outside
                // Adding zprod cut
                if (event[i].isAncestor(iSignal[0])){
                    if (is_in_acceptance(event, i, min_eta, max_eta) == 0){ // outside lhcb accetance in eta
                        any_daughter_outside = true;
                        continue;
                    }
                    if (event[i].zProd() > 500 || event[i].zProd() < -500){
                        any_daughter_outside = true;
                        continue;
                    }
                }
            }
            if (any_daughter_outside) continue; // do not accept the event if not all signal tracks are within
        } else if (gen_bb) {
            bool any_b_inside = false;
            for (int i: iHeavyHad) {
                if (is_in_acceptance(event, i, min_eta, max_eta)) {
                    any_b_inside = true;
                }
            }
            if (any_b_inside == false) continue;
        }

        std::vector<int> nFinals(iHeavyHad.size(), 0);
        // Extract the PV coordinates
        std::vector<double> pv_pos = {
            pythia.process[0].xProd(), pythia.process[0].yProd(), pythia.process[0].zProd()
        };
        int nTracks = 0;
        for (int i = 0; i < event.size(); ++i) {
            if (event[i].isFinal() && event[i].isCharged() && is_in_acceptance(event, i, min_eta, max_eta)) {
                nTracks += 1;
                // check ancestor of heavy hadron
                for (size_t k = 0; k < iHeavyHad.size(); ++k) {
                    if (event[i].isAncestor(iHeavyHad[k])) {
                        nFinals[k] += 1;
                    }
                }
            }
        }
        if (nTracks < 4) {
            continue;
        }
        std::vector<double> reco_pv_pos = reco_PV(pv_pos, nTracks);


        for (int i = 0; i < event.size(); ++i) {
            int particle_type = 0;
            int is_frag = -1, frag_type = -1, frag_partner = -1;
            bool in_acc = is_in_acceptance(event, i, min_eta, max_eta);
            int heavy_hadron_index = -1;
            int is_sig = -1;
            int nfinals = -1;

            for (size_t iSig = 0; iSig < iSignal.size(); ++iSig) {
                if (event[i].isAncestor(iSignal[iSig])) {
                    is_sig = iSignal[iSig];
                }
            }

            // Saving heavy hadrons
            if (std::find(iHeavyHad.begin(), iHeavyHad.end(), i) != iHeavyHad.end()) {
                for (size_t k = 0; k < iHeavyHad.size(); ++k) {
                    if (i == iHeavyHad[k]) {
                        nfinals = nFinals[k];
                    }
                }
                particle_type = 2;
                get_particle_info_new(event, iEvent, i, particle_type,
                                      heavy_hadron_index, is_sig, nfinals,
                                      pv_pos, reco_pv_pos,
                                      in_acc,
                                      is_frag, frag_type, frag_partner,
                                      track_info);
                outtree.Fill();
                continue;
            }

            // Saving heavy hadrons intermediate and final states
            for (int iB = 0; iB < nHeavyHad; ++iB) {
                if (event[i].isAncestor(iHeavyHad[iB])) {
                    heavy_hadron_index = iHeavyHad[iB];
                    nfinals = nFinals[iB];
                }
            }

            if (heavy_hadron_index != -1) {
                if (!event[i].isFinal()) {
                    particle_type = 1;
                }
                save_frag(event, i, iBMother, iBHad, fragmentation_index, fragmentation_flag, fragmentation_partner,
                          is_frag, frag_type, frag_partner);
                get_particle_info_new(event, iEvent, i, particle_type,
                                      heavy_hadron_index, is_sig, nfinals,
                                      pv_pos, reco_pv_pos,
                                      in_acc,
                                      is_frag, frag_type, frag_partner,
                                      track_info);
                outtree.Fill();
                continue;
            }

            // Remaining tracks within the event
            if (event[i].isFinal() && event[i].isCharged() && in_acc) {
                save_frag(event, i, iBMother, iBHad, fragmentation_index, fragmentation_flag, fragmentation_partner,
                          is_frag, frag_type, frag_partner);
                get_particle_info_new(event, iEvent, i, particle_type,
                                      heavy_hadron_index, is_sig, nfinals,
                                      pv_pos, reco_pv_pos,
                                      in_acc,
                                      is_frag, frag_type, frag_partner,
                                      track_info);
                outtree.Fill();
                continue;
            }
            // if we want to log all  particles withn the event
            if (false) {
                get_particle_info_new(event, iEvent, i, particle_type,
                                      heavy_hadron_index, is_sig, nfinals,
                                      pv_pos, reco_pv_pos,
                                      in_acc,
                                      is_frag, frag_type, frag_partner,
                                      track_info);
                outtree.Fill();
            }
        }

        iEvent += 1;
        if (iEvent % step == 0) {
            showProgressBar(iEvent, nEvent, 50);
            std::cout << "-- processing event " << iEvent << "/" << lastEvent + 1 << " = " << std::round(
                iEvent * 100. / (lastEvent + 1)) << "%" << std::endl;
            outfile.Flush();
        }
    }
    std::cout << "finished simulation" << std::endl;
    outfile.cd();
    outtree.Write();
    outfile.Close();
}