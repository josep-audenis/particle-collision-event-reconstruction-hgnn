#include "frag_helper.h"
#include <iostream>
#include <random>
#include <math.h>

const std::map<char, char> to_quark = {
    {'1', 'd'},
    {'2', 'u'},
    {'3', 's'},
    {'4', 'c'},
    {'5', 'b'},
    {'6', 't'}
};

std::string get_quark_content(int pid, bool is_diquark) {
    std::string quarks = "";
    std::string pid_str = std::to_string(pid);
    int pid_length = pid_str.size();
    //std::cout << pid << std::endl;
    if (is_diquark) {
        for (int i = 0; i < pid_length - 2; i++) {
            quarks += to_quark.at(pid_str[i]);
        }
    } else if (pid_length == 1) {
        // quark
        for (int i = 0; i < pid_length; i++) {
            quarks += to_quark.at(pid_str[i]);
        }
    } else if (pid_length == 3 || pid_length == 4) {
        // Meson, boson
        for (int i = 0; i < pid_length - 1; i++) {
            quarks += to_quark.at(pid_str[i]);
        }
    }

    //std::cout << quarks << std::endl;
    return quarks;
}


int get_origin(const Pythia8::Event &event, int index) {
    // Function which backpropagates quarks to find the gluon/proton
    if (index < 0 || index >= event.size()) {
        std::cout << "Undefined initial index: " << index << std::endl;
        return -1;
    }

    int mother1 = event[index].mother1();
    int mother2 = event[index].mother2();

    if (mother1 < 0 || mother1 >= event.size() ||
        mother2 < 0 || mother2 >= event.size()) {
        std::cout << "Undefined mother index" << std::endl;
        return -1;
    }

    if (mother1 == mother2) {
        if (event[index].id() == event[mother1].id()) {
            return get_origin(event, mother1);
        } else {
            std::cout << "undefined case not sure how to handle" << std::endl;
            std::cout << "Partents: " << event[mother1].id() << ", " << event[mother2].id() << std::endl;
            return -1;
        }
    } else if (event[index].id() == event[mother1].id()) {
        // q -> q g
        return get_origin(event, mother1);
    } else if (event[index].id() == event[mother2].id()) {
        // q -> q g
        return get_origin(event, mother2);
    } else {
        // g -> qq~
        if (abs(event[mother1].id()) == 21) {
            return mother1;
        } else if (abs(event[mother2].id()) == 21) {
            return mother2;
        }
        // Here pp -> qq~ produced during the intial pp collision
        if (abs(event[mother1].id()) == 2212) {
            return mother1;
        } else if (abs(event[mother2].id()) == 2212) {
            return mother2;
        }
        return -1;
    }
}

void forward_propagation(const Pythia8::Event &event, int index, int pid, std::vector<int> &final) {
    auto daughters = event[index].daughterList();

    // Check if q is part of the daughters, if not save particle for fragmentation
    int flag = 0;
    for (int daughter: daughters) {
        if (event[daughter].id() == pid) {
            flag = 1;
        }
    }

    if (flag == 0) {
        // end of chain returning the index
        final.push_back(index);
        return;
    }

    // Move downwards until the end of the quark chain
    for (int daughter: daughters) {
        if (event[daughter].id() == pid) {
            forward_propagation(event, daughter, pid, final);
        }
    }

    return;
}

void get_frag_nodes(Pythia8::Event &event, std::vector<int> daughters, std::string frag_quarks,
                    std::vector<int> &fragmentation_index, std::vector<int> &fragmentation_flag,
                    int origin_flag) {
    for (char frag_quark: frag_quarks) {
        // std::cout << "search frag " << frag_quark << " quark" << std::endl;
        for (int daughter_idx: daughters) {
            std::string daughter_quark = get_quark_content(abs(event[daughter_idx].id()),
                                                           event[daughter_idx].isDiquark());
            if (daughter_quark.find(frag_quark) != std::string::npos) {
                // TODO: obtain information whether the quark is q or q~, which can furhter thin the
                // possible fragmentation tracks

                fragmentation_index.push_back(daughter_idx);
                fragmentation_flag.push_back(origin_flag);
            }
        }
    }
}


void get_fragmentation_idx(Pythia8::Event &event, int event_i, std::vector<int> &fragmentation_index,
                           std::vector<int> &fragmentation_flag) {
    /// Obtain information of the primary B and the quark content of the mothers
    // B structure
    // std::cout << "B " << event[event_i].id() << std::endl;
    std::string b_hadron_quarks = get_quark_content(abs(event[event_i].id()), false);

    // Store mother index and quark content
    std::vector<int> mother_index;
    std::vector<std::string> mother_quarks;


    int m1_idx = event[event_i].mother1();
    mother_index.push_back(m1_idx);
    std::string m1 = get_quark_content(abs(event[m1_idx].id()), event[m1_idx].isDiquark());
    mother_quarks.push_back(m1);
    int m2_idx = event[event_i].mother2();
    mother_index.push_back(m2_idx);
    std::string m2 = get_quark_content(abs(event[m2_idx].id()), event[m2_idx].isDiquark());
    mother_quarks.push_back(m2);

    // std::cout << "mother 1:  " << event[event[event_i].mother1()].id() << std::endl;
    // std::cout << "mother 2: " << event[event[event_i].mother2()].id() << std::endl;

    // std::cout << "=======" << std::endl;

    /// Start finding the fragmentation tracks
    std::string frag_quarks = b_hadron_quarks;
    if (frag_quarks.find('b') != std::string::npos) {
        // Obtain the fragmentation tracks which forms the signal b hadron
        frag_quarks.erase(frag_quarks.find('b'), 1);
    }

    // Same tree fragmentation particles
    // TODO: Check here if daughtesr of m1 and m2 are the same, so we can only use index 0
    std::vector<int> daughters = event[mother_index[0]].daughterList();
    get_frag_nodes(event, daughters, frag_quarks, fragmentation_index, fragmentation_flag, 1);


    // If the mother particle has the identical quark flavour as the signal B frag track, we need to
    // check the opposite tree and tag all possible fragmentation particle

    // If any mother contain the fragmentation quark we backpropagate to the other tree, but excluding diquarks
    // Since diquarks seems to come from proton u (ud)
    int idx = -1;
    if (m1.find(frag_quarks) != std::string::npos && !event[mother_index[0]].isDiquark()) {
        idx = 0;
    } else if (m2.find(frag_quarks) != std::string::npos && !event[mother_index[1]].isDiquark()) {
        idx = 1;
    }

    // Moving towards the other tree
    if (idx >= 0) {
        int origin = get_origin(event, mother_index[idx]);
        if (origin >= 0) {
            // std::cout << "Checking OS tree fragmnetation, foward propagting " << event[mother_index[
            //    idx]].id() * -1 << std::endl;
            std::vector<int> final;
            forward_propagation(event, origin, event[mother_index[idx]].id() * -1, final);
            int origin_flag = 3;
            for (int i: final) {
                if (abs(event[i].id()) == 21) {
                    origin_flag = 2;
                }
                daughters = event[i].daughterList();
                get_frag_nodes(event, daughters, frag_quarks, fragmentation_index, fragmentation_flag, origin_flag);
                // fragmentation type = 1 from same side tree
                //                    = 2 gluon marker
                //                    = 3 OS tree marker
            }
        }
    }

    return;
}


void save_frag(Pythia8::Event &event, int event_i, std::vector<int> iBMother, std::vector<int> iBHad,
               std::vector<int> fragmentation_index, std::vector<int> fragmentation_flag,
               std::vector<int> fragmentation_partner,
               int &is_frag, int &frag_type, int &frag_partner) {
    // Taking care of the daugthers of the heavy hadron to not be flagged as frag
   for (int i = 0; i < static_cast<int>(iBHad.size()); ++i){
        if (event[event_i].isAncestor(iBHad[i]) || event_i == i) {
            return;
        }
    }
    // Check if it is frag part by new algorithm
    for (size_t ifrag = 0; ifrag < fragmentation_index.size(); ++ifrag) {
        int frag_idx = fragmentation_index[ifrag];
        if (event_i == frag_idx || event[event_i].isAncestor(frag_idx)) {
            is_frag = 1;
            frag_type = fragmentation_flag[ifrag];
            frag_partner = fragmentation_partner[ifrag];
            break;
        }
    }

    for (size_t iB = 0; iB < iBHad.size(); ++iB) {
        bool mother_bool = event[event_i].isAncestor(iBMother[2 * iB]) || event[event_i].isAncestor(
                               iBMother[2 * iB + 1]);
        if (mother_bool && !event[event_i].isAncestor(iBHad[iB])) {
            if (is_frag != 1) {
                // Check if it is frag part only by old algorithm
                is_frag = 2;
                frag_partner = iBHad[iB];
            } else {
                // Check if it is frag part by both algorithm
                is_frag = 3;
            }
            break;
        }
    }
    return;
}