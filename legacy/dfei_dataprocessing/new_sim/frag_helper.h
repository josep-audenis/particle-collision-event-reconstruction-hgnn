#ifndef FRAG_HELPER_H
#define FRAG_HELPER_H

#include "Pythia8/Pythia.h"
#include <map>
#include <string>
#include <vector>


void get_fragmentation_idx(Pythia8::Event &event, int id, std::vector<int> &fragmentation_index,
                           std::vector<int> &fragmentation_flag);

void save_frag(Pythia8::Event &event, int event_i, std::vector<int> iBMother, std::vector<int> iBHad,
               std::vector<int> fragmentation_index, std::vector<int> fragmentation_flag,
               std::vector<int> fragmentation_partner, int &is_frag, int &frag_type, int &frag_partner);
#endif