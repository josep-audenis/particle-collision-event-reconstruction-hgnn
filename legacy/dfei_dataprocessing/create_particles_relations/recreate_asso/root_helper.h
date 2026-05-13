#pragma once

#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

// Data read in struct
struct Part_data {
    Long64_t EventNumber, ParticleIndex;
    Int_t evtidx;
    Double_t xPV, yPV, zPV, true_xPV, true_yPV, true_zPV;
    int isbhadron, is_sig;
};

struct Rel_data {
    Int_t EventNumber, FirstParticleIndex, SecondParticleIndex;
};

// PV struct
struct PV_data {
    float xPV, yPV, zPV;
    std::vector<int> index, particleindex, is_sig;
};


void get_var_part(Part_data & vars, TChain & chain);

void get_var_rel(Rel_data & vars, TChain & chain);

void pv_separator(Part_data data, std::vector<PV_data> &pv_data, int index, bool use_true);