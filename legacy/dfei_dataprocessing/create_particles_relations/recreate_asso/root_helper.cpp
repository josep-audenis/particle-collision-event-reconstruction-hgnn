#include "root_helper.h"
#include <iostream>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

void get_var_part(Part_data &vars, TChain &chain) {
    chain.SetBranchAddress("ExpandedEventNumber", &vars.EventNumber);
    chain.SetBranchAddress("ExpandedParticleIndex", &vars.ParticleIndex);
    chain.SetBranchAddress("EventNumber", &vars.evtidx);

    chain.SetBranchAddress("associated_xPV", &vars.xPV);
    chain.SetBranchAddress("associated_yPV", &vars.yPV);
    chain.SetBranchAddress("associated_zPV", &vars.zPV);
    chain.SetBranchAddress("xPV", &vars.true_xPV);
    chain.SetBranchAddress("yPV", &vars.true_yPV);
    chain.SetBranchAddress("zPV", &vars.true_zPV);

    chain.SetBranchAddress("isbhadron", &vars.isbhadron);
    chain.SetBranchAddress("is_sig", &vars.is_sig);
}


void get_var_rel(Rel_data &vars, TChain &chain) {
    chain.SetBranchAddress("ExpandedEventNumber", &vars.EventNumber);
    chain.SetBranchAddress("FirstParticleIndex", &vars.FirstParticleIndex);
    chain.SetBranchAddress("SecondParticleIndex", &vars.SecondParticleIndex);
}

void pv_separator(Part_data data, std::vector<PV_data> &pv_data, int index, bool use_true) {
    float xPV, yPV, zPV;
    if (data.isbhadron == 1) {
        xPV = data.true_xPV;
        yPV = data.true_yPV;
        zPV = data.true_zPV;
    } else {
        if (use_true){
            xPV = data.true_xPV;
            yPV = data.true_yPV;
            zPV = data.true_zPV;
        }else{
            xPV = data.xPV;
            yPV = data.yPV;
            zPV = data.zPV;
        }
    }

    bool found = false;
    for (auto &pv: pv_data) {
        // small tolerance of floats
        if (std::abs(pv.xPV - xPV) < 1e-6 &&
            std::abs(pv.yPV - yPV) < 1e-6 &&
            std::abs(pv.zPV - zPV) < 1e-6) {
            pv.index.push_back(index);
            pv.particleindex.push_back(data.ParticleIndex);
            pv.is_sig.push_back(data.is_sig);
            found = true;
            break;
        }
    }
    if (!found) {
        PV_data new_pv;
        new_pv.xPV = xPV;
        new_pv.yPV = yPV;
        new_pv.zPV = zPV;
        new_pv.index.push_back(index);
        new_pv.particleindex.push_back(data.ParticleIndex);
        new_pv.is_sig.push_back(data.is_sig);
        pv_data.push_back(new_pv);
    }
}