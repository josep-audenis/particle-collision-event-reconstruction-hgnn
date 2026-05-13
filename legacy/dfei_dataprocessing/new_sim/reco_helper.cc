#include "reco_helper.h"
#include <random>
#include <math.h>

#include "helpers.h"

std::vector<double> reco_PV(std::vector<double> pv_pos, int nTracks) {
    // Parameterisation of the PV smearing
    // Smearing parameters taken from https://twiki.cern.ch/twiki/bin/view/LHCb/ConferencePlots (2015 data, assumed to be the same for the Upgrade)
    double PV_reso_A_x = 0.1003;
    double PV_reso_B_x = 0.63;
    double PV_reso_C_x = -0.0003;
    double PV_reso_A_z = 1.222;
    double PV_reso_B_z = 0.92;
    double PV_reso_C_z = 0.0138;

    double sigma_x = PV_reso_A_x / pow(nTracks, PV_reso_B_x) + PV_reso_C_x;
    double sigma_z = PV_reso_A_z / pow(nTracks, PV_reso_B_z) + PV_reso_C_z;
    std::vector<double> PV_reco;
    PV_reco.push_back(pv_pos[0] + getRandomNumber() * sigma_x);
    PV_reco.push_back(pv_pos[2] + getRandomNumber() * sigma_x);
    PV_reco.push_back(pv_pos[1] + getRandomNumber() * sigma_z);
    return PV_reco;
}