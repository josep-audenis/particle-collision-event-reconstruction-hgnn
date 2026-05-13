#include "helpers.h"
#include <iostream>
#include <random>
#include <math.h>
#include <chrono>

// Function to check and adjust the decay filename
std::string adjustDecayFileName(const std::string &decayname) {
    std::string prefix = "";
    std::string suffix = ".dec";
    std::string adjustedName = decayname;

    // Check if the prefix is already there
    if (adjustedName.find(prefix) != 0) {
        adjustedName = prefix + adjustedName;
    }

    // Check if the suffix is already there
    if (adjustedName.size() < suffix.size() || adjustedName.substr(adjustedName.size() - suffix.size()) != suffix) {
        adjustedName += suffix;
    }

    return adjustedName;
}

// Function to clean the decay name
std::string cleanDecayName(const std::string &decayname) {
    std::string cleanName = decayname;
    std::string prefix = "decfiles/";
    std::string suffix = ".dec";

    // Remove prefix if it exists
    if (cleanName.find(prefix) == 0) {
        cleanName = cleanName.substr(prefix.length());
    }

    // Remove suffix if it exists
    if (cleanName.size() >= suffix.size() && cleanName.substr(cleanName.size() - suffix.size()) == suffix) {
        cleanName = cleanName.substr(0, cleanName.size() - suffix.size());
    }

    return cleanName;
}

// Function to display the progress bar
void showProgressBar(int current, int total, int barWidth) {
    float progress = (float)current / total;
    int pos = barWidth * progress;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// Random seed generation
namespace MyRandom {
    std::mt19937 mersenne{static_cast<std::mt19937::result_type>(std::time(nullptr))};
}

float getRandomNumber() {
    std::normal_distribution<float> normal_distr(0., 1.);
    return normal_distr(MyRandom::mersenne);
}

float getRandomSeed() {
    // Return random number
    std::uniform_int_distribution<> uni_int_distr(1, 1.e+6);
    return uni_int_distr(MyRandom::mersenne);
}

// Identifying if it is a b hadron
int nth_digit(int value, int digit) {
    return (value / (int) pow(10, digit)) % 10;
}


// Function to check if a pid corresponds to that of a b-hadron
bool is_bhadron(int pid) {
    return nth_digit(pid, 2) == 5 || nth_digit(pid, 3) == 5;
}

// Function to check if a pid corresponds to that of a c-hadron
bool is_chadron(int pid) {
    return nth_digit(pid, 2) == 4 || nth_digit(pid, 3) == 4;
}

bool is_primaryheavyhadron(int pid, int mother_status, bool is_hadron) {
    return ((abs(mother_status) >= 71 && abs(mother_status) <= 79) && is_hadron && (
                is_bhadron(pid) || is_chadron(pid)));
}


bool is_primarybhadron(int pid, int mother_status, bool is_hadron) {
    return ((abs(mother_status) >= 71 && abs(mother_status) <= 79) && is_hadron && is_bhadron(pid));
}

// Check if within lhcb acceptance
bool is_in_acceptance(const Pythia8::Event &event, int iparticle, double min_eta, double max_eta) {
    if (event[iparticle].pz() > 0 && event[iparticle].eta() > min_eta && event[iparticle].eta() < max_eta) {
        return 1;
    }
    return 0;
}

//
double calc_ip(double xProd, double yProd, double zProd, double px, double py, double pz, double xPV, double yPV,
               double zPV) {
    double p = sqrt(px * px + py * py + pz * pz);
    double px_norm = px / p;
    double py_norm = py / p;
    double pz_norm = pz / p;
    double A = (xProd - xPV) * (xProd - xPV) + (yProd - yPV) * (yProd - yPV) + (zProd - zPV) * (zProd - zPV);
    double B = (xProd - xPV) * px_norm + (yProd - yPV) * py_norm + (zProd - zPV) * pz_norm;
    double C = px_norm * px_norm + py_norm * py_norm + pz_norm * pz_norm;
    if ((A - B * B / C) < 0.) {
        return 0.;
    }
    return sqrt(A - B * B / C);
}


void get_particle_info(const Pythia8::Event &event, int iEvent, int iparticle,
                       int is_frag, int frag_type,
                       int &EventNumber_var, int &ParticleIndex_var,
                       int &MotherIndex_var_1, int &MotherIndex_var_2, int &MotherId_var_1, int &MotherId_var_2,
                       int &id_var, int &is_frag_var, int &frag_type_var) {
    EventNumber_var = iEvent;
    ParticleIndex_var = iparticle;
    MotherIndex_var_1 = event[iparticle].mother1();
    MotherIndex_var_2 = event[iparticle].mother2();
    MotherId_var_1 = event[event[iparticle].mother1()].id();
    MotherId_var_2 = event[event[iparticle].mother2()].id();

    id_var = event[iparticle].id();

    is_frag_var = is_frag;
    frag_type_var = frag_type;

    return;
}

// Function used to approximate the reconstruction of the first VELO hit of a track
std::vector<double> reco_vProd(double xProd, double yProd, double zProd, double px, double py, double pz) {
    // Parameterisation of the production-vertex smearing
    // (VELO geometry information taken from https://cds.cern.ch/record/2147229/files/10.1016_j.nima.2016.04.077.pdf,
    // smearing parameters taken from https://twiki.cern.ch/twiki/bin/view/LHCb/ConferencePlots)
    int num_VELO_modules = 52;
    double VELO_length = 1000.;
    double VELO_delta_z = VELO_length / num_VELO_modules;
    double VELO_transverse_reso = 0.0085;
    double z_VELO_intersec = ceil(zProd / VELO_delta_z) * VELO_delta_z;
    double x_VELO_intersec = xProd + px / pz * (z_VELO_intersec - zProd);
    double y_VELO_intersec = yProd + py / pz * (z_VELO_intersec - zProd);
    std::vector<double> vProd_reco;
    vProd_reco.push_back(x_VELO_intersec + getRandomNumber() * VELO_transverse_reso);
    vProd_reco.push_back(y_VELO_intersec + getRandomNumber() * VELO_transverse_reso);
    vProd_reco.push_back(z_VELO_intersec);
    return vProd_reco;
}

// Function used to approximate the reconstructed momentum of a track
std::vector<double> reco_momentum(double px, double py, double pz) {
    // Parameterisation of the momentum module smearing
    // (smearing parameters taken from https://twiki.cern.ch/twiki/bin/view/LHCb/ConferencePlots)
    double momentum_relative_reso = 0.004;

    // Parameterisation of the momentum slope smearing
    // (estimations taken from Fig.1 of https://arxiv.org/pdf/2101.12040.pdf, then fitted to a custom function y(x) = A*(x+B)**C, finally converted from MeV to GeV)
    double momentum_slope_par_A = 0.0073626431;
    double momentum_slope_par_B = 9.43912599;
    double momentum_slope_par_C = -1.07724536;
    double p = sqrt(px * px + py * py + pz * pz);
    double sigma_slope = momentum_slope_par_A * pow(p + momentum_slope_par_B, momentum_slope_par_C);
    double tx_prime = px / pz + getRandomNumber() * sigma_slope;
    double ty_prime = py / pz + getRandomNumber() * sigma_slope;
    double p_scaling = 1. + getRandomNumber() * momentum_relative_reso;
    double pz_reco = p * p_scaling / sqrt(1. + tx_prime * tx_prime + ty_prime * ty_prime);
    double px_reco = tx_prime * pz_reco;
    double py_reco = ty_prime * pz_reco;
    std::vector<double> vp_reco;
    vp_reco.push_back(px_reco);
    vp_reco.push_back(py_reco);
    vp_reco.push_back(pz_reco);
    return vp_reco;
}


void get_particle_info_new(const Pythia8::Event &event, int iEvent, int iparticle, int particle_type,
                           int heavy_hadron_index, int is_sig, int nfinals,
                           std::vector<double> pv_pos, std::vector<double> pv_pos_reco,
                           bool in_acc,
                           int is_frag, int frag_type, int frag_partner_idx,
                           struct data_holder &track_info) {
    track_info.evt_number = iEvent;
    track_info.part_idx = iparticle;
    track_info.mother_idx = event[iparticle].mother1();
    track_info.primariy_heavyhadron_idx = heavy_hadron_index;
    track_info.is_sig = is_sig;
    track_info.nFinals = nfinals;
    // legacy for plotting
    track_info.mother_idx_1 = event[iparticle].mother1();
    track_info.mother_idx_2 = event[iparticle].mother2();
    track_info.mother_id_1 = event[event[iparticle].mother1()].id();
    track_info.mother_id_2 = event[event[iparticle].mother2()].id();

    track_info.id = event[iparticle].id();
    track_info.m = event[iparticle].m();
    track_info.px = event[iparticle].px();
    track_info.py = event[iparticle].py();
    track_info.pz = event[iparticle].pz();
    track_info.xprod = event[iparticle].xProd();
    track_info.yprod = event[iparticle].yProd();
    track_info.zprod = event[iparticle].zProd();

    track_info.xPV = pv_pos[0];
    track_info.yPV = pv_pos[1];
    track_info.zPV = pv_pos[2];
    track_info.xPV_reco = pv_pos_reco[0];
    track_info.yPV_reco = pv_pos_reco[1];
    track_info.zPV_reco = pv_pos_reco[2];

    track_info.is_final = event[iparticle].isFinal();
    track_info.is_charged = event[iparticle].isCharged();
    track_info.is_neutral = event[iparticle].isNeutral();
    track_info.in_geom_acc = in_acc;
    track_info.charge = event[iparticle].charge();
    track_info.tau = event[iparticle].tau();

    track_info.is_frag = is_frag;
    track_info.frag_type = frag_type;
    track_info.frag_partner_idx = frag_partner_idx;

    if (particle_type == 0) {
        if (event[iparticle].isCharged() == 1) {
            double ip = calc_ip(track_info.xprod, track_info.yprod, track_info.zprod,
                                 track_info.px, track_info.py, track_info.pz,
                                 track_info.xPV, track_info.yPV, track_info.zPV);
            std::vector<double> reco_vProd_ = reco_vProd(track_info.xprod, track_info.yprod, track_info.zprod,
                                                    track_info.px, track_info.py, track_info.pz);
            std::vector<double> reco_momentum_ = reco_momentum(track_info.px, track_info.py, track_info.pz);
            double ip_reco = calc_ip(reco_vProd_[0], reco_vProd_[1], reco_vProd_[2],
                                 reco_momentum_[0], reco_momentum_[1], reco_momentum_[2],
                                 track_info.xPV_reco, track_info.yPV_reco, track_info.zPV_reco);
            track_info.px_reco = reco_momentum_[0];
            track_info.py_reco = reco_momentum_[1];
            track_info.pz_reco = reco_momentum_[2];
            track_info.xprod_reco = reco_vProd_[0];
            track_info.yprod_reco = reco_vProd_[1];
            track_info.zprod_reco = reco_vProd_[2];
            track_info.ip = ip;
            track_info.ip_reco = ip_reco;
        } else {
        // to be implemented calo response for pi0 and gamma
        }
    }

    if (particle_type == 2) {
        track_info.is_b_hadron = is_bhadron(event[iparticle].idAbs());
        track_info.is_c_hadron = is_chadron(event[iparticle].idAbs());
    } else {
        track_info.is_b_hadron = -1;
        track_info.is_c_hadron = -1;
    }
    if (event[iparticle].m() > 0) {
        track_info.gammactau = event[iparticle].e() / event[iparticle].m() * event[iparticle].tau();
    } else {
        track_info.gammactau  = -1.;
    }
    int num_vis_daugh = -1; // Number of "visible" daughters (not stable neutral particles)
    if (particle_type == 1 || particle_type == 2) {
        num_vis_daugh = 0;
        for (int i = 0; i < ((int)event[iparticle].daughterList().size()); ++i) {
            if ((event[event[iparticle].daughterList()[i]].isCharged() == 1) || ((event[event[iparticle].daughterList()[i]].isCharged() == 0) && (event[event[iparticle].daughterList()[i]].isFinal() == 0))) {
                num_vis_daugh += 1;
            }
        }
    }
    track_info.num_vis_daugh  = num_vis_daugh;

    return;
}

/*
struct data_holder{
 int evt_number, part_idx, mother_idx, id;
 // here we explicitly limit to 1 mother which is fine for nearly everything, except for particle which come from quark
 // but they wouldnt be final anyways.
 int primariy_heavyhadron_idx, frag_partner_idx, is_frag, frag_type;
 int is_final, is_charged, is_neutral, in_geom_acc, is_b_hadron, is_c_hadron, charge;
 double m, px, py, pz, px_reco, py_reco, pz_reco;
 double xprod, yprod, zprod, xprod_reco, yprod_reco, zprod_reco;
 double ip, ip_reco;
 double xPV, yPV, zPV, xPV_reco, yPV_reco, zPV_reco;
 double ecal_x, ecal_y, ecal_x_reco, ecal_y_reco;
 double tau, gammactau;
 int num_vis_daugh;
};
* */