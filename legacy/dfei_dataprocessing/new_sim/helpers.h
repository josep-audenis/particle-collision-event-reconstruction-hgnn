#ifndef HELPERS_H
#define HELPERS_H

#include "Pythia8/Pythia.h"
#include <map>
#include <string>

std::string adjustDecayFileName(const std::string &decayname);
std::string cleanDecayName(const std::string &decayname);
void showProgressBar(int current, int total, int barWidth);


float getRandomNumber();
float getRandomSeed();

bool is_primarybhadron(int pid, int mother_status, bool is_hadron);
bool is_primaryheavyhadron(int pid, int mother_status, bool is_hadron);

bool is_in_acceptance(const Pythia8::Event &event, int iparticle, double min_eta, double max_eta);


void get_particle_info(const Pythia8::Event &event, int iEvent, int iparticle,
                       int is_frag, int frag_type,
                       int &EventNumber_var, int &ParticleIndex_var,
                       int &MotherIndex_var_1, int &MotherIndex_var_2, int &MotherId_var_1, int &MotherId_var_2,
                       int &id_var, int &is_frag_var, int &frag_type_var);

void get_particle_info_new(const Pythia8::Event &event, int iEvent, int iparticle, int particle_type,
                           int heavy_hadron_index, int is_sig, int nfinals,
                           std::vector<double> pv_pos, std::vector<double> pv_pos_reco,
                           bool in_acc,
                           int is_frag, int frag_type, int frag_partner_idx,
                           struct data_holder &track_info);

struct data_holder {
    int evt_number, part_idx, mother_idx, id;
    // here we explicitly limit to 1 mother which is fine for nearly everything, except for particle which come from quark
    // but they wouldnt be final anyways.
    int primariy_heavyhadron_idx, frag_partner_idx, is_frag, frag_type;
    int is_sig, nFinals;
    int is_final, is_charged, is_neutral, in_geom_acc, is_b_hadron, is_c_hadron, charge;
    double m, px, py, pz, px_reco, py_reco, pz_reco;
    double xprod, yprod, zprod, xprod_reco, yprod_reco, zprod_reco;
    double ip, ip_reco;
    double xPV, yPV, zPV, xPV_reco, yPV_reco, zPV_reco;
    double ecal_x, ecal_y, ecal_x_reco, ecal_y_reco, ecal_region;
    double tau, gammactau;
    int num_vis_daugh;
    // legacy for drawing events
    int mother_idx_1, mother_idx_2, mother_id_1, mother_id_2;
};
#endif