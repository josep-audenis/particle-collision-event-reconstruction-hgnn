#pragma once

#include <string>

// Options structure
struct Options {
    std::string decay;
    int start = -1;
    int end = -1;
    bool valid = false;
    std::string indir = "/eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_pythia_particles_relations/";
    int bunch_size = 500;
    std::string mode = "true";
};

// Function declarations
void printUsage(const char* program);
Options parseOptions(int argc, char** argv);