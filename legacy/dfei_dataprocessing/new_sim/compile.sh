#!/usr/bin/env bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_107a/x86_64-el9-gcc13-opt/setup.sh

export LOCAL_BASE=/afs/cern.ch/work/y/yukaiz/public/sim_packages_test

export PYTHIA8=$LOCAL_BASE
export PYTHIA8DATA=$PYTHIA8/share/Pythia8/xmldoc
export HEPMC=$LOCAL_BASE
export EVTGEN=$LOCAL_BASE

export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/tbb/2021.10.0-2a247/x86_64-el9-gcc13-opt/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHIA8/lib:$EVTGEN/lib64:$HEPMC/lib64:$LD_LIBRARY_PATH

g++ -g -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -o gen\
    gen.cc helpers.cc frag_helper.cc reco_helper.cc\
    -I$PYTHIA8/include \
    -I$EVTGEN/include \
    -I$HEPMC/include \
    -L$EVTGEN/lib64 -Wl,-rpath,$EVTGEN/lib64 -lEvtGen -lEvtGenExternal \
    -L$HEPMC/lib64 -Wl,-rpath,$HEPMC/lib64 -lHepMC3 \
    -L$PYTHIA8/lib -Wl,-rpath,$PYTHIA8/lib -lpythia8 \
    -L/lib64 -lssl -lcrypto \
    -L/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt/lib \
    -Wl,-rpath,/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt/lib \
    $(root-config --cflags --glibs)
