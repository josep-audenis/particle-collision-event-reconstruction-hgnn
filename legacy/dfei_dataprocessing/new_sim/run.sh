#!/bin/bash
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/tbb/2021.10.0-2a247/x86_64-el9-gcc13-opt/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/gcc/13.1.0/x86_64-el9/lib64:$LD_LIBRARY_PATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_107a/x86_64-el9-gcc13-opt/setup.sh
export PYTHIA8DATA=''

if [[ -f decfiles.tar.gz ]]; then
    tar -xzf decfiles.tar.gz
fi

echo "Running: ./gen $@"
./gen "$@"
