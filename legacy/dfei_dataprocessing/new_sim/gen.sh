#!/bin/bash

export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/tbb/2021.10.0-2a247/x86_64-el9-gcc13-opt/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/gcc/13.1.0/x86_64-el9/lib64:$LD_LIBRARY_PATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_107a/x86_64-el9-gcc13-opt/setup.sh
export PYTHIA8DATA=''

mkdir -p PYTHIA_tuples

#./gen -D decfiles/Bs_Jpsiphi.dec -f 0 -n 5
# ./gen -D decfiles/Bd_Dpi.dec -f 0 -n 5
# ./gen -f 40 -n 5 --inclusive
# ./gen -f 40000 -n 1 --background
# ./gen -f 0 -n 100 --background
#./gen -f 800000 -n 400000 --background
#./gen -f 1200000 -n 400000 --background
#./gen -f 1600000 -n 400000 --background
#./gen -f 2000000 -n 400000 --background
#./gen -f 2400000 -n 400000 --background
#./gen -f 2800000 -n 400000 --background
./gen -f 4800000 -n 400000 --background
#./gen -D decfiles/Bd_JpsiKs.dec -f 0 -n 40000
#./gen -D decfiles/Bd_JpsiKs.dec -f 40000 -n 40000
#./gen -D decfiles/Bd_Dpi.dec -f 40000 -n 40000
#./pp2bb13TeVLHCbUpgrade -D decfiles/Bs_Jpsiphi.dec -f 0 -n 50000
