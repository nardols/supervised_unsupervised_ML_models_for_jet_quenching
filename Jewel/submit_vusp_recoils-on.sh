#!/bin/bash

#Jewel2.4


#export LD_LIBRARY_PATH=$HOME/lhapdf/lib
export LD_LIBRARY_PATH=/sampa/archive/leonardo/Jets/lhapdf6/lib:$LD_LIBRARY_PATH


export LHAPATH=/sampa/archive/leonardo/Jets/lhapdf6/share/LHAPDF

source /sampa/llimadas/.bashrc
source /cvmfs/alice.cern.ch/etc/login.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

python /sampa/llimadas/Jewel2.4/analysis/submit_vusp_recoils-on.py --experiment_num $1 --nevent $2 --background $3 --MLtype $4 --average_medium $5


