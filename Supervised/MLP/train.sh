#!/bin/bash

export LD_LIBRARY_PATH=/sampa/archive/leonardo/Jets/lhapdf6/lib:$LD_LIBRARY_PATH
export LHAPATH=/sampa/archive/leonardo/Jets/lhapdf6/share/LHAPDF

source /sampa/llimadas/.bashrc
source /cvmfs/alice.cern.ch/etc/login.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

medium=$1
features=$2
pt_label=$3

echo
echo "======================================================================================"
echo "        RUNNING TRAINING OF MLP FOR pT RANGE: ${pt_label}, FEATURES: ${features}       "
echo "======================================================================================"
echo 

echo
echo ">>> [TRAINING] Starting training for medium=${medium}, features=${features}, pt=${pt_label}"
python /sampa/llimadas/ML_models/MLP/TPE/train.py --medium $medium --features $features --pt_label $pt_label
echo
echo "Training completed successfully (medium=${medium}, features=${features}, pt=${pt_label})."
echo