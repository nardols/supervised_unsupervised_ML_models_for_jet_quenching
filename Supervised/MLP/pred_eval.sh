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
echo " RUNNING PREDICTION AND EVALUATION OF MLP FOR pT RANGE: ${pt_label}, FEATURES: ${features} "
echo "======================================================================================"
echo 

# ---------------- #
# prediction
# ---------------- #
echo
echo ">>> [PREDICTION] Running predictions for medium=${medium}, features=${features}, pt=${pt_label}"

python /sampa/llimadas/ML_models/MLP/TPE/predict.py --mode test --medium $medium --features $features --pt_label $pt_label
echo "Prediction completed (medium=${medium}, features=${features}, pt=${pt_label})."

python /sampa/llimadas/ML_models/MLP/TPE/predict.py --mode test --medium $medium --features $features --pt_label $pt_label --inverter on
echo "Prediction completed with inverted medium (medium=${medium}, features=${features}, pt=${pt_label})."
echo

# ---------------- #
# evaluation
# ---------------- #
echo
echo ">>> [EVALUATION] Evaluating predictions for medium=${medium}, features=${features}, pt=${pt_label}"

python /sampa/llimadas/ML_models/MLP/TPE/evaluate.py --mode test --medium $medium --features $features --pt_label $pt_label --threshold 0.5
echo "Evaluation completed (medium=${medium}, features=${features}, pt=${pt_label}, threshold=0.5)."

python /sampa/llimadas/ML_models/MLP/TPE/evaluate.py --mode test --medium $medium --features $features --pt_label $pt_label --threshold 0.5 --inverter on
echo "Evaluation completed with inverted medium (medium=${medium}, features=${features}, pt=${pt_label}, threshold=0.5)."
echo