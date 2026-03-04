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
echo "---------------------------------------------##-------------------------------------------------"
echo "  RUNNING RANDOM FOREST PREDICTION + EVALUATION FOR pT RANGE: ${pt_label}, MEDIUM: ${medium}    "
echo "---------------------------------------------##-------------------------------------------------"
echo 

# prediction
echo ">>> [PREDICTION] Starting..."
python /sampa/llimadas/ML_models/random_forest/TPE/predict.py --mode test --medium $medium --features $features --pt_label $pt_label
echo "[OK] Prediction completed (medium=$medium, features=$features, pt=$pt_label)"

python /sampa/llimadas/ML_models/random_forest/TPE/predict.py --mode test --medium $medium --features $features --pt_label $pt_label --inverter on
echo "[OK] Inverted prediction completed (medium=$medium, features=$features, pt=$pt_label)"
echo

# evaluation
echo ">>> [EVALUATION] Starting..."
python /sampa/llimadas/ML_models/random_forest/TPE/evaluate.py --mode test --medium $medium --features $features --pt_label $pt_label
echo "[OK] Evaluation completed (medium=$medium, features=$features, pt=$pt_label)"

python /sampa/llimadas/ML_models/random_forest/TPE/evaluate.py --mode test --medium $medium --features $features --pt_label $pt_label --inverter on
echo "[OK] Inverted evaluation completed (medium=$medium, features=$features, pt=$pt_label)"
echo
