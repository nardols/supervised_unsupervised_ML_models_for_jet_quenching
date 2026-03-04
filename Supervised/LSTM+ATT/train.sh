#!/bin/bash

pt_label=$1
medium=$2

echo
echo "======================================================================================"
echo "   RUNNING LSTM+ATT TRAINING FOR pT RANGE: ${pt_label}, MEDIUM: ${medium}"
echo "======================================================================================"
echo

echo ">>> [TRAINING] Starting training WITHOUT scaling"
python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/train.py --scaler off --medium $medium --pt_label $pt_label
echo ">>> [TRAINING COMPLETED] medium=${medium}, scaler=OFF"
echo

echo ">>> [TRAINING] Starting training WITH scaling"
python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/train.py --scaler on --medium $medium --pt_label $pt_label
echo ">>> [TRAINING COMPLETED] medium=${medium}, scaler=ON"
echo