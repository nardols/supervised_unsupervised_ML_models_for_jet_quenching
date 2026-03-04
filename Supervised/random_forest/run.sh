#!/bin/bash

cd /sampa/llimadas/ML_models/random_forest
chmod +x *

medium=$1
features=$2
pt_label=$3

#./train.sh $medium $features $pt_label
./pred_eval.sh $medium $features $pt_label

