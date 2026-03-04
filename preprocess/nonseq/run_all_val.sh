#!/bin/bash
set -e

python preprocess.py --mode val --type features --medium default --smote off --scaler off
python preprocess.py --mode val --type features --medium vusp --smote off --scaler off
python preprocess.py --mode val --type PCA --medium default --smote off --scaler off
python preprocess.py --mode val --type PCA --medium vusp --smote off --scaler off