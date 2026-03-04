#!/bin/bash
set -e

python preprocess.py --mode test --type features --medium default --smote off --scaler off
python preprocess.py --mode test --type features --medium vusp --smote off --scaler off
python preprocess.py --mode test --type PCA --medium default --smote off --scaler off
python preprocess.py --mode test --type PCA --medium vusp --smote off --scaler off