#!/bin/bash
set -e

python preprocess.py --mode train --type features --medium default --smote off --scaler off
python preprocess.py --mode train --type features --medium vusp --smote off --scaler off
python preprocess.py --mode train --type PCA --medium default --smote off --scaler off
python preprocess.py --mode train --type PCA --medium vusp --smote off --scaler off