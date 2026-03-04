#!/bin/bash

python /sampa/llimadas/seq_pre-processor/preprocess.py --mode test --scaler on --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode test --scaler on --medium vusp
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode test --scaler off --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode test --scaler off --medium vusp
