#!/bin/bash

python /sampa/llimadas/seq_pre-processor/preprocess.py --mode val --scaler on --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode val --scaler on --medium vusp
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode val --scaler off --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode val --scaler off --medium vusp
