#!/bin/bash

python /sampa/llimadas/seq_pre-processor/preprocess.py --mode train --scaler on --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode train --scaler on --medium vusp
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode train --scaler off --medium default
python /sampa/llimadas/seq_pre-processor/preprocess.py --mode train --scaler off --medium vusp
