#!/bin/bash

DATA_NAME="20ng"

python SpareNTM.py \
  --adam_beta1 0.9\
  --adam_beta2 0.999\
  --learning_rate 0.0001\
  --dir_prior 0.02\
  --bern_prior 0.05\
  --bs 200\
  --n_topic 50\
  --warm_up_period 100\
  --data_dir ./data/${DATA_NAME}/ \
  --data_name ${DATA_NAME}