#!/bin/bash

python3 file_test.py \
  --data_file="30var_ER4_neuralgplvm_1000" \
  --model_type="cnp" \
  --d_model=1024 \
  --dim_feedforward=4096 \
  --nhead=8 \
  --num_layers_encoder=4 \
  --run_name="DFalldata_neuralgplvm_1000_MTcnp_DM1024_DFF4096_NH8_NL4_NC10_BS8_LR0.00030_WR0.02" \
  --sample_size=500 \
  --num_mixture_components=10 \
  --batch_size=8 \
  --num_workers=0 \
  --results_path="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments" \
