#!/bin/bash

python3 file_test.py \
  --data_file="sachs" \
  --model_type="cnp" \
  --d_model=256 \
  --dim_feedforward=1024 \
  --nhead=8 \
  --num_layers_encoder=4 \
  --run_name="DFalldata_neuralgplvm_1000_MTcnp_DM256_DFF1024_NH8_NL4_NC10_BS32_LR0.00050_WR0.02" \
  --sample_size=500 \
  --num_mixture_components=10 \
  --batch_size=1 \
  --results_path="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments" \
