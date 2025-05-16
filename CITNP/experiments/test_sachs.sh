#!/bin/bash

python3 file_test.py \
  --data_file="sachs" \
  --model_type="cnp" \
  --d_model=512 \
  --dim_feedforward=2048 \
  --nhead=8 \
  --num_layers_encoder=8 \
  --run_name="resnet_8to14var" \
  --sample_size=500 \
  --num_mixture_components=10 \
  --batch_size=1 \
  --num_workers=0 \
  --results_path="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments" \
