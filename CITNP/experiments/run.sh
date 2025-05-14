#!/bin/bash

python3 iterable_train.py \
    --work_dir="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP" \
    --function_generator="resnetgplvm" \
    --model_type="cnp" \
    --d_model=512 \
    --dim_feedforward=2048 \
    --nhead=8 \
    --num_layers_encoder=8 \
    --num_mixture_components=10 \
    --batch_size=8 \
    --learning_rate=0.0005 \
    --run_name="resnetgplvm_8to14var" \
    --sample_size=1200 \
    --epochs=4 \
    --mean_loss_across_samples \
    --lr_warmup_ratio=0.01 \
    --num_workers=2 \
    --iterations_per_epoch=125000 \
    --log_step=1000 \
    --results_path="/vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments" \
