#!/bin/bash


COMMON="alldata_neuralgplvm_1000,cnp,512,2048,8,6,10,32"

# List of parameters as space-separated strings
param_list=()

LR_list=(0.00001 0.00003 0.00007 0.00010 0.00030 0.00050)
WR_list=(0.00 0.01 0.02 0.04 0.08)

# Add every LR × WR combination to param_list
for LR in "${LR_list[@]}"; do
  for WR in "${WR_list[@]}"; do
    param_list+=("${COMMON},${LR},${WR}")
  done
done

# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r DF MT DM DFF NH NL NC BS LR WR <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v DF=$DF,MT=$MT,DM=$DM,DFF=$DFF,NH=$NH,NL=$NL,NC=$NC,BS=$BS,LR=$LR,WR=$WR -N "DF${DF}_MT${MT}_DM${DM}_DFF${DFF}_NH${NH}_NL${NL}_NC${NC}_BS${BS}_LR${LR}_WR${WR}" /gpfs/home/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments/hpc_files/hpc_run.pbs
done
