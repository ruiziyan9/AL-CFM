#!/bin/bash

# List of parameters as space-separated strings
param_list=(
    "alldata_neuralgplvm_1000,cnp,512,1024,8,20,10,32,0.0005"
    "alldata_neuralgplvm_1000,cnp,512,1024,8,18,10,32,0.0005"
)

# Iterate over each parameter set
for i in "${param_list[@]}"
do
    # Split the string into individual parameters
    IFS="," read -r DF MT DM DFF NH NL NC BS LR <<< "$i"

    # Submit the job with the appropriate variables
    qsub -q hx -v DF=$DF,MT=$MT,DM=$DM,DFF=$DFF,NH=$NH,NL=$NL,NC=$NC,BS=$BS,LR=$LR -N "DF${DF}_MT${MT}_DM${DM}_DFF${DFF}_NH${NH}_NL${NL}_NC${NC}_BS${BS}_LR${LR}" /gpfs/home/ad6013/Research/CausalInferenceNeuralProcess/CITNP/experiments/hpc_files/hpc_run.pbs
done
