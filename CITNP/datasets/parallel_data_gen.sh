#!/bin/bash
# Define the list of GPU nodes
nodes_array=()
for i in $(seq -f "%02g" 1 36); do
  if [ "$i" -ne 19 ]; then
    nodes_array+=("ad6013@gpu${i}.doc.ic.ac.uk") # Add username for sshlogin
  fi
done

# Define parameter ranges (exclusive of end values)
var_start=8
var_end=14  # Last value is 14
data_start=0
data_end=3  # Last value is 3

# Calculate total combinations
var_count=$((var_end - var_start + 1))  # 7 values (8 through 14)
data_count=$((data_end - data_start + 1))  # 4 values (0 through 3)
total_combinations=$((var_count * data_count))  # 28 combinations

# Determine how to distribute jobs across nodes
num_nodes=${#nodes_array[@]}
# Use fewer nodes if we have more nodes than jobs
effective_num_nodes=$((num_nodes > total_combinations ? total_combinations : num_nodes))

echo "Total parameter combinations: ${total_combinations} (${var_count} variable values × ${data_count} data values)"
echo "Distributing across ${effective_num_nodes} nodes"

# --- GNU Parallel Setup ---
JOBLOG_FILE="dataget.log"
PARALLEL_TAG="datagen_$$"
rm -f "${JOBLOG_FILE}"*

echo "Starting jobs with GNU Parallel. Tag: ${PARALLEL_TAG}"
echo "Job log will be written to: ${JOBLOG_FILE}.node*"
echo "---"

# Generate all parameter combinations
combinations=()
for var in $(seq ${var_start} ${var_end}); do
  for data in $(seq ${data_start} ${data_end}); do
    combinations+=("${var},${data}")
  done
done

# Distribute combinations across nodes
node_index=0
for ((i=0; i<${#combinations[@]}; i++)); do
  # Get the combination
  IFS=',' read var_val data_val <<< "${combinations[$i]}"

  # Select node in round-robin fashion
  node_host=${nodes_array[$((i % effective_num_nodes))]}
  node_specific_index=$((i % effective_num_nodes))

  echo "Queueing on ${node_host}: var=${var_val}, data=${data_val}"

  # Command to be executed remotely
  remote_command="cd /vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/CITNP/datasets/ && \
/vol/bitbucket/ad6013/anaconda3/envs/meta_causal_inf/bin/python3 create_save_synth_data.py \
--data_start=${data_val} --data_end=$((data_val + 1)) --var_start=${var_val} --var_end=$((var_val + 1))"

  # Send command to the node
  echo "${remote_command}" | parallel --sshlogin ${node_host} \
    --joblog "${JOBLOG_FILE}.node${node_specific_index}" \
    --tag --tagstring "${PARALLEL_TAG}_job${i}_node${node_specific_index}" "{}" &

  # Brief pause to avoid overwhelming SSH connections
  sleep 0.1
done

echo "---"
echo "All jobs dispatched (${total_combinations} total combinations)"
echo "Monitor progress with: tail -f ${JOBLOG_FILE}.node*"
echo "Or check running jobs with: ps aux | grep '${PARALLEL_TAG}'"
echo "To kill all jobs associated with this batch:"
echo "  pkill -f '${PARALLEL_TAG}'"
echo ""
echo "Waiting for locally launched parallel commands to complete their dispatch..."
wait
echo "All local parallel dispatch commands have finished. Remote jobs may still be running."
echo "Script finished."