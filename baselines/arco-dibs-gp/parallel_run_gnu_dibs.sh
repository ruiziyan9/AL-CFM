#!/bin/bash

# Define the list of GPU nodes
nodes_array=()
for i in $(seq -f "%02g" 1 20); do
  nodes_array+=("ad6013@willow${i}.doc.ic.ac.uk") # Add username for sshlogin
done

# Convert array to comma-separated string for parallel's --sshlogin
nodes_csv=$(IFS=,; echo "${nodes_array[*]}")

# Define the total number of runs and how many to do per node
total_runs=100
num_nodes=${#nodes_array[@]}
runs_per_node_approx=$(( (total_runs + num_nodes - 1) / num_nodes )) # Ceiling division

# --- GNU Parallel Setup ---
# Job log file to track progress
JOBLOG_FILE="dibs_gp_runs.log"
# Tag for easy identification and killing of these specific jobs
PARALLEL_TAG="dibsgp_batch_$$" # $$ adds the current script's PID for uniqueness

# Clean up old job log if it exists
rm -f "${JOBLOG_FILE}"* # Also remove node-specific logs if they exist

echo "Starting jobs with GNU Parallel. Tag: ${PARALLEL_TAG}"
echo "Job log will be written to: ${JOBLOG_FILE}.node*"
echo "Distributing approximately ${runs_per_node_approx} runs per node across ${num_nodes} nodes."
echo "---"

node_index=0
current_run_start_outer=0

while [ $current_run_start_outer -lt $total_runs ] && [ $node_index -lt $num_nodes ]; do
  node_host=${nodes_array[$node_index]}
  current_run_end_outer=$((current_run_start_outer + runs_per_node_approx -1))
  if [ $current_run_end_outer -ge $total_runs ]; then
    current_run_end_outer=$((total_runs - 1))
  fi
  data_end_val=$((current_run_end_outer + 1))

  echo "Queueing on ${node_host}: data_start=${current_run_start_outer} data_end=${data_end_val}"

  # The command to be executed remotely.
  # Note: Quoting is important here.
  remote_command="cd /vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/arco-dibs-gp/ && \
/vol/bitbucket/ad6013/anaconda3/envs/metacausinf_baselines/bin/python3 run_dibggp.py --data_start=${current_run_start_outer} --data_end=${data_end_val}"

  # Forcing a single job to a single server from the list for each block.
  # This is a bit less "GNU Parallel idiomatic" for job distribution but matches your original logic.
  # CORRECTED LINE BELOW:
  echo "${remote_command}" | parallel --sshlogin ${node_host} --joblog "${JOBLOG_FILE}.node${node_index}" --tag --tagstring "${PARALLEL_TAG}_node${node_index}" "{}" &

  current_run_start_outer=$((current_run_end_outer + 1))
  node_index=$((node_index + 1))
done


echo "---"
echo "All job blocks dispatched to GNU Parallel."
echo "Monitor progress with: tail -f ${JOBLOG_FILE}.node*"
echo "Or check running jobs with: ps aux | grep '${PARALLEL_TAG}'" # General grep
echo "For more detailed parallel status, you might need to check individual job logs or remote nodes."
echo "To kill all jobs associated with this batch, you can try:"
echo "  pkill -f '${PARALLEL_TAG}'  # This will kill the local parallel processes and their SSH sessions"
echo "  echo \"This may leave remote python processes orphaned if ssh is killed abruptly. A more robust kill requires remote process ID tracking.\""
echo "A more robust way to kill GNU Parallel jobs (if they were launched as one large parallel task):"
echo "  parallel --kill --id '${PARALLEL_TAG}' # Corrected --kill usage if you had a single parallel instance"
echo "(The pkill command is likely more appropriate for the current multi-dispatch structure)"
echo ""
echo "Consider using a PID file on remote hosts for each python script if granular remote kill is essential."
echo "Waiting for locally launched parallel commands to complete their dispatch..."
wait
echo "All local parallel dispatch commands have finished. Remote jobs may still be running."
echo "Script finished."