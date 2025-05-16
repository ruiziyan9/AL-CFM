#!/bin/bash
# Define the list of GPU nodes
nodes_array=()
for i in $(seq -f "%02g" 1 36); do
  if [ $i -ne 24 ]; then
    nodes_array+=("ad6013@gpu${i}.doc.ic.ac.uk") # Add username for sshlogin
  fi
done

# Define the specific runs you want
run_start_values=()
for i in $(seq 36 99); do
  run_start_values+=($i)
done
run_start_values+=(23)

# Define the total number of runs
total_runs=${#run_start_values[@]}
num_nodes=${#nodes_array[@]}
runs_per_node_approx=$(( (total_runs + num_nodes - 1) / num_nodes )) # Ceiling division

# --- GNU Parallel Setup ---
JOBLOG_FILE="deciruns.log"
PARALLEL_TAG="deci_batch_$$"

# Clean up old job log if it exists
rm -f "${JOBLOG_FILE}"*

echo "Starting jobs with GNU Parallel. Tag: ${PARALLEL_TAG}"
echo "Job log will be written to: ${JOBLOG_FILE}.node*"
echo "Distributing approximately ${runs_per_node_approx} runs per node across ${num_nodes} nodes."
echo "---"

# Iterate through all runs and distribute them across nodes
for current_run_index in $(seq 0 $((total_runs-1))); do
  # Determine which node to use (round-robin)
  node_index=$((current_run_index % num_nodes))
  node_host=${nodes_array[$node_index]}

  # Get the current data_start value from the array
  current_run_start_outer=${run_start_values[$current_run_index]}

  # We want data_end to be data_start + 1
  data_end_val=$((current_run_start_outer + 1))

  echo "Queueing on ${node_host}: data_start=${current_run_start_outer} data_end=${data_end_val}"

  # The command to be executed remotely
  remote_command="cd /vol/bitbucket/ad6013/Research/CausalInferenceNeuralProcess/baselines/deci/ && \
/vol/bitbucket/ad6013/anaconda3/envs/causica/bin/python3 run_deci.py --data_start=${current_run_start_outer} --data_end=${data_end_val}"

  # Execute the command
  echo "${remote_command}" | parallel --sshlogin ${node_host} --joblog "${JOBLOG_FILE}.node${node_index}" --tag --tagstring "${PARALLEL_TAG}_node${node_index}" "{}" &

  # Add a small delay to avoid overwhelming SSH connections
  sleep 0.1
done

echo "---"
echo "All job blocks dispatched to GNU Parallel."
echo "Monitor progress with: tail -f ${JOBLOG_FILE}.node*"
echo "Or check running jobs with: ps aux | grep '${PARALLEL_TAG}'"
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