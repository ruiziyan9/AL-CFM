#!/bin/bash

# --- Configuration ---
USER="ad6013"
HOST_PREFIX="willow"
DOMAIN="doc.ic.ac.uk"
START_NUM=1
END_NUM=20
# --- End Configuration ---

echo "Starting SSH connections to add host keys for ${HOST_PREFIX}${START_NUM}-${END_NUM}.${DOMAIN}..."

for i in $(seq -f "%02g" ${START_NUM} ${END_NUM}); do
  # The -o ConnectTimeout=5 makes SSH give up if a host is unreachable after 5 seconds
  # The "exit" command ensures the SSH session closes immediately after connecting.
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${USER}@${HOST_PREFIX}${i}.${DOMAIN} "exit"
  # SSH will typically be silent if the connection and key addition are successful.
  # It will print an error message if it cannot connect (e.g., host down, network issue).
done

echo "Finished attempts to add host keys."