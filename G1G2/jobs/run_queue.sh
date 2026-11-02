#!/bin/bash
# Run every command of a queue file, NPAR at a time, on the single GPU of the current job.
# train.py skips finished runs, so after a timeout simply resubmit the same queue.
# usage: bash jobs/run_queue.sh jobs/queues/tune.txt 2
set -u
QUEUE=$1
NPAR=${2:-2}
mkdir -p logs
grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$QUEUE" | xargs -d '\n' -P "$NPAR" -I CMD bash -c CMD
STAGE=$(basename "$QUEUE" .txt)
echo "queue $QUEUE finished: $(ls results/"$STAGE"/*.json 2>/dev/null | wc -l) of $(grep -c -- '--stage' "$QUEUE") runs have results"
