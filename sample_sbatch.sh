#!/bin/bash

if [[ "$HOSTNAME" != "biwidl215" ]]; then
echo "Not on biwidl314, abort..."
exit 1
fi

OUTPUT="$(sbatch --parsable --output=/home/segerm/scratch/slurm_logs/%j.out "$@")"
JOB_ID=$OUTPUT

echo "Job submitted with ID $JOB_ID"

FILE="/home/segerm/scratch/slurm_logs/${JOB_ID}.out"
echo "File name ${FILE}"
echo "Waiting for job to start"

while [ ! -f "$FILE" ]
do
sleep 2
done

echo "Job started! Reading output file ${FILE}:"
tail -f "$FILE"
