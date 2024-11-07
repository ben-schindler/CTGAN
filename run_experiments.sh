#!/bin/bash

# Submit jobs with different ENS_MULTIPLIER values
echo "Submitting job with ENS_MULTIPLIER = 1"
sbatch main.job 1

echo "Submitting job with ENS_MULTIPLIER = 2"
sbatch main.job 2

echo "Submitting job with ENS_MULTIPLIER = 3"
sbatch main.job 3

echo "Submitting job with ENS_MULTIPLIER = 5"
sbatch main.job 5

echo "Submitting job with ENS_MULTIPLIER = 10"
sbatch main.job 10

echo "All jobs submitted!"