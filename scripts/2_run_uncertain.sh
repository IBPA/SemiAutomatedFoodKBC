#!/bin/bash

#SBATCH --job-name=al_uncertain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.com
#SBATCH --output=/your/path/to/logs/%j.out
#SBATCH --error=/your/path/to/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=10-00:00:00

for run in {1..100}
do
    for round in {1..10}
    do
        ./run_round.sh uncertain $run $round
    done
done
