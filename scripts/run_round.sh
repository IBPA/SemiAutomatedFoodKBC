#!/bin/bash

#SBATCH --job-name=al_round
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

cd ..

echo "AL strategy: $1"
echo "Run: $2"
echo "Round: $3"

AL=$1
RUN=$2
ROUND=$3

# Compute random seed based on the inputs to make sure no random seed is repeated.
if [ $1 == 'uncertain' ]
then
    RANDOM_SEED=1
elif [ $1 == 'stratified' ]
then
    RANDOM_SEED=3
fi

RANDOM_SEED=$((RANDOM_SEED * 10000 + RUN))

echo $RANDOM_SEED

PATH_OUTPUT=/your/path/to/outputs/$AL/run_${RUN}/round_${ROUND}
PATH_TRAIN_POOL=outputs/data_generation/train_pool_small.tsv
PATH_VAL=outputs/data_generation/val_small.tsv
PATH_TEST=outputs/data_generation/test_small.tsv

cd src/data_generation
python prepare_training_data.py \
    --sampling_strategy=$AL \
    --run=$RUN \
    --round=$ROUND \
    --total_rounds=10 \
    --random_state=$RANDOM_SEED \
    --train_pool_filepath=../../$PATH_TRAIN_POOL
cd ../..

python -m src.entailment.run_grid_search \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/train.tsv \
    $PATH_VAL \
    biobert \
    $PATH_OUTPUT/grid_search \
    --batch-sizes 16,32 \
    --learning-rates 2e-5,5e-5 \
    --nums-epochs 3,4 \
    --seeds $RANDOM_SEED \

python -m src.entailment.run_best_model_evaluation \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/train.tsv \
    $PATH_VAL \
    $PATH_TEST \
    biobert \
    $PATH_OUTPUT/grid_search/grid_search_result_summary.csv \
    $PATH_OUTPUT/eval_best_model \
    --seeds $RANDOM_SEED \

python -m src.entailment.run_unlabeled_data_prediction \
    outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/to_predict.tsv \
    biobert \
    $PATH_OUTPUT/eval_best_model \
    --path-output-data-to-predict outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/predicted.tsv \

python -m src.entailment.run_unlabeled_data_prediction \
    $PATH_TEST \
    biobert \
    $PATH_OUTPUT/eval_best_model \
    --path-output-data-to-predict outputs/data_generation/$AL/run_${RUN}/round_${ROUND}/test_probs.tsv \

cd scripts
