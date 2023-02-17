#!/bin/bash

for TRAIN in imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa
do
    CUDA_VISIBLE_DEVICES=7 python -m elk.train --project_name CCS --model deberta-v2-xxlarge-mnli --prefix normal  --dataset $TRAIN --num_data 1000 --seed 1
    for EVAL in imdb amazon-polarity ag-news dbpedia-14 copa rte boolq qnli piqa
    do
        CUDA_VISIBLE_DEVICES=7 python -m elk.evaluate --project_name CCS --model deberta-v2-xxlarge-mnli --prefix normal --dataset $TRAIN --dataset_eval $EVAL  --num_data 1000 --seed 1
    done
done