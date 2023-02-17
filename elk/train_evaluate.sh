#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python -m elk.train --project_name CCS --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb --num_data 1000 --seed 1
CUDA_VISIBLE_DEVICES=7 python -m elk.evaluate --project_name CCS --model deberta-v2-xxlarge-mnli --prefix normal  --dataset imdb --dataset_eval boolq  --num_data 1000 --seed 1
