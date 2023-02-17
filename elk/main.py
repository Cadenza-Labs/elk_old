import os
import random

import numpy as np
import pickle

from train import train, save_trained_models
from evaluate import evaluate
from elk.utils_evaluation.parser import get_args
from pathlib import Path
import wandb
import pandas as pd
import torch

if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")
    os.makedirs(args.trained_models_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    logistic_regression_model, ccs_model, all_ccs_per_prompt = train(args)
    save_trained_models(args, logistic_regression_model, ccs_all_prompts, all_ccs_per_prompt)

    evaluate(args, logistic_regression_model, ccs_model, all_ccs_per_prompt)

    
    