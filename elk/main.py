import os
import random

import numpy as np
import pickle

from train import train
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

    logistic_regression_model, ccs_model = train(args)

    # save models
    # TODO: use better filename for the pkls, so they don't get overwritten
    with open(args.trained_models_path / "logistic_regression_model.pkl", "wb") as file:
        pickle.dump(logistic_regression_model, file)
    torch.save(ccs_model.state_dict(), args.trained_models_path / "ccs_model.pth")

    evaluate(args, logistic_regression_model, ccs_model)

    evaluation_df = pd.read_csv(args.save_dir / f"{args.model}_{args.prefix}_{args.seed}.csv")
    evaluation_table = wandb.Table(dataframe=evaluation_df)
    # Add the table to an Artifact to make it easier to reuse!
    evaluation_table_artifact = wandb.Artifact("evaluation_artifact", type="dataset")
    evaluation_table_artifact.add(evaluation_table, "evaluation_table")
    # We will also log the raw csv file within an artifact to preserve our data
    evaluation_table_artifact.add_file(args.save_dir / f"{args.model}_{args.prefix}_{args.seed}.csv")
    # Log the table to visualize with a run...
    run.log({"evaluation": evaluation_table})
    # and Log as an Artifact to increase the available row limit!
    run.log_artifact(evaluation_table_artifact)

    model_artifacts = wandb.Artifact('trained_models', type='model')
    # model_artifacts.add_dir(args.trained_models_path)
    model_artifacts.add_file(args.trained_models_path / "ccs_model.pth")

    run.log_artifact(model_artifacts)
    
    