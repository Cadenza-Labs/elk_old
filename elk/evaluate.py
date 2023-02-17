import os
import pickle
import numpy as np
import pandas as pd

from elk.utils_evaluation.utils_evaluation import (
    get_hidden_states,
    get_permutation,
    split,
    append_stats,
)
from elk.utils_evaluation.parser import get_args
from elk.utils_evaluation.utils_evaluation import save_df_to_csv
from pathlib import Path
import wandb 


def evaluate(args, logistic_regression_model, ccs_model, ccs_models_per_prompt):
    os.makedirs(args.save_dir, exist_ok=True)

    hidden_states = get_hidden_states(
        hidden_states_directory=args.hidden_states_directory,
        model_name=args.model,
        dataset_name=args.dataset_eval,
        prefix=args.prefix,
        language_model_type=args.language_model_type,
        layer=args.layer,
        mode=args.mode,
        num_data=args.num_data,
    )
    permutation = get_permutation(hidden_states)

    accuracies_ccs = []
    losses_ccs = []
    accuracies_lr = []
    losses_lr = []
    for prompt_idx in range(len(hidden_states)):
        data, labels = split(
            hidden_states=hidden_states,
            permutation=permutation,
            prompts=[prompt_idx],
            split="test",
        )

        # evaluate classification model
        print("evaluate classification model")
        acc_lr = logistic_regression_model.score(data, labels)
        accuracies_lr.append(acc_lr)
        losses_lr.append(0)  # TODO: get loss from lr somehow

        # evaluate ccs model
        print("evaluate ccs model on all prompts")
        half = data.shape[1] // 2
        data = [data[:, :half], data[:, half:]]
        acc_ccs, loss_ccs = ccs_model.score(data, labels, getloss=True)
        accuracies_ccs.append(acc_ccs)
        losses_ccs.append(loss_ccs)

        # evaluate ccs model
        accuracies_ccs_per_prompt = []
        losses_ccs_per_prompt = []
        print("evaluate ccs model per prompt")
        for model in ccs_models_per_prompt:
            acc_ccs, loss_ccs = model.score(data, labels, getloss=True)
            accuracies_ccs_per_prompt.append(acc_ccs)
            losses_ccs_per_prompt.append(loss_ccs)

    avg_accuracy_ccs_per_prompt = np.mean(accuracies_ccs_per_prompt)
    avg_accuracy_std_ccs_per_prompt = np.std(accuracies_ccs_per_prompt)
    avg_loss_ccs_per_prompt = np.mean(losses_ccs_per_prompt)

    avg_accuracy_ccs = np.mean(accuracies_ccs)
    avg_accuracy_std_ccs = np.std(accuracies_ccs)
    avg_loss_ccs = np.mean(losses_ccs)

    avg_accuracy_lr = np.mean(accuracies_lr)
    avg_accuracy_std_lr = np.std(accuracies_lr)
    avg_loss_lr = np.mean(losses_lr)

    print("avg_accuracy_ccs_per_prompt", avg_accuracy_ccs_per_prompt)
    print("avg_accuracy_std_ccs_per_prompt", avg_accuracy_std_ccs_per_prompt)
    print("avg_loss_ccs_per_prompt", avg_loss_ccs_per_prompt)

    print("avg_accuracy_ccs", avg_accuracy_ccs)
    print("avg_accuracy_std_ccs", avg_accuracy_std_ccs)
    print("avg_loss_ccs", avg_loss_ccs)

    print("avg_accuracy_lr", avg_accuracy_lr)
    print("avg_accuracy_std_lr", avg_accuracy_std_lr)
    print("avg_loss_lr", avg_loss_lr)

    stats_df = pd.DataFrame(
        columns=[
            "model",
            "prefix",
            "method",
            "prompt_level",
            "train",
            "test",
            "accuracy",
            "std",
        ]
    )
    stats_df = append_stats(
        stats_df, args, "ccs_per_prompt", avg_accuracy_ccs_per_prompt, avg_accuracy_std_ccs_per_prompt, avg_loss_ccs_per_prompt
    )
    stats_df = append_stats(
        stats_df, args, "ccs", avg_accuracy_ccs, avg_accuracy_std_ccs, avg_loss_ccs
    )
    stats_df = append_stats(
        stats_df, args, "lr", avg_accuracy_lr, avg_accuracy_std_lr, avg_loss_lr
    )
    save_df_to_csv(args, stats_df, args.prefix, "After finish")

def log_evaluation_results(args, run):
    dir = args.save_dir / f"{args.model}_{args.prefix}_{args.seed}_trained_{args.dataset}_eval_{args.dataset_eval}.csv"
    evaluation_df = pd.read_csv(dir)
    evaluation_table = wandb.Table(dataframe=evaluation_df)
    # Add the table to an Artifact to make it easier to reuse!
    evaluation_table_artifact = wandb.Artifact("evaluation_artifact", type="dataset")
    evaluation_table_artifact.add(evaluation_table, "evaluation_table")
    # We will also log the raw csv file within an artifact to preserve our data
    evaluation_table_artifact.add_file(dir)
    # Log the table to visualize with a run...
    run.log({"evaluation": evaluation_table})
    # and Log as an Artifact to increase the available row limit!
    run.log_artifact(evaluation_table_artifact)

if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")

    run = wandb.init(project=args.project_name, entity='kozaronek')

    # load pickel from file
    ccs_models_per_prompt = []
    files = Path(args.trained_models_path / f"on_{args.dataset}/ccs_per_prompt").glob('*')   
    for ccs_per_prompt_path in files:
        with open(ccs_per_prompt_path, "rb") as file:
            ccs_models_per_prompt.append(pickle.load(file))
    with open(args.trained_models_path / f"on_{args.dataset}/logistic_regression_model.pkl", "rb") as file:
        logistic_regression_model = pickle.load(file)
    with open(args.trained_models_path / f"on_{args.dataset}/ccs_model.pkl", "rb") as file:
        ccs_model = pickle.load(file)

    evaluate(args, logistic_regression_model, ccs_model, ccs_models_per_prompt)
    log_evaluation_results(args, run)
    run.finish()