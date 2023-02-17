import pickle
import os 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
from elk.utils_evaluation.ccs import CCS
from elk.utils_evaluation.utils_evaluation import (
    get_hidden_states,
    get_permutation,
    split,
)
from elk.utils_evaluation.parser import get_args
import wandb 
import numpy as np 

def train_per_prompt(args, hidden_states, split='train', rate=0.6):
    """
    Train a linear probe on each prompt and return the best loss and accuracy.

    Args:
        args (argparse.Namespace): Arguments.
        hidden_states (list): List of hidden states for each prompt.
    
    Returns:
        losses (list): List of best losses for each prompt.
        accuracies (list): List of best accuracies for each prompt.
    """
    losses = []
    accuracies = []
    models = []

    for i, prompt_hiddens in enumerate(hidden_states):
        run = wandb.init(project=args.project_name, 
                         group=f'Individual Prompts',
                         entity='kozaronek', 
                         reinit=True)
        run.name = f'CCS on Prompt {i}'

        data, labels = prompt_hiddens

        # Emulate get_permutation
        labels_length = len(labels)
        permutation = np.random.permutation(range(labels_length)).reshape(-1)
        permutation = [
            permutation[: int(labels_length * rate)],
            permutation[int(labels_length * rate) :],
        ]

        # Emulate split
        split_idx = 0 if split == "train" else 1
        data = data[permutation[split_idx]]
        labels = labels[permutation[split_idx]]

        print("train probes only on prompt related hidden states")
        ccs_per_prompt = CCS(
            verbose=True, num_tries=1, device=args.model_device, on_each_prompt=True
        )
        half = data.shape[1] // 2
        data = [data[:, :half], data[:, half:]]
        _, best_loss, best_acc, _, _ = ccs_per_prompt.fit(data=data, label=labels)
        losses.append(best_loss)
        accuracies.append(best_acc)
        models.append(ccs_per_prompt)
        print(f"done training prompt related ccs model number: {i}")
        
        wandb.log({
            "loss": best_loss,
            "accuracy": best_acc,
            "prompt": i
        })
        run.finish()

    return losses, accuracies, models


def train(args):

    hidden_states = get_hidden_states(
        hidden_states_directory=args.hidden_states_directory,
        model_name=args.model,
        dataset_name=args.dataset,
        prefix=args.prefix,
        language_model_type=args.language_model_type,
        layer=args.layer,
        mode=args.mode,
        num_data=args.num_data,
    )

    losses, accuracies, all_ccs_per_prompt = train_per_prompt(args, hidden_states)

    # TODO: Set the random seed for the permutation

    # Set the random seed for the permutation
    permutation = get_permutation(hidden_states)
    data, labels = split(
        hidden_states, permutation, prompts=range(len(hidden_states)), split="train"
    )
    assert len(data.shape) == 2
    
    print("train classification model")
    run = wandb.init(project=args.project_name,
                     group=f'Logistic Regression',
                     entity='kozaronek', 
                     reinit=True)
    run.name = f'Logistic Regression on all Prompts'
    
    logistic_regression_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
    logistic_regression_model.fit(data, labels)
    y_pred = logistic_regression_model.predict(data) # TODO: data should be X_test
    logistic_regression_acc = accuracy_score(labels, y_pred)
    
    wandb.log({"accuracy": logistic_regression_acc})
    run.finish()
    print("done training classification model")


    print("train ccs model")
    run = wandb.init(project=args.project_name,
                     group=f'All Prompts',
                     entity='kozaronek', 
                     reinit=True)
    run.name = f'CCS on all Prompts'

    ccs_all_prompts = CCS(verbose=True, device=args.model_device)

    half = data.shape[1] // 2
    data = [data[:, :half], data[:, half:]]
    _, ccs_best_loss, ccs_best_acc, _, _ = ccs_all_prompts.fit(data=data, label=labels)    
    
    wandb.log({
            "loss": ccs_best_loss,
            "accuracy": ccs_best_acc,
        })
    run.finish()

    return logistic_regression_model, ccs_all_prompts, all_ccs_per_prompt

def save_trained_models(args, logistic_regression_model, ccs_all_prompts, all_ccs_per_prompt):
    # save models
    os.makedirs(args.trained_models_path / f"on_{args.dataset}", exist_ok=True)
    os.makedirs(args.trained_models_path / f"on_{args.dataset}/ccs_per_prompt", exist_ok=True)

    # TODO: use better filename for the pkls, so they don't get overwritten
    with open(args.trained_models_path / f"on_{args.dataset}/logistic_regression_model.pkl", "wb") as file:
        pickle.dump(logistic_regression_model, file)
    with open(args.trained_models_path / f"on_{args.dataset}/ccs_model.pkl", "wb") as file:
        pickle.dump(ccs_model, file)
    for i, ccs_per_prompt in enumerate(all_ccs_per_prompt):
        with open(args.trained_models_path / f"on_{args.dataset}/ccs_per_prompt/model_{i}.pkl", "wb") as file:
            pickle.dump(ccs_per_prompt, file)

if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")
    print(f"-------- args = {args} --------")
    
    logistic_regression_model, ccs_model, all_ccs_per_prompt = train(args)
    save_trained_models(args, logistic_regression_model, ccs_model, all_ccs_per_prompt)
