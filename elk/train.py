import pickle

from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
from elk.utils_evaluation.ccs import CCS
from elk.utils_evaluation.utils_evaluation import (
    get_hidden_states,
    get_permutation,
    split,
)
from elk.utils_evaluation.parser import get_args


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

    losses = []
    accuraries = []
    # Set the random seed for the permutation#
    for i, prompt_hiddens in enumerate(hidden_states):
        data, labels = prompt_hiddens
        # Now train linear probe on hiddens
        print("train probes only on prompt related hidden states")
        ccs_per_prompt = CCS(
            verbose=True, num_tries=1, device=args.model_device, on_each_prompt=True
        )
        half = data.shape[1] // 2
        data = [data[:, :half], data[:, half:]]
        _, loss, acc = ccs_per_prompt.fit(data=data, label=labels)
        losses.append(loss)
        accuraries.append(acc)
        print(f"done training prompt related ccs model number: {i}")

    permutation = get_permutation(hidden_states)
    data, labels = split(
        hidden_states, permutation, prompts=range(len(hidden_states)), split="train"
    )
    assert len(data.shape) == 2

    print("train classification model")
    logistic_regression_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
    logistic_regression_model.fit(data, labels)
    print("done training classification model")

    print("train ccs model")
    ccs_per_prompt = CCS(verbose=True, device=args.model_device)
    half = data.shape[1] // 2
    data = [data[:, :half], data[:, half:]]
    _, ccs_best_loss, ccs_best_acc = ccs_per_prompt.fit(data=data, label=labels)
    print("done training ccs model")

    print("\n\nRESULTS")

    print("\n\nAverage results of training on each prompt.")
    print(f"Average loss: {sum(losses)/len(losses)}")
    print(f"Average accuracy: {sum(accuraries)/len(accuraries)}")

    print("\n\nAverage results of training on each prompt.")
    print(f"Best loss: {ccs_best_loss}")
    print(f"Best accuracy: {ccs_best_acc}")

    return logistic_regression_model, ccs_per_prompt


if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")
    print(f"-------- args = {args} --------")

    logistic_regression_model, ccs_model = train(args)

    # save models
    # TODO: use better filename for the pkls, so they don't get overwritten
    os.makedirs(args.trained_models_path, exist_ok=True)
    with open(args.trained_models_path / "logistic_regression_model.pkl", "wb") as file:
        pickle.dump(logistic_regression_model, file)
    with open(args.trained_models_path / "ccs_model.pkl", "wb") as file:
        pickle.dump(ccs_model, file)
