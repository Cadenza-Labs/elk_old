from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
)
import os
import torch
import pandas as pd
from elk.utils_generation.construct_prompts import construct_prompt_dataframe, prompt_dict
from elk.utils_generation.save_utils import get_directory
from datasets import load_dataset
from promptsource.templates import DatasetTemplates



def load_model(mdl_name, cache_dir):
    """
    Load model from cache_dir or from HuggingFace model hub.

    Args:
        mdl_name (str): name of the model
        cache_dir (str): path to the cache directory

    Returns:
        model (torch.nn.Module): model in evaluation mode
    """
    if mdl_name in ["gpt-neo-2.7B", "gpt-j-6B"]:
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{mdl_name}", cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"bigscience/{mdl_name}", cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        model = T5ForConditionalGeneration.from_pretrained(f"allenai/{mdl_name}", cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{mdl_name}", cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            mdl_name, cache_dir=cache_dir
        )
    elif "t5" in mdl_name:
        model = AutoModelWithLMHead.from_pretrained(mdl_name, cache_dir=cache_dir)
    
    # We only use the models for inference, so we don't need to train them and hence don't need to track gradients
    return model.eval()


def put_model_on_device(model, parallelize, device="cuda"):
    """
    Put model on device (hardware accelearator like "cuda", "mps" or simply "cpu").
    
    Args:
        model (torch.nn.Module): model to put on device
        parallelize (bool): whether to parallelize the model
        device (str): device to put the model on

    Returns:
        model (torch.nn.Module): model on device
    """
    if device == "mps":
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
        else:
            mps_device = torch.device("mps")
            model.to(mps_device)
    elif parallelize:
        model.parallelize()
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
        else:
            model.to("cuda")
    else:
        model.to("cpu")

    return model


def load_tokenizer(mdl_name, cache_dir):
    """
    Load tokenizer for the model.

    Args:
        mdl_name (str): name of the model
        cache_dir (str): path to the cache directory

    Returns:
        tokenizer: tokenizer for the model
    """
    if mdl_name in ["gpt-neo-2.7B", "gpt-j-6B"]:
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{mdl_name}", cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2Tokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"bigscience/{mdl_name}", cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"allenai/{mdl_name}", cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{mdl_name}", cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "t5" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)

    return tokenizer

def get_sample_data(dataset_name, split_dataframes, sample_amount):
    '''
    Get shuffled sample data from the dataset.
    Args:
        dataset_name:   the name of the dataset, some datasets have special token name.
        split_dataframes:  a list of dataframes corresponding to the split of the dataset (train, test, eval etc.)
        num_data:    number of data point we want to sample, default is twice as final size, considering that some examples are too long and could be dropped.
   
    Returns:
        shuffled_dataframe: a dataframe containing the sample data
    '''

    label_column_name = "label" if dataset_name != "story-cloze" else "answer_right_ending"
    all_label_names = set(split_dataframes[0][label_column_name].to_list())
    num_labels = len(all_label_names)
    balanced_sample_num = get_balanced_sample_num(sample_amount, num_labels)

    # TODO: Do we need to shuffle the data twice?
    # randomize data frac=1 means that we take the return 100% of the dataset
    shuffled_split_dataframes = [dataframe.sample(frac=1).reset_index(drop=True) for dataframe in split_dataframes]

    tmp_dataframes = []
    # These splits do not necessarily correspond to train and test, they could also be train and eval or something else 
    # (It depends on the dataset splits that the dataset offers through HuggingFace)
    train_split = shuffled_split_dataframes[0]
    test_split = shuffled_split_dataframes[1]
    for label_idx, label in enumerate(all_label_names):
        label_subset = test_split[label_column_name] == label
        train_subset = train_split[label_subset] 
        train_split_size = len(train_subset)
        if train_split_size < balanced_sample_num[label_idx]:
            # Sample only until train_split_size, because we don't want to sample more than we have
            test_subset = test_split[label_subset][: balanced_sample_num[label_idx] - train_split_size]
            tmp_dataframes.append(pd.concat([train_subset, test_subset], ignore_index=True))
        else:
            sample_amount = balanced_sample_num[label_idx]
            tmp_dataframes.append(train_subset.sample(sample_amount).reset_index(drop=True))

    # TODO: WHY ARE WE CONCATENATING THE Train and Test split data?
    full_dataframe = pd.concat(tmp_dataframes)
    # TODO: Do we need to shuffle the data twice?
    shuffled_dataframe = full_dataframe.sample(frac=1).reset_index(drop=True)
    return shuffled_dataframe


def get_balanced_sample_num(sample_amount, num_labels):
    """
    Get the number of samples per label to balance the dataset.
    
    Args:
        sample_amount: int, the number of samples to take.
        num_labels: int, the number of labels in the dataset.
        
    Returns:
        balanced_sample_num: list of int, the number of samples per label.
    """

    samples_per_group = sample_amount // num_labels
    remaining_samples = sample_amount - samples_per_group * num_labels
    balanced_sample_num = []
    for i in range(num_labels):
        if i < num_labels - remaining_samples:
            balanced_sample_num.append(samples_per_group)
        else:
            balanced_sample_num.append(samples_per_group + 1)
    return balanced_sample_num


def get_hugging_face_load_name(dataset_name):
    """
    Get the name of the dataset in the right format for the huggingface datasets library.
    
    Args:
        dataset_name: str, the name of the dataset.
    
    Returns:
        list of str, the name of the dataset in the right format for the huggingface datasets library.
    """
    if dataset_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "piqa"]:
        return [dataset_name.replace("-", "_")]
    elif dataset_name in ["copa", "rte", "boolq"]:
        return ["super_glue", dataset_name.replace("-", "_")]
    elif dataset_name in ["qnli"]:
        return ["glue", dataset_name.replace("-", "_")]
    elif dataset_name == "story-cloze":
        return ["story_cloze", "2016"]

def get_raw_dataset(dataset_name, cache_dir):
    '''
    This function will load datasets from module or raw csv, and then return a pd DataFrame.
    This DataFrame can be used to construct the example.

    Args:
        dataset_name:   the name of the dataset
        cache_dir:  the directory where the dataset might have been cached in the past

    Returns:
        raw_dataset:   A `DatasetDict` object containing the raw dataset
    '''
    if dataset_name != "story-cloze":
        raw_dataset = load_dataset(*get_hugging_face_load_name(dataset_name), cache_dir=cache_dir)
    else:
        raw_dataset = load_dataset(*get_hugging_face_load_name(dataset_name),cache_dir=cache_dir, data_dir="./datasets/rawdata")

    if dataset_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]:
        dataset_split_names = ["test", "train"]
    elif dataset_name in ["copa", "rte", "boolq", "piqa", "qnli"]:
        dataset_split_names = ["validation", "train"]
    elif dataset_name in ["story-cloze"]:
        dataset_split_names = ["test", "validation"]

    return raw_dataset, dataset_split_names

def create_setname_to_promptframe(data_base_dir, all_dataset_names, num_prompts_per_dataset, num_data, tokenizer, save_base_dir, model, prefix, token_place):
    """
    This function will create a dictionary of dataframes,
    where the key is the dataset name and the value is the dataframe.

    Args:
        data_base_dir: str, the directory to save the raw data csv files.
        all_dataset_names: list of str, the names of the datasets.
        num_prompts_per_dataset: list of int, the number of prompts per dataset.
        num_data: int, the number of data points that will be used for each dataset.
        tokenizer: transformers.PreTrainedTokenizer, tokenizer associated with the model.
        save_base_dir: str, the directory to save the prompt dataframes.
        model: str, the name of the model.
        prefix: str, the prefix of the prompt.
        token_place: str, Determine which token's hidden states will be generated. Can be `first` or `last` or `average`.

    Returns:
        name_to_dataframe: dict, the dictionary of dataframes.
    """
    create_directory(data_base_dir)
    name_to_dataframe = {}
    for dataset_name, num_prompts in zip(all_dataset_names, num_prompts_per_dataset):
        for prompt_idx in range(num_prompts):
            path = os.path.join(data_base_dir, f"rawdata_{dataset_name}_{num_data}.csv")
        
            dataset_name_with_num = f"{dataset_name}_{num_data}_prompt{prompt_idx}"
            complete_path = get_directory(save_base_dir, model, dataset_name_with_num, prefix, token_place)
            dataframe_path = os.path.join(complete_path, "frame.csv")
 
            if os.path.exists(dataframe_path):
                prompt_dataframe = pd.read_csv(dataframe_path, converters={"selection": eval})
                name_to_dataframe[dataset_name_with_num] = prompt_dataframe
            else:  
                cache_dir = os.path.join(data_base_dir, "cache")
                create_directory(cache_dir)

                # This is a dataframe with random order data
                # Can just take enough data from scratch and then stop as needed
                # the length of raw_data will be 2 times as the intended length
                raw_dataset, dataset_split_names = get_raw_dataset(dataset_name, cache_dir)
                split_dataframes = [raw_dataset[split_name].to_pandas() for split_name in dataset_split_names]
                raw_data = get_sample_data(dataset_name, split_dataframes, 2 * num_data)
                raw_data.to_csv(path, index=False)

                prompt_dataframe = construct_prompt_dataframe(dataset_name, raw_data,
                                                              prompt_idx=prompt_idx, mdl_name=model, 
                                                              tokenizer=tokenizer, max_num = num_data, 
                                                              confusion = prefix)
                name_to_dataframe[dataset_name_with_num] = prompt_dataframe

                # Save data
                create_directory(save_base_dir)
                create_directory(complete_path)
                complete_frame_csv_path = os.path.join(complete_path, "frame.csv")
                prompt_dataframe.to_csv(complete_frame_csv_path, index = False)

    return name_to_dataframe


def create_directory(name):
    """
    This function will create a directory if it does not exist.
    """
    if not os.path.exists(name):
        os.makedirs(name)

def get_num_templates_per_dataset(all_dataset_names):
    """
    This function calculates the number of available prompt templates per dataset in a list of dataset names.

    Args:
        all_dataset_names: list of str, the names of the datasets.
        
    Returns:
        num_templates_per_dataset: list of int, contains number of prompt templates per dataset.
    """
    num_templates_per_dataset = []

    for dataset_name in all_dataset_names:
        amount_of_templates = 0
        # if dataset_name in prompt_dict.keys():
        #     amount_of_templates += len(prompt_dict[dataset_name])
        if dataset_name not in ["ag-news", "dbpedia-14"]:
            amount_of_templates += len(DatasetTemplates("imdb").all_template_names)
        if dataset_name == "copa":
            amount_of_templates -= 4  # do not use the last four prompts
        num_templates_per_dataset.append(amount_of_templates)

    print("num_templates_per_dataset", num_templates_per_dataset)
    return num_templates_per_dataset
