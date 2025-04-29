"""
Custom dataset split utilities for GLUE tasks.
This module provides functions to create custom train/validation/test splits
for GLUE benchmark tasks.
"""

import random
import numpy as np
from datasets import load_dataset, concatenate_datasets

def prepare_custom_split_datasets(task_name, tokenizer, train_ratio=0.8, 
                                 validation_ratio=0.1, test_ratio=0.1, 
                                 max_length=128, random_seed=42):
    """
    Create custom train/validation/test splits from the original dataset.
    
    Args:
        task_name (str): The name of the GLUE task
        tokenizer: The tokenizer to use for processing the text
        train_ratio (float): Ratio of data to use for training
        validation_ratio (float): Ratio of data to use for validation
        test_ratio (float): Ratio of data to use for testing
        max_length (int): Maximum sequence length
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, eval_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load original dataset
    raw_datasets = load_dataset("glue", task_name)
    
    # For MNLI, handle the two validation sets
    if task_name == "mnli":
        # Combine all data for custom splitting
        all_data = raw_datasets["train"]
        
        # We'll make custom splits including the validation data
        if "validation_matched" in raw_datasets and "validation_mismatched" in raw_datasets:
            val_matched = raw_datasets["validation_matched"]
            val_mismatched = raw_datasets["validation_mismatched"]
            # Combine all data
            all_data = concatenate_datasets([all_data, val_matched, val_mismatched])
    else:
        # Combine train and validation for custom splitting
        train_data = raw_datasets["train"]
        if "validation" in raw_datasets:
            val_data = raw_datasets["validation"]
            all_data = concatenate_datasets([train_data, val_data])
        else:
            all_data = train_data
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Get relevant keys for the task
        sentence1_key, sentence2_key = task_to_keys[task_name]
        
        # Create text pairs for tokenization
        texts = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        
        # Tokenize
        result = tokenizer(*texts, padding="max_length", max_length=max_length, truncation=True)
        
        # Copy label
        if "label" in examples:
            result["labels"] = examples["label"]
        
        return result
    
    # Define task to keys mapping
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    
    # Shuffle and split the dataset
    all_data = all_data.shuffle(seed=random_seed)
    total_size = len(all_data)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * validation_ratio)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_dataset = all_data.select(range(train_size))
    eval_dataset = all_data.select(range(train_size, train_size + val_size))
    test_dataset = all_data.select(range(train_size + val_size, total_size))
    
    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
    )
    
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on validation dataset",
    )
    
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on test dataset",
    )
    
    print(f"Created custom splits with sizes: train={len(train_dataset)}, "
          f"validation={len(eval_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, eval_dataset, test_dataset

def prepare_original_val_datasets(task_name, tokenizer, train_ratio=0.9, 
                                 test_ratio=0.1, max_length=128, random_seed=42):
    """
    Use original validation set and split training data into train/test.
    
    Args:
        task_name (str): The name of the GLUE task
        tokenizer: The tokenizer to use for processing the text
        train_ratio (float): Ratio of original training data to use for training
        test_ratio (float): Ratio of original training data to use for testing
        max_length (int): Maximum sequence length
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, eval_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load original dataset
    raw_datasets = load_dataset("glue", task_name)
    
    # Get validation dataset
    if task_name == "mnli":
        eval_dataset = raw_datasets["validation_matched"]
    else:
        eval_dataset = raw_datasets["validation"]
    
    # Split training data
    train_data = raw_datasets["train"].shuffle(seed=random_seed)
    total_train_size = len(train_data)
    
    actual_train_size = int(total_train_size * train_ratio)
    test_size = total_train_size - actual_train_size
    
    # Create train and test splits from original training data
    actual_train_dataset = train_data.select(range(actual_train_size))
    test_dataset = train_data.select(range(actual_train_size, total_train_size))
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Get relevant keys for the task
        sentence1_key, sentence2_key = task_to_keys[task_name]
        
        # Create text pairs for tokenization
        texts = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        
        # Tokenize
        result = tokenizer(*texts, padding="max_length", max_length=max_length, truncation=True)
        
        # Copy label
        if "label" in examples:
            result["labels"] = examples["label"]
        
        return result
    
    # Define task to keys mapping
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    
    # Apply preprocessing
    actual_train_dataset = actual_train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
    )
    
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on validation dataset",
    )
    
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on test dataset",
    )
    
    print(f"Created splits with original validation set: train={len(actual_train_dataset)}, "
          f"validation={len(eval_dataset)}, test={len(test_dataset)}")
    
    return actual_train_dataset, eval_dataset, test_dataset 