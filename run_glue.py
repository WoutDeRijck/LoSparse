# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PretrainedConfig,
)
from transformers.utils import get_full_repo_name, send_example_telemetry
import utils
import numpy as np

logger = logging.getLogger(__name__)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task to train on.", choices=list(task_to_keys.keys()))
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use mixed precision training")
    parser.add_argument("--low_rank_parameter_ratio", type=float, default=0.05, help="parameter number of low rank matrix / parameter number of original matrix")
    parser.add_argument("--initial_threshold", type=float, default=1.0)
    parser.add_argument("--final_threshold", type=float, default=0.1)
    parser.add_argument("--initial_warmup", type=int, default=1)
    parser.add_argument("--final_warmup", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=6400)
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=1.)
    parser.add_argument("--deltaT", type=int, default=10)
    parser.add_argument("--eval_checkpoint", type=str, default=None, help="Directory containing model checkpoint for evaluation")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="For debugging purposes or quicker evaluation, truncate the number of evaluation examples to this value if set.")
    args = parser.parse_args()

    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    args = parse_args()
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no')

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.task_name is not None:
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    # Load or create model based on whether we're evaluating
    if args.eval_checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.eval_checkpoint} for evaluation")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.eval_checkpoint,
            num_labels=num_labels,
            finetuning_task=args.task_name
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        # Apply model modifications only for training
        allow_name = ['Wqkv', 'Wo', 'Wi', 'dense']
        block_name = ['embeddings', 'norm', 'head', 'classifier', 'final_norm']

        utils.substitute_layer_weights(
            module=model,
            allow_name=allow_name,
            block_name=block_name,
            parameter_ratio=args.low_rank_parameter_ratio,
            do_svd=True
        )

    model.resize_token_embeddings(len(tokenizer))

    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if not is_regression:
                # Use model's label mapping
                result["labels"] = examples["label"]
            else:
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Truncated training dataset to {max_train_samples} examples")

    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Truncated evaluation dataset to {max_eval_samples} examples")

    data_collator = DataCollatorWithPadding(
        tokenizer, 
        pad_to_multiple_of=(8 if args.fp16 else None)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    class PruningCallback(TrainerCallback):
        def __init__(self, pruner):
            self.pruner = pruner
            self.step = 0
            self.logger = get_logger(__name__)

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.step += 1
            if self.step % args.gradient_accumulation_steps == 0:
                model = kwargs['model']
                threshold, mask_threshold = self.pruner.update_and_pruning(model, state.global_step)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and 'eval_accuracy' in metrics:
                self.logger.info(f"epoch {state.epoch:.0f}: {{'accuracy': {metrics['eval_accuracy']}}}")

    # Calculate max_train_steps if not provided
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        print(f"Total training steps: {args.max_train_steps}")

    pruner = utils.Pruner(
        model=model,
        args=args,
        total_step=args.max_train_steps,
        mask_param_name=['sparse'],
        pruner_name='PLATON'
    )

    # If we're only evaluating
    if args.eval_checkpoint is not None:
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return
    
    # Regular training path
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PruningCallback(pruner)],
    )

    trainer.train()
    
    # Save the final model
    if args.output_dir is not None:
        trainer.save_model(args.output_dir)
    
    if args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()