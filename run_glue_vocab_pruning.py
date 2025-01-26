#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

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

class VocabPruningCallback(TrainerCallback):
    def __init__(self, tokenizer, min_freq_threshold=1e-5, update_freq=100, initial_collection_steps=1000):
        self.tokenizer = tokenizer
        self.min_freq_threshold = min_freq_threshold
        self.update_freq = update_freq
        self.initial_collection_steps = initial_collection_steps  # Steps to collect statistics before pruning
        self.token_counts = torch.zeros(len(tokenizer))
        self.total_tokens = 0
        self.step = 0
        self.pruned_vocab = None
        self.token_mapping = None
        self.best_eval_metric = None
        self.eval_metric_name = None
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Track evaluation metrics"""
        if metrics is None:
            return
            
        # Determine which metric to track (once)
        if self.eval_metric_name is None:
            # Choose appropriate metric based on task
            for metric_name in ['accuracy', 'f1', 'matthews_correlation', 'pearson']:
                if metric_name in metrics:
                    self.eval_metric_name = metric_name
                    break
            if self.eval_metric_name is None:
                self.eval_metric_name = list(metrics.keys())[0]  # Fallback to first metric
                
        current_metric = metrics.get(self.eval_metric_name)
        if current_metric is not None:
            if self.best_eval_metric is None or current_metric > self.best_eval_metric:
                self.best_eval_metric = current_metric
                
            logger.info(f"\nEvaluation metrics:")
            logger.info(f"Current {self.eval_metric_name}: {current_metric:.4f}")
            logger.info(f"Best {self.eval_metric_name}: {self.best_eval_metric:.4f}")
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.step += 1
        
        # Only update token counts periodically
        if self.step % self.update_freq != 0:
            return
            
        # Get the embedding layer for ModernBERT
        if hasattr(model, "module"):
            embeddings = model.module.embeddings.tok_embeddings
        else:
            embeddings = model.embeddings.tok_embeddings
            
        # Update token frequency counts
        input_ids = kwargs.get("inputs", {}).get("input_ids")
        if input_ids is not None:
            unique_tokens, counts = torch.unique(input_ids, return_counts=True)
            self.token_counts[unique_tokens] += counts
            self.total_tokens += input_ids.numel()
            
            # Only start pruning after initial collection period
            if self.step <= self.initial_collection_steps:
                if self.step % (self.update_freq * 10) == 0:
                    logger.info(f"Step {self.step}: Collecting token statistics...")
                    logger.info(f"Unique tokens seen: {(self.token_counts > 0).sum()}")
                return
            
            # Calculate token frequencies
            token_freqs = self.token_counts / max(1, self.total_tokens)
            
            # Identify rare tokens (below threshold)
            rare_tokens = torch.where(token_freqs < self.min_freq_threshold)[0]
            
            if len(rare_tokens) > 0:
                # Create clusters of similar tokens based on embedding similarity
                all_embeddings = embeddings.weight.data
                rare_embeddings = all_embeddings[rare_tokens]
                
                # Use cosine similarity to find similar tokens
                sim = torch.nn.functional.cosine_similarity(
                    rare_embeddings.unsqueeze(1),
                    all_embeddings.unsqueeze(0),
                    dim=2
                )
                
                # For each rare token, find the most similar non-rare token
                _, most_similar = sim.topk(k=2, dim=1)  # k=2 to get the most similar non-self token
                replacement_tokens = most_similar[:, 1]  # Skip self-similarity
                
                # Create token mapping
                if self.token_mapping is None:
                    self.token_mapping = torch.arange(len(self.tokenizer))
                self.token_mapping[rare_tokens] = replacement_tokens
                
                # Update embeddings for rare tokens
                with torch.no_grad():
                    embeddings.weight[rare_tokens] = all_embeddings[replacement_tokens]
                
                # Log statistics
                if self.step % (self.update_freq * 10) == 0:
                    logger.info(f"\nStep {self.step} statistics:")
                    logger.info(f"Pruned {len(rare_tokens)} rare tokens")
                    logger.info(f"Vocabulary size reduced from {len(self.tokenizer)} to {len(self.tokenizer) - len(rare_tokens)}")
                    logger.info(f"Embedding layer size: {embeddings.weight.shape}")
                    compression_ratio = 1 - (len(self.tokenizer) - len(rare_tokens)) / len(self.tokenizer)
                    logger.info(f"Vocabulary compression ratio: {compression_ratio:.2%}")
                    if self.best_eval_metric is not None:
                        logger.info(f"Best {self.eval_metric_name}: {self.best_eval_metric:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task with vocabulary pruning")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task to train on.", choices=list(task_to_keys.keys()))
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use mixed precision training")
    
    # Vocabulary pruning parameters
    parser.add_argument("--min_freq_threshold", type=float, default=1e-5, help="Minimum frequency threshold for token pruning")
    parser.add_argument("--vocab_update_freq", type=int, default=100, help="How often to update vocabulary statistics")
    parser.add_argument("--initial_collection_steps", type=int, default=1000, help="Number of steps to collect token statistics before pruning")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no')
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load dataset
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
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    
    # Log initial embedding size
    if hasattr(model, "module"):
        embeddings = model.module.embeddings.tok_embeddings
    else:
        embeddings = model.embeddings.tok_embeddings
    
    initial_vocab_size = embeddings.weight.shape[0]
    embedding_dim = embeddings.weight.shape[1]
    initial_embedding_params = initial_vocab_size * embedding_dim
    
    logger.info(f"Initial vocabulary size: {initial_vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Initial embedding parameters: {initial_embedding_params:,}")
    
    # Initialize vocabulary pruning
    vocab_pruning_callback = VocabPruningCallback(
        tokenizer,
        min_freq_threshold=args.min_freq_threshold,
        update_freq=args.vocab_update_freq,
        initial_collection_steps=args.initial_collection_steps
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            fp16=args.fp16,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=100,
        ),
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"] if "validation" in raw_datasets else None,
        tokenizer=tokenizer,
        callbacks=[vocab_pruning_callback]
    )
    
    # Train the model
    trainer.train()
    
    # Final embedding statistics
    if vocab_pruning_callback.token_mapping is not None:
        unique_tokens = len(torch.unique(vocab_pruning_callback.token_mapping))
        final_embedding_params = unique_tokens * embedding_dim
        logger.info("\nFinal embedding statistics:")
        logger.info(f"Initial vocabulary size: {initial_vocab_size}")
        logger.info(f"Final effective vocabulary size: {unique_tokens}")
        logger.info(f"Initial embedding parameters: {initial_embedding_params:,}")
        logger.info(f"Final effective embedding parameters: {final_embedding_params:,}")
        logger.info(f"Embedding compression ratio: {1 - final_embedding_params/initial_embedding_params:.2%}")
    
    # Save the final model
    trainer.save_model(args.output_dir)
    
    # Save the pruned vocabulary mapping
    if vocab_pruning_callback.token_mapping is not None:
        torch.save(
            vocab_pruning_callback.token_mapping,
            os.path.join(args.output_dir, "token_mapping.pt")
        )

if __name__ == "__main__":
    main() 