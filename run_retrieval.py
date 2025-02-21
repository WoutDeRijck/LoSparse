"""Finetuning a SentenceTransformer model for retrieval tasks (MSMARCO, MLDR)."""
import argparse
import logging
import math
import os
from pathlib import Path

import datasets
from datasets import load_dataset
from huggingface_hub import Repository, create_repo, get_full_repo_name
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from accelerate.utils import set_seed
from transformers import TrainerCallback
import utils

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a SentenceTransformer model on a retrieval task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the task to train on.", choices=["msmarco", "mldr"])
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use mixed precision training")
    parser.add_argument("--low_rank_parameter_ratio", type=float, default=0.05, help="parameter number of low rank matrix / parameter number of original matrix")
    parser.add_argument("--initial_threshold", type=float, default=1.0)
    parser.add_argument("--final_threshold", type=float, default=0.15)
    parser.add_argument("--initial_warmup", type=int, default=1)
    parser.add_argument("--final_warmup", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of steps for warmup, will be calculated if not provided")
    parser.add_argument("--beta1", type=float, default=0.85)
    parser.add_argument("--beta2", type=float, default=1.)
    parser.add_argument("--deltaT", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="For debugging purposes or quicker evaluation, truncate the number of evaluation examples to this value if set.")
    parser.add_argument("--mldr_language", type=str, default="en", choices=['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh'], help="Language for MLDR dataset")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Set up repository for model hub
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

    # Load datasets
    if args.task_name == "msmarco":
        # Load MS MARCO dataset
        raw_datasets = load_dataset(
            "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
            "triplet-hard",
            split="train",
        )
        # Split into train and validation
        dataset_dict = raw_datasets.train_test_split(test_size=1000, seed=12)
        raw_datasets = datasets.DatasetDict({
            "train": dataset_dict["train"].select(range(1_250_000)),
            "validation": dataset_dict["test"]
        })
    else:  # MLDR
        # Load MLDR dataset for specific language
        train_dataset = load_dataset('Shitao/MLDR', args.mldr_language, split='train', trust_remote_code=True)
        eval_dataset = load_dataset('Shitao/MLDR', args.mldr_language, split='dev', trust_remote_code=True)
        
        # Convert dataset to the format needed for training, exactly like in train_st_mldr.py
        def format_for_training(dataset):
            from datasets import Dataset
            # Extract text from passages and ensure they're strings
            return Dataset.from_dict({
                'query': dataset['query'],
                'positive': [str(passages[0]['text']) for passages in dataset['positive_passages']],
                'negative': [str(passages[0]['text']) for passages in dataset['negative_passages']]
            })
        
        train_data = format_for_training(train_dataset)
        eval_data = format_for_training(eval_dataset)

        if args.max_train_samples is not None:
            train_data = train_data.select(range(min(len(train_data), args.max_train_samples)))
        if args.max_eval_samples is not None:
            eval_data = eval_data.select(range(min(len(eval_data), args.max_eval_samples)))

        # Initialize model with flash attention
        model = SentenceTransformer(args.model_name_or_path, 
                                  model_kwargs={"attn_implementation": "flash_attention_2"})
        
        # Move model to GPU first
        model = model.to('cuda')
        
        # Apply pruning modifications
        allow_name = ['Wqkv', 'Wo', 'Wi', 'dense']
        block_name = ['embeddings', 'norm', 'head', 'classifier', 'final_norm']
        
        # Get the underlying transformer model
        transformer_model = model._first_module().auto_model
        utils.substitute_layer_weights(
            module=transformer_model,
            allow_name=allow_name,
            block_name=block_name,
            parameter_ratio=args.low_rank_parameter_ratio,
            do_svd=True
        )
        
        # Update the transformer model in the sentence transformer and ensure it's on GPU
        model._first_module().auto_model = transformer_model.to('cuda')
        # Ensure the model is in the correct mode
        model.train()

        # Create pruning callback
        class STPruningCallback(TrainerCallback):
            def __init__(self, pruner: utils.Pruner, sentence_transformer: SentenceTransformer):
                self.pruner = pruner
                self.sentence_transformer = sentence_transformer
                self.step = 0
                self.logger = logging.getLogger(__name__)
                # Store the pruning masks
                self.pruning_masks = {}
                # Ensure pruning masks are on GPU
                self.device = next(sentence_transformer.parameters()).device

            def count_zero_params(self, model):
                total_params = 0
                zero_params = 0
                for name, param in model.named_parameters():
                    if 'sparse' in name:  # Only count parameters that can be pruned
                        param_count = param.numel()
                        zero_count = (param == 0).sum().item()
                        total_params += param_count
                        zero_params += zero_count
                return total_params, zero_params

            def on_init_end(self, args, state, control, **kwargs):
                # Count initial zero parameters
                total_params, zero_params = self.count_zero_params(self.sentence_transformer._first_module().auto_model)
                self.logger.info(f"Initial prunable parameters: {total_params}")
                self.logger.info(f"Initial zero parameters: {zero_params} ({100 * zero_params / total_params:.2f}%)")
                return control

            def on_step_end(self, args, state, control, **kwargs):
                self.step += 1
                if self.step % args.gradient_accumulation_steps == 0:
                    # Get the underlying transformer model
                    transformer_model = self.sentence_transformer._first_module().auto_model
                    
                    # Ensure gradients are properly synced
                    for name, param in transformer_model.named_parameters():
                        if 'sparse' in name and param.grad is None and hasattr(param, '_grad_from_st'):
                            param.grad = param._grad_from_st.to(self.device)

                    # Update and prune
                    threshold, mask_threshold = self.pruner.update_and_pruning(transformer_model, state.global_step)
                    
                    if mask_threshold is not None:
                        # Store pruning masks
                        for name, param in transformer_model.named_parameters():
                            if 'sparse' in name:
                                self.pruning_masks[name] = (param.data == 0).clone().to(self.device)
                    elif self.pruning_masks:
                        # Reapply existing masks
                        for name, param in transformer_model.named_parameters():
                            if name in self.pruning_masks:
                                param.data.masked_fill_(self.pruning_masks[name].to(self.device), 0.0)
                    
                    # Count and log zero parameters after pruning
                    total_params, zero_params = self.count_zero_params(transformer_model)
                    self.logger.info(f"Step {state.global_step}: Zero parameters: {zero_params}/{total_params} ({100 * zero_params / total_params:.2f}%)")
                    
                    # Update the transformer model in the sentence transformer
                    self.sentence_transformer._first_module().auto_model = transformer_model

                    # Register hooks to capture gradients from sentence transformer
                    def grad_hook(grad, param_name):
                        param = dict(transformer_model.named_parameters())[param_name]
                        param._grad_from_st = grad.to(self.device)
                        return grad

                    for name, param in transformer_model.named_parameters():
                        if 'sparse' in name:
                            param.register_hook(lambda grad, n=name: grad_hook(grad, n))

                return control

            def on_save(self, args, state, control, **kwargs):
                """Called before saving a checkpoint"""
                if self.pruning_masks:
                    # Get the transformer model
                    transformer_model = self.sentence_transformer._first_module().auto_model
                    
                    # Reapply masks before saving
                    for name, param in transformer_model.named_parameters():
                        if name in self.pruning_masks:
                            param.data.masked_fill_(self.pruning_masks[name].to(self.device), 0.0)
                    
                    # Store masks in the model's state
                    transformer_model.pruning_masks = {k: v.clone() for k, v in self.pruning_masks.items()}
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Called during evaluation"""
                # Get the transformer model
                transformer_model = self.sentence_transformer._first_module().auto_model
                
                # First reapply masks before any evaluation happens
                if hasattr(transformer_model, 'pruning_masks'):
                    for name, param in transformer_model.named_parameters():
                        if name in transformer_model.pruning_masks:
                            param.data.masked_fill_(transformer_model.pruning_masks[name].to(self.device), 0.0)
                elif self.pruning_masks:
                    for name, param in transformer_model.named_parameters():
                        if name in self.pruning_masks:
                            param.data.masked_fill_(self.pruning_masks[name].to(self.device), 0.0)
                
                # Count zero parameters before evaluation
                total_params, zero_params = self.count_zero_params(transformer_model)
                self.logger.info(f"\nPre-evaluation state at epoch {state.epoch:.0f}:")
                self.logger.info(f"Zero parameters: {zero_params}/{total_params} ({100 * zero_params / total_params:.2f}%)")
                
                # After evaluation completes
                if metrics is not None:
                    self.logger.info(f"Evaluation metrics: {metrics}")
                    
                    # Verify masks are still applied
                    post_total_params, post_zero_params = self.count_zero_params(transformer_model)
                    if post_zero_params != zero_params:
                        self.logger.warning("Zero parameter count changed during evaluation!")
                        self.logger.warning(f"Pre-evaluation zeros: {zero_params}, Post-evaluation zeros: {post_zero_params}")
                        # Reapply masks if they were lost
                        if hasattr(transformer_model, 'pruning_masks'):
                            for name, param in transformer_model.named_parameters():
                                if name in transformer_model.pruning_masks:
                                    param.data.masked_fill_(transformer_model.pruning_masks[name].to(self.device), 0.0)
                        elif self.pruning_masks:
                            for name, param in transformer_model.named_parameters():
                                if name in self.pruning_masks:
                                    param.data.masked_fill_(self.pruning_masks[name].to(self.device), 0.0)
                
                return control

            def on_train_end(self, args, state, control, **kwargs):
                """Called at the end of training"""
                # Get the transformer model
                transformer_model = self.sentence_transformer._first_module().auto_model
                
                # Reapply masks for final model
                if self.pruning_masks:
                    for name, param in transformer_model.named_parameters():
                        if name in self.pruning_masks:
                            param.data.masked_fill_(self.pruning_masks[name].to(self.device), 0.0)
                
                # Count and log final zero parameters
                total_params, zero_params = self.count_zero_params(transformer_model)
                self.logger.info("\n=== Final Model Statistics ===")
                self.logger.info(f"Total parameters: {total_params}")
                self.logger.info(f"Zero parameters: {zero_params}")
                self.logger.info(f"Pruning ratio: {100 * zero_params / total_params:.2f}%")
                return control

        # Calculate max steps for pruner
        num_update_steps_per_epoch = math.ceil(len(train_data) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
        # Adjust warmup steps for more gradual pruning
        if args.warmup_steps is None:
            # Use 10% of total steps for warmup (more gradual than 6%)
            args.warmup_steps = max(100, int(0.10 * max_train_steps))
        
        logger.info(f"Total training steps: {max_train_steps}")
        logger.info(f"Steps per epoch: {num_update_steps_per_epoch}")
        logger.info(f"Warmup steps: {args.warmup_steps}")
        logger.info(f"Initial warmup period: {args.initial_warmup * args.warmup_steps} steps")
        logger.info(f"Final warmup period: {args.final_warmup * args.warmup_steps} steps")
        logger.info("Pruning schedule:")
        logger.info(f"- No pruning: steps 0-{args.initial_warmup * args.warmup_steps}")
        logger.info(f"- Gradual pruning: steps {args.initial_warmup * args.warmup_steps + 1}-{max_train_steps - args.final_warmup * args.warmup_steps}")
        logger.info(f"- Final pruning: steps {max_train_steps - args.final_warmup * args.warmup_steps + 1}-{max_train_steps}")
        
        # Initialize pruner with transformer model and ensure it's on GPU
        pruner = utils.Pruner(
            model=transformer_model.to('cuda'),
            args=args,
            total_step=max_train_steps,
            mask_param_name=['sparse'],
            pruner_name='PLATON'
        )

        # Create evaluator with more detailed metrics
        dev_evaluator = InformationRetrievalEvaluator(
            queries=dict(zip(range(len(eval_data['query'])), eval_data['query'])),
            corpus=dict(zip(range(len(eval_data['positive'])), eval_data['positive'])),
            relevant_docs={i: {i} for i in range(len(eval_data['query']))},
            corpus_chunk_size=512,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1],
            precision_recall_at_k=[10],
            map_at_k=[10],
            show_progress_bar=True,
            name=f"mldr-{args.mldr_language}-dev"
        )

        # Evaluate base model
        logger.info("\n=== Evaluating Base Model ===")
        base_score = dev_evaluator(model)
        logger.info(f"Base Model Score: {base_score}\n")

        # Define loss function for MLDR
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

        run_name = f"{args.model_name_or_path.split('/')[-1]}-MLDR-{args.mldr_language}-{args.learning_rate}"
        st_training_args = SentenceTransformerTrainingArguments(
            output_dir=f"{args.output_dir}/{run_name}",
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            warmup_steps=args.warmup_steps,
            fp16=False,
            bf16=True,
            learning_rate=args.learning_rate,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            report_to=["none"],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=max_train_steps,
        )

        # Create trainer with pruning callback
        trainer = SentenceTransformerTrainer(
            model=model,
            args=st_training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            loss=loss,
            evaluator=dev_evaluator,
            callbacks=[STPruningCallback(pruner, model)]
        )

        # Train and save
        trainer.train()
        
        # Evaluate final model
        logger.info("\n=== Evaluating Final Model ===")
        final_score = dev_evaluator(model)
        logger.info(f"Final Model Score: {final_score}\n")
        
        model.save_pretrained(f"{args.output_dir}/{run_name}/final")

        if args.push_to_hub:
            trainer.push_to_hub()

if __name__ == "__main__":
    main() 