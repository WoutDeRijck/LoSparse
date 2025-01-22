import torch
import torch.nn as nn
from pss import ParameterSharedEmbedding
import math
from transformers import AutoModelForSequenceClassification

def test_embedding_approximation(model: nn.Module, parameter_ratio=0.05, num_samples=1000):
    # Print parameter counts per layer
    for name, module in model.named_modules():
        if "embeddings.tok_embeddings" in name:
            orig_embedding = module
            break
    
    # Add logging for original embedding details
    print(f"\nOriginal Embedding Details:")
    print(f"Vocabulary Size: {orig_embedding.num_embeddings:,}")
    print(f"Embedding Dimension: {orig_embedding.embedding_dim}")
    print(f"Total Parameters: {orig_embedding.num_embeddings * orig_embedding.embedding_dim:,}")
    
    epsilon = math.sqrt(1 / (parameter_ratio * orig_embedding.num_embeddings))
    print(f"\nPSS Configuration:")
    print(f"Parameter Ratio: {parameter_ratio:.3f}")
    print(f"Epsilon: {epsilon:.4f}")
    
    # Create PSS embedding layer
    pss_embedding = ParameterSharedEmbedding(
        num_embeddings=orig_embedding.num_embeddings,
        embedding_dim=orig_embedding.embedding_dim,
        epsilon=epsilon,
        device=orig_embedding.weight.device
    )

    # Initialize from pretrained embeddings
    pss_embedding.from_pretrained(orig_embedding.weight)
    """Test the quality of PSS embedding approximation"""
    device = orig_embedding.weight.device
    vocab_size = orig_embedding.num_embeddings
    
    # Add logging for sample statistics
    print(f"\nSampling Statistics:")
    print(f"Number of token pairs sampled: {num_samples:,}")
    print(f"Random token range: [0, {vocab_size:,})")
    
    # Sample random token pairs
    tokens1 = torch.randint(0, vocab_size, (num_samples,), device=device)
    tokens2 = torch.randint(0, vocab_size, (num_samples,), device=device)
    
    # Get original embeddings
    orig_emb1 = orig_embedding(tokens1)
    orig_emb2 = orig_embedding(tokens2)
    
    # Get PSS embeddings
    pss_emb1 = pss_embedding(tokens1.unsqueeze(0)).squeeze(0)
    pss_emb2 = pss_embedding(tokens2.unsqueeze(0)).squeeze(0)
    
    # Compute inner products
    orig_inner = torch.sum(orig_emb1 * orig_emb2, dim=1)
    pss_inner = torch.sum(pss_emb1 * pss_emb2, dim=1)
    
    # Compute relative error
    rel_error = torch.abs(orig_inner - pss_inner) / (torch.norm(orig_emb1, dim=1) * torch.norm(orig_emb2, dim=1))
    
    # Add detailed error analysis
    print(f"\nApproximation Error Analysis:")
    print(f"Mean Relative Error: {rel_error.mean().item():.6f}")
    print(f"Max Relative Error: {rel_error.max().item():.6f}")
    print(f"Error Std Dev: {rel_error.std().item():.6f}")
    print(f"Theoretical Error Bound: {pss_embedding.epsilon:.6f}")
    
    # Calculate what percentage of samples are within different error bounds
    within_bound = (rel_error <= pss_embedding.epsilon).float().mean().item()
    within_half_bound = (rel_error <= pss_embedding.epsilon/2).float().mean().item()
    print(f"\nError Distribution:")
    print(f"Samples within theoretical bound: {within_bound*100:.2f}%")
    print(f"Samples within half theoretical bound: {within_half_bound*100:.2f}%")
    
    return {
        'mean_error': rel_error.mean().item(),
        'max_error': rel_error.max().item(),
        'std_error': rel_error.std().item(),
        'theoretical_bound': pss_embedding.epsilon
    } 

model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base")
result = test_embedding_approximation(model, parameter_ratio=0.05)
print(result)