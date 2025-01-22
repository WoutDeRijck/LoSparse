import torch
import torch.nn as nn
import math

class JLTMatrix(nn.Module):
    """Johnson-Lindenstrauss Transform matrix implementation"""
    def __init__(self, n: int, k: int, sparse: bool = True, device=None):
        """
        Args:
            n: Original dimension (vocabulary size)
            k: Target dimension after projection
            sparse: Whether to use sparse JLT matrix (more memory efficient)
            device: torch device
        """
        super().__init__()
        self.n = n
        self.k = k
        self.sparse = sparse
        
        if sparse:
            # Sparse JLT matrix implementation
            # Each column has s=log(n) non-zero entries
            s = int(math.ceil(math.log(n)))
            indices = []
            values = []
            
            for j in range(n):
                # Randomly select s positions for non-zero entries
                pos = torch.randperm(k)[:s]
                signs = torch.randint(0, 2, (s,), device=device) * 2 - 1  # Random ±1
                indices.extend([(p.item(), j) for p in pos])
                values.extend((signs / math.sqrt(s)).tolist())
            
            indices = torch.tensor(indices, device=device).t()
            values = torch.tensor(values, device=device)
            
            self.S = nn.Parameter(
                torch.sparse_coo_tensor(indices, values, (k, n)),
                requires_grad=False
            )
        else:
            # Dense JLT matrix implementation
            self.S = nn.Parameter(
                torch.randn(k, n, device=device) / math.sqrt(k),
                requires_grad=False
            )

    def forward(self, x):
        if self.sparse:
            return torch.sparse.mm(self.S, x)
        return torch.mm(self.S, x)

class ParameterSharedEmbedding(nn.Module):
    """Parameter Shared Setup (PSS) for embedding tables"""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        epsilon: float = 0.1,
        sparse_jlt: bool = True,
        device=None
    ):
        """
        Args:
            num_embeddings: Size of the vocabulary (n)
            embedding_dim: Dimension of embeddings (d)
            epsilon: Error tolerance for JLT
            sparse_jlt: Whether to use sparse JLT matrix
            device: torch device
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Calculate reduced dimension k based on theorem
        self.k = int(math.ceil(
            1 / (epsilon ** 2) * max(
                embedding_dim ** 2,
                embedding_dim * math.log(num_embeddings)
            )
        ))
        
        # Create JLT matrix S
        self.jlt = JLTMatrix(num_embeddings, self.k, sparse=sparse_jlt, device=device)
        
        # Learnable parameters M = SE (initialized randomly)
        self.M = nn.Parameter(
            torch.randn(self.k, embedding_dim, device=device) / math.sqrt(self.k)
        )

    def forward(self, indices):
        """
        Args:
            indices: Long tensor of token indices (batch_size, seq_len)
        Returns:
            embeddings: Float tensor (batch_size, seq_len, embedding_dim)
        """
        # Convert indices to one-hot vectors
        batch_size, seq_len = indices.shape
        one_hot = torch.zeros(
            batch_size * seq_len, self.num_embeddings,
            device=indices.device
        )
        one_hot.scatter_(
            1, indices.view(-1, 1), 1
        )
        
        # Compute (Se_i)^T M for all indices in batch
        Se_i = self.jlt(one_hot.t()).t()  # (batch*seq_len, k)
        embeddings = torch.mm(Se_i, self.M)  # (batch*seq_len, embedding_dim)
        
        return embeddings.view(batch_size, seq_len, self.embedding_dim)

    def from_pretrained(self, embeddings: torch.Tensor):
        """Initialize from pretrained embeddings E by computing M = SE"""
        with torch.no_grad():
            self.M.data = self.jlt(embeddings.t()).t() 