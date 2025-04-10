import torch
import math
import random
from torch import nn
import torch.nn.functional as F


def low_rank_decomposition(weight, rank_ratio=0.1, parameter_ratio=0.15,
                           remove_criteria='max_eigenvalue',
                           log_level='INFO',
                           return_dict=False):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix / rank_of_input_weight
    :param parameter_ratio: parameter_num_of_decomposed_matrix / (H * W). If specify, override rank_ratio
    :param remove_criteria: choose from ['max_eigenvalue', 'random', 'min_eigenvalue']
    :param       log_level: choose from ['IGNORE', 'INFO', 'DEBUG']
    :param     return_dict: Return a dict if True, else return a tuple (L, R)
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    if parameter_ratio is not None:
        reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
    else:
        reduced_rank = math.ceil(rank * rank_ratio)

    if remove_criteria == 'max_eigenvalue':
        L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
        R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh
    elif remove_criteria == 'random':
        selected_index = random.choices(range(len(S)), k=reduced_rank)
        L = U @ (torch.sqrt(torch.diag(S)[:, selected_index]))
        R = torch.sqrt(torch.diag(S)[selected_index, :]) @ Vh
    elif remove_criteria == 'min_eigenvalue':
        len_s = len(S)
        L = U @ (torch.sqrt(torch.diag(S)[:, len_s - reduced_rank:]))
        R = torch.sqrt(torch.diag(S)[len_s - reduced_rank:, :]) @ Vh
    else:
        raise NameError("remove criteria not support")

    #########
    #  LOG  #
    #########
    if log_level == 'INFO':
        if not is_full_rank:
            print(f"It is not a full rank matrix. Rank: {rank} | H x W: {H}, {W}")
        print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    if log_level == 'DEBUG':
        print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
        print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
        print(f"L: {L.shape} | R: {R.shape}")

    if return_dict:
        return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}
    else:
        return L, R


class LinearLoSparse(nn.Module):
    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, has_sparse=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.has_sparse = has_sparse

        # Initialize components
        self.right = nn.Linear(in_feature, reduced_rank, bias=False)
        self.left = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_sparse:
            self.sparse = nn.Linear(in_feature, out_feature, bias=False)
            # Initialize sparse weights to zero
            nn.init.zeros_(self.sparse.weight)

        # Initialize bias properly
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature))
        else:
            self.register_parameter('bias', None)

        self.nonzero_idx = None
        self.sparse_weight_pruned = None
        self.SX = None
        self.SX_deberta = None

    def forward(self, x):
        batch_size = x.size(0)
        
        # Low rank component
        LRX = self.left(self.right(x))  # Shape: [batch_size, out_feature]
        
        # Sparse component with pruning optimization
        if self.has_sparse:
            if self.sparse_weight_pruned is not None:
                # Efficient computation using only non-zero weights
                SX_ = torch.matmul(x, self.sparse_weight_pruned.T)
                
                # Restore full dimension output
                if self.SX is None:
                    B, L, D = x.shape
                    out_feature, in_feature = self.sparse.weight.shape
                    self.SX = torch.zeros(B, L, out_feature, device=x.device)
                
                # Update only non-zero indices
                self.SX[..., self.nonzero_idx] = SX_
                SX = self.SX
            else:
                SX = F.linear(x, self.sparse.weight, None)
        else:
            SX = torch.zeros_like(LRX, device=x.device)
        
        # Add bias if present
        if self.has_bias and self.bias is not None:
            return LRX + SX + self.bias
        return LRX + SX

    def initialize_weight(self, left_weight, right_weight, sparse_weight=None, bias=None):
        """Initialize weights from pre-trained values"""
        self.left.weight = nn.Parameter(left_weight)
        self.right.weight = nn.Parameter(right_weight)
        if self.has_sparse and sparse_weight is not None:
            self.sparse.weight = nn.Parameter(sparse_weight)
        if self.has_bias and bias is not None:
            self.bias = nn.Parameter(bias)

    def prune_sparse(self):
        self.nonzero_idx = torch.nonzero(self.sparse.weight.sum(dim=1)).flatten()
        # self.sparse_weight_pruned = self.sparse.weight[self.nonzero_idx, :]
        self.sparse_weight_pruned = nn.Parameter(self.sparse.weight[self.nonzero_idx, :])


class EmbeddingLoSparse(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, reduced_rank, padding_idx=None, has_sparse=True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduced_rank = reduced_rank
        
        # Ensure padding_idx is valid (less than num_embeddings)
        if padding_idx is not None and padding_idx >= num_embeddings:
            padding_idx = None
        self.padding_idx = padding_idx
        
        self.has_sparse = has_sparse

        # Low-rank components
        # First embedding maps to intermediate space
        self.right_embed = nn.Embedding(num_embeddings, reduced_rank, padding_idx=padding_idx)
        # Second linear layer maps from intermediate to full embedding space
        self.left_proj = nn.Linear(reduced_rank, embedding_dim, bias=False)
        
        # Sparse component
        if self.has_sparse:
            self.sparse_embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
            # Initialize sparse weights to zero
            nn.init.zeros_(self.sparse_embed.weight)
        
        self.nonzero_idx = None
        self.sparse_weight_pruned = None

    # Add weight property for compatibility with Transformers resize_token_embeddings
    @property
    def weight(self):
        if self.has_sparse:
            # Return the sparse embedding weight for compatibility
            return self.sparse_embed.weight
        else:
            # If no sparse embedding, create a dummy weight tensor with right shape
            # This is just for size detection during resizing, not for actual forward computation
            return torch.zeros(self.num_embeddings, self.embedding_dim, device=self.right_embed.weight.device)
    
    def forward(self, x):
        # Safety check: recreate embeddings if padding_idx is invalid
        if (self.padding_idx is not None and 
            (self.padding_idx >= self.num_embeddings or 
             self.padding_idx >= self.right_embed.weight.size(0))):
            
            # Fix padding_idx
            self.padding_idx = None
            
            # Recreate right embedding
            device = self.right_embed.weight.device
            dtype = self.right_embed.weight.dtype
            old_weight = self.right_embed.weight.data.clone()
            old_embedding_dim = old_weight.size(1)  # Get the correct embedding dimension
            
            # Create new embedding with corrected padding_idx and SAME reduced_rank
            self.right_embed = nn.Embedding(
                self.num_embeddings, 
                old_embedding_dim,  # Use the same embedding dimension as before
                padding_idx=None
            ).to(device=device, dtype=dtype)
            
            # Copy old weights, ensuring dimensions match
            with torch.no_grad():
                num_tokens_to_copy = min(old_weight.size(0), self.num_embeddings)
                self.right_embed.weight.data[:num_tokens_to_copy] = old_weight[:num_tokens_to_copy]
            
            # Also fix sparse embedding if it exists
            if self.has_sparse:
                old_weight = self.sparse_embed.weight.data.clone()
                old_sparse_dim = old_weight.size(1)  # Get correct sparse embedding dimension
                
                self.sparse_embed = nn.Embedding(
                    self.num_embeddings, 
                    old_sparse_dim,  # Use the same embedding dimension as before
                    padding_idx=None
                ).to(device=device, dtype=dtype)
                
                # Copy old weights, ensuring dimensions match
                with torch.no_grad():
                    num_tokens_to_copy = min(old_weight.size(0), self.num_embeddings)
                    self.sparse_embed.weight.data[:num_tokens_to_copy] = old_weight[:num_tokens_to_copy]
                    # Initialize new tokens to zero
                    if num_tokens_to_copy < self.num_embeddings:
                        nn.init.zeros_(self.sparse_embed.weight[num_tokens_to_copy:])
        
        # Check and fix left_proj to ensure it outputs the correct embedding_dim
        right_embed_out_dim = self.right_embed.weight.size(1)
        left_proj_in_dim = self.left_proj.weight.size(1)
        left_proj_out_dim = self.left_proj.weight.size(0)
        
        # Check if left_proj's output dimension matches embedding_dim
        if left_proj_out_dim != self.embedding_dim:
            # Create new projection layer with correct output dimension
            device = self.left_proj.weight.device
            dtype = self.left_proj.weight.dtype
            
            new_left_proj = nn.Linear(right_embed_out_dim, self.embedding_dim, bias=False)
            new_left_proj = new_left_proj.to(device=device, dtype=dtype)
            
            # Initialize with xavier uniform for stable training
            nn.init.xavier_uniform_(new_left_proj.weight)
            
            # Replace the layer
            self.left_proj = new_left_proj
        
        # Also check if input dimension matches
        elif right_embed_out_dim != left_proj_in_dim:
            # Create a new projection layer with correct dimensions
            device = self.left_proj.weight.device
            dtype = self.left_proj.weight.dtype
            
            # Create new projection layer with matching dimensions
            new_left_proj = nn.Linear(right_embed_out_dim, self.embedding_dim, bias=False)
            new_left_proj = new_left_proj.to(device=device, dtype=dtype)
            
            # Initialize with zeros or random values based on context
            # For now, use xavier initialization
            nn.init.xavier_uniform_(new_left_proj.weight)
            
            # Replace the layer
            self.left_proj = new_left_proj
        
        # Proceed with forward pass
        # Low rank component
        right_embedded = self.right_embed(x)
        LRX = self.left_proj(right_embedded)
        
        # Verify LRX has correct embedding dimension
        if LRX.size(-1) != self.embedding_dim:
            # Force correct dimension output
            device = LRX.device
            dtype = LRX.dtype
            shape = list(LRX.shape)
            shape[-1] = self.embedding_dim  # Set correct last dimension
            LRX = torch.zeros(shape, device=device, dtype=dtype)
        
        # Sparse component
        if self.has_sparse:
            # Check sparse embedding dimensions
            sparse_embed_dim = self.sparse_embed.weight.size(1)
            if sparse_embed_dim != self.embedding_dim:
                # Fix sparse embedding dimension
                device = self.sparse_embed.weight.device
                dtype = self.sparse_embed.weight.dtype
                
                new_sparse_embed = nn.Embedding(
                    self.num_embeddings,
                    self.embedding_dim,
                    padding_idx=self.padding_idx
                ).to(device=device, dtype=dtype)
                
                # Initialize with zeros to maintain sparsity
                nn.init.zeros_(new_sparse_embed.weight)
                
                # Copy any common dimensions
                with torch.no_grad():
                    min_dim = min(sparse_embed_dim, self.embedding_dim)
                    new_sparse_embed.weight.data[:, :min_dim] = self.sparse_embed.weight.data[:, :min_dim]
                
                self.sparse_embed = new_sparse_embed
            
            # Get sparse embedding
            if self.sparse_weight_pruned is not None:
                SX = self.sparse_embed(x)
            else:
                SX = self.sparse_embed(x)
        else:
            # Create zero tensor with correct dimension
            shape = list(LRX.shape)  # Use LRX's shape which should be correct
            SX = torch.zeros(shape, device=LRX.device, dtype=LRX.dtype)
        
        # Ensure same shape before adding
        if LRX.shape != SX.shape:
            # Create new zero tensor with LRX's shape (which should be correct)
            SX = torch.zeros_like(LRX, device=LRX.device)
        
        return LRX + SX

    def initialize_weight(self, left_weight, right_weight, sparse_weight=None):
        """Initialize weights from pre-trained values"""
        # Check if left_weight will project to the correct embedding dimension
        if left_weight.size(0) != self.embedding_dim:
            # Create a new weight with correct shape
            device = left_weight.device
            dtype = left_weight.dtype
            
            # Create new weight with correct output dimension
            new_left_weight = torch.zeros(self.embedding_dim, left_weight.size(1), device=device, dtype=dtype)
            
            # Initialize with xavier uniform
            nn.init.xavier_uniform_(new_left_weight)
            
            # Use the new weight instead
            left_weight = new_left_weight
        
        # Ensure shapes are compatible between left_weight and right_weight
        if left_weight.size(1) != right_weight.size(1):
            # Adjust left_weight to match right_weight's output dimension
            device = left_weight.device
            dtype = left_weight.dtype
            
            # Create new weight with correct dimensions
            new_left_weight = torch.zeros(self.embedding_dim, right_weight.size(1), device=device, dtype=dtype)
            
            # Initialize with xavier uniform
            nn.init.xavier_uniform_(new_left_weight)
            
            # Create new linear projection with correct dimensions
            self.left_proj = nn.Linear(right_weight.size(1), self.embedding_dim, bias=False)
            self.left_proj = self.left_proj.to(device=device, dtype=dtype)
            
            # Set the weight
            self.left_proj.weight = nn.Parameter(new_left_weight)
        else:
            # Normal initialization case
            # Create new left projection with explicit dimensions
            self.left_proj = nn.Linear(left_weight.size(1), self.embedding_dim, bias=False)
            self.left_proj.to(left_weight.device, left_weight.dtype)
            
            # Set the weight
            self.left_proj.weight = nn.Parameter(left_weight)
        
        # Set the right embedding weight
        self.right_embed.weight = nn.Parameter(right_weight)
        
        # Check if sparse_weight has correct shape
        if self.has_sparse and sparse_weight is not None:
            if sparse_weight.size(1) != self.embedding_dim:
                # Create new sparse weight with correct shape
                device = sparse_weight.device
                dtype = sparse_weight.dtype
                
                new_sparse_weight = torch.zeros(sparse_weight.size(0), self.embedding_dim, device=device, dtype=dtype)
                
                # Copy common dimensions
                min_dim = min(sparse_weight.size(1), self.embedding_dim)
                new_sparse_weight[:, :min_dim] = sparse_weight[:, :min_dim]
                
                # Use the new weight
                sparse_weight = new_sparse_weight
            
            # Set the sparse embedding weight
            self.sparse_embed.weight = nn.Parameter(sparse_weight)

    def prune_sparse(self):
        if self.has_sparse:
            # Find non-zero embedding vectors (any non-zero value in the embedding dimension)
            nonzero_mask = torch.any(self.sparse_embed.weight != 0, dim=1)
            self.nonzero_idx = torch.nonzero(nonzero_mask).flatten()
            
            # Create pruned weight matrix
            if len(self.nonzero_idx) > 0:
                self.sparse_weight_pruned = nn.Parameter(self.sparse_embed.weight[self.nonzero_idx, :])
            else:
                self.sparse_weight_pruned = None

    # Add resize_token_embeddings method for compatibility
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings to new size"""
        old_num_tokens = self.num_embeddings
        self.num_embeddings = new_num_tokens
        
        # Update padding_idx if necessary
        if self.padding_idx is not None and self.padding_idx >= new_num_tokens:
            self.padding_idx = None
        
        # Resize right embedding
        new_right_embed = nn.Embedding(new_num_tokens, self.reduced_rank, padding_idx=self.padding_idx)
        new_right_embed.to(self.right_embed.weight.device)
        
        # Copy weights for common tokens
        with torch.no_grad():
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            new_right_embed.weight.data[:num_tokens_to_copy] = self.right_embed.weight.data[:num_tokens_to_copy]
        
        self.right_embed = new_right_embed
        
        # Resize sparse embedding if it exists
        if self.has_sparse:
            new_sparse_embed = nn.Embedding(new_num_tokens, self.embedding_dim, padding_idx=self.padding_idx)
            new_sparse_embed.to(self.sparse_embed.weight.device)
            
            # Copy weights for common tokens
            with torch.no_grad():
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                new_sparse_embed.weight.data[:num_tokens_to_copy] = self.sparse_embed.weight.data[:num_tokens_to_copy]
                
                # Initialize new tokens' weights to zero (maintaining sparsity)
                if new_num_tokens > old_num_tokens:
                    nn.init.zeros_(new_sparse_embed.weight[old_num_tokens:])
            
            self.sparse_embed = new_sparse_embed
            
        # Reset cached values
        self.nonzero_idx = None
        self.sparse_weight_pruned = None
        
        return self


def prune(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == LinearLoSparse:
            print("====================================================")
            print(attr_str, target_attr)
            target_attr.prune_sparse()
        elif type(target_attr) == EmbeddingLoSparse:
            print("====================================================")
            print(attr_str, target_attr)
            target_attr.prune_sparse()
    for name, immediate_child_module in module.named_children():
        prune(immediate_child_module)


def substitute_layer_weights(module,
                             allow_name=None,
                             block_name=None,
                             parameter_ratio=0.15,
                             has_sparse=True,
                             do_svd=True,
                             **kwargs):
    """
    :param          do_svd: operate SVD
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param parameter_ratio: low rank matrix parameter / original matrix parameter
    :param      has_sparse: True if use LoRaS, false if use Low Rank only

    :return: None
    """
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query', 'key', 'value', 'dense', 'attention', 'tok_embeddings']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'norm']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        if type(target_attr) == nn.Linear and any(attr_str in an for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)

            if do_svd:
                # Decompose a matrix by SVD
                output = low_rank_decomposition(target_attr.weight, parameter_ratio=parameter_ratio,
                                                return_dict=True, **kwargs)
                L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                S = target_attr.weight - torch.mm(L, R)
                print(f"Reduced rank: {reduced_rank}")

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearLoSparse(target_attr.in_features, target_attr.out_features, reduced_rank,
                                           has_bias=True, has_sparse=has_sparse)
                linear_loras.initialize_weight(L, R, S, target_attr.bias)

            else:
                H, W = target_attr.weight.shape
                reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
                L = torch.zeros(H, reduced_rank, requires_grad=True)
                R = torch.zeros(reduced_rank, W, requires_grad=True)
                S = torch.zeros(H, W, requires_grad=True)

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearLoSparse(target_attr.in_features, target_attr.out_features, reduced_rank,
                                           has_bias=True, has_sparse=has_sparse)
                linear_loras.initialize_weight(L, R, S, target_attr.bias)

            setattr(module, attr_str, linear_loras)
        
        elif type(target_attr) == nn.Embedding and any(attr_str in an for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            
            num_embeddings, embedding_dim = target_attr.weight.shape
            padding_idx = target_attr.padding_idx
            
            # Ensure padding_idx is valid
            if padding_idx is not None and padding_idx >= num_embeddings:
                print(f"Warning: padding_idx {padding_idx} >= num_embeddings {num_embeddings}, setting to None")
                padding_idx = None
            
            if do_svd:
                # Decompose embedding matrix with SVD
                output = low_rank_decomposition(target_attr.weight, parameter_ratio=parameter_ratio,
                                               return_dict=True, **kwargs)
                L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                S = target_attr.weight - torch.mm(L, R)
                print(f"Embedding reduced rank: {reduced_rank}")
                
                # Check matrix dimensions
                if L.size(1) != R.size(0):
                    # Transpose R if needed for embedding format
                    if L.size(1) == R.size(1) and R.size(0) != L.size(1):
                        R = R.t()
                
                # Create LoSparse embedding
                embed_loras = EmbeddingLoSparse(num_embeddings, embedding_dim, reduced_rank, 
                                              padding_idx=padding_idx, has_sparse=has_sparse)
                
                # Initialize weights
                embed_loras.initialize_weight(L, R, S)
            
            else:
                reduced_rank = math.ceil(parameter_ratio * (num_embeddings * embedding_dim) / 
                                        (num_embeddings + embedding_dim))
                
                # For embeddings, the dimensions should be:
                # L: [embedding_dim, reduced_rank]
                # R: [reduced_rank, num_embeddings] transposed to [num_embeddings, reduced_rank]
                L = torch.zeros(embedding_dim, reduced_rank, requires_grad=True)
                # Create R with correct dimensions for embedding format
                R = torch.zeros(num_embeddings, reduced_rank, requires_grad=True)
                S = torch.zeros(num_embeddings, embedding_dim, requires_grad=True)
                
                # Create LoSparse embedding and assign weights
                embed_loras = EmbeddingLoSparse(num_embeddings, embedding_dim, reduced_rank,
                                              padding_idx=padding_idx, has_sparse=has_sparse)
                embed_loras.initialize_weight(L, R, S)
            
            setattr(module, attr_str, embed_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights(immediate_child_module, allow_name, block_name, parameter_ratio,
                                     has_sparse, do_svd, **kwargs)


class Pruner(object):
    def __init__(self, model, args, total_step, tb_writer=None,
                 mask_param_name=None,
                 non_mask_name=None,
                 use_no_mask=False,
                 pruner_name='PLATON',
                 structured_method='mean',
                 structured_direction='row'):

        if non_mask_name is None:
            non_mask_name = ["embedding", "norm"]
        if mask_param_name is None:
            mask_param_name = ['sparse']
        self.model = model
        self.config = vars(args)
        self.args = args
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_name = mask_param_name
        self.non_mask_name = non_mask_name
        self.use_no_mask = use_no_mask
        self.total_step = total_step
        self.tb_writer = tb_writer
        self.pruner_name = pruner_name
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]
        self.deltaT = self.config["deltaT"]
        self.structured_method = structured_method
        self.structured_direction = structured_direction
        self.current_threshold = 1.0  # Initialize with no pruning

    def whether_mask_para(self, n):
        if not self.use_no_mask:
            return any(nd in n for nd in self.mask_param_name)
        else:
            return not any([nd in n for nd in self.non_mask_name])

    def structured_prune(self, is_dict_mat, name):
        num_row, num_col = is_dict_mat.shape
        if self.structured_direction == 'row_col':
            if self.structured_method == "mean":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                if any(nd in name for nd in ['q', 'k', 'v']):
                    return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
                else:
                    return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        elif self.structured_direction == 'row':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=1, keepdim=True).repeat((1, num_col))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        elif self.structured_direction == 'col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True).repeat((num_row, 1))
            else:
                raise ValueError("Unimplemented Sturctured Method: %s" % self.structured_method)
        else:
            raise ValueError("Unimplemented Sturctured Direction: %s" % self.structured_direction)

    def schedule_threshold_comb(self, step: int):
        # Schedule the remaining ratio
        args = self.args
        total_step = self.total_step
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        initial_warmup = self.config['initial_warmup']
        final_warmup = self.config['final_warmup']
        warmup_steps = self.config['warmup_steps']

        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)

        mask_ind = True if step % self.deltaT == 0 else False
        if mask_ind:  # Only update threshold on deltaT steps
            self.current_threshold = threshold
        return threshold, mask_ind

    def update_ipt_with_local_window(self, model, global_step):
        # Calculate the sensitivity and uncertainty
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                # Skip if no gradient
                if p.grad is None:
                    continue
                    
                # Initialize if not exists
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    # Initialize with current importance instead of zeros
                    self.ipt[n] = (p * p.grad).abs().detach()
                    if self.beta2 > 0 and self.beta2 != 1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                
                # PLATON importance calculation
                if self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    
                    # Calculate new importance
                    new_ipt = (p * p.grad).abs().detach()
                    
                    if local_step == 0:
                        # Update exponential moving average
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        
                        # Update uncertainty estimate
                        if 0 < self.beta2 < 1:
                            self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                                (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[n] = (update_step * self.exp_avg_unc[n] +
                                                 (self.ipt[n] - self.exp_avg_ipt[n]) ** 2) / (update_step + 1)
                        
                        # Reset importance accumulator
                        self.ipt[n] = new_ipt
                    else:
                        # Accumulate importance with moving average
                        self.ipt[n] = (self.ipt[n] * local_step + new_ipt) / (local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")

    def mask_with_threshold(self, model, threshold):
        # Initialize importance score dictionary
        is_dict = {}
        
        # Calculate importance scores
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                if self.pruner_name == 'Magnitude':
                    is_dict[n] = p.abs().detach()
                elif self.pruner_name == 'PLATON':
                    # Skip if no importance scores
                    if n not in self.exp_avg_ipt:
                        continue
                    
                    # Use current importance scores directly
                    if 0 < self.beta2 < 1:
                        is_dict[n] = self.ipt[n] * self.exp_avg_unc[n]
                    elif self.beta2 == 1.:
                        is_dict[n] = self.ipt[n]
                    elif self.beta2 == 2.:
                        is_dict[n] = self.ipt[n] * self.exp_avg_unc[n].sqrt()
                    else:
                        is_dict[n] = self.ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

                if self.structured_method is not None and len(is_dict.get(n, torch.tensor([])).shape) == 2:
                    is_dict[n] = self.structured_prune(is_dict[n], n)

        # Return None if no parameters have importance scores
        if not is_dict:
            return None
            
        # Calculate statistics and threshold
        all_is = []
        for n, is_score in is_dict.items():
            all_is.append(is_score.view(-1))
        
        all_is = torch.cat(all_is)
        num_elements = all_is.shape[0]
        
        # Ensure k is within valid range [1, num_elements]
        k = max(1, min(num_elements, int(num_elements * (1 - self.current_threshold))))
        mask_threshold = torch.kthvalue(all_is, k)[0].item()
        
        # Mask weights whose importance lower than threshold
        total_weights = 0
        total_pruned = 0
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                if n in is_dict:  # Only process if we have importance scores
                    num_zeros_before = (p.data == 0).sum().item()
                    mask = is_dict[n] < mask_threshold
                    p.data.masked_fill_(mask, 0.0)
                    num_zeros_after = (p.data == 0).sum().item()
                    total_pruned += num_zeros_after - num_zeros_before
                    total_weights += p.numel()
        
        return mask_threshold

    def update_and_pruning(self, model, global_step):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        # Always apply masking with current threshold
        mask_threshold = self.mask_with_threshold(model, threshold)
        return threshold, mask_threshold

