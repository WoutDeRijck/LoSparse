import torch
import math
import random
from torch import nn
import torch.nn.functional as F


def low_rank_decomposition(weight, rank_ratio=0.1, parameter_ratio=0.15,
                           remove_criteria='max_eigenvalue',
                           log_level='INFO',
                           return_dict=False,
                           device=None):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix / rank_of_input_weight
    :param parameter_ratio: parameter_num_of_decomposed_matrix / (H * W). If specify, override rank_ratio
    :param remove_criteria: choose from ['max_eigenvalue', 'random', 'min_eigenvalue']
    :param       log_level: choose from ['IGNORE', 'INFO', 'DEBUG']
    :param     return_dict: Return a dict if True, else return a tuple (L, R)
    :param          device: Device to perform computation on (default: same as weight)
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use the same device as weight if not specified
    if device is None:
        device = weight.device
    
    # Move weight to specified device for computation
    weight_device = weight.to(device)

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight_device, full_matrices=False)
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

    # Move results back to original device if needed
    if L.device != weight.device:
        L = L.to(weight.device)
        R = R.to(weight.device)

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
        
        # Register forward pre-hook for optimized computation
        self.register_forward_pre_hook(self._update_pruned_weight)
        
        # Use JIT compilation for faster inference when possible
        self._use_jit = False
        try:
            # Check if JIT is available
            if torch.__version__ >= '1.7.0':
                self._use_jit = True
        except:
            pass

    def _update_pruned_weight(self, module, input):
        # This pre-hook updates the pruned weight representation if needed
        if self.has_sparse and self.nonzero_idx is None and (self.sparse.weight != 0).any():
            self.prune_sparse()

    def forward(self, x):
        # Low rank component - this is always needed
        # Use sequential computation for better memory efficiency
        right_output = self.right(x)
        LRX = self.left(right_output)
        
        # Sparse component with pruning optimization
        if self.has_sparse:
            # Fast path: if sparse weights are all zero or none are pruned yet
            if hasattr(self.sparse.weight, 'data') and (self.sparse.weight.data == 0).all():
                # All zeros - skip computation
                if self.has_bias and self.bias is not None:
                    return LRX + self.bias
                return LRX
            
            # Use efficient linear operation
            SX = F.linear(x, self.sparse.weight, None)
        else:
            # No sparse component
            SX = 0
        
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
        """Identify and store non-zero indices for optimized computation"""
        if not self.has_sparse:
            return
            
        # Find indices of rows with any non-zero values
        row_mask = torch.any(self.sparse.weight != 0, dim=1)
        self.nonzero_idx = torch.nonzero(row_mask).flatten()
        
        # Store the pruned weight - only keep non-zero rows
        if len(self.nonzero_idx) > 0:
            self.sparse_weight_pruned = self.sparse.weight[self.nonzero_idx]
        else:
            # All weights are zero
            self.sparse_weight_pruned = None
        
        # Clear cached tensors to free memory
        self.SX = None
        self.SX_deberta = None


def prune(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == LinearLoSparse:
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
                             device=None,
                             batch_size=10,
                             verbose=False,
                             **kwargs):
    """
    :param          do_svd: operate SVD
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param parameter_ratio: low rank matrix parameter / original matrix parameter
    :param      has_sparse: True if use LoRaS, false if use Low Rank only
    :param          device: Device to use for SVD computation (default: GPU if available, else CPU)
    :param      batch_size: Number of layers to process in parallel
    :param        verbose: Whether to print progress information
    :return: None
    """
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query', 'key', 'value', 'dense', 'attention']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
        
    # Determine computation device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First pass: identify eligible linear layers for replacement
    eligible_layers = []
    
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Linear and any(an in attr_str for an in allow_name):
            eligible_layers.append((attr_str, target_attr))
    
    # Process eligible layers in batches
    for i in range(0, len(eligible_layers), batch_size):
        batch = eligible_layers[i:i+batch_size]
        
        # Process batch
        for j, (attr_str, target_attr) in enumerate(batch):
            print("====================================================")
            print(attr_str, target_attr)
                
            if do_svd:
                # Decompose a matrix by SVD (compute on specified device)
                output = low_rank_decomposition(target_attr.weight, parameter_ratio=parameter_ratio,
                                            device=device, return_dict=True, **kwargs)
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
                L = torch.zeros(H, reduced_rank, requires_grad=True, device=target_attr.weight.device)
                R = torch.zeros(reduced_rank, W, requires_grad=True, device=target_attr.weight.device)
                S = torch.zeros(H, W, requires_grad=True, device=target_attr.weight.device)

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearLoSparse(target_attr.in_features, target_attr.out_features, reduced_rank,
                                         has_bias=True, has_sparse=has_sparse)

                linear_loras.initialize_weight(L, R, S, target_attr.bias)

            setattr(module, attr_str, linear_loras)
            
    # Process child modules
    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights(immediate_child_module, allow_name, block_name, parameter_ratio,
                                 has_sparse, do_svd, device, batch_size, verbose, **kwargs)


class Pruner(object):
    def __init__(self, model, args, total_step, tb_writer=None,
                 mask_param_name=None,
                 non_mask_name=None,
                 use_no_mask=False,
                 pruner_name='PLATON',
                 structured_method='mean',
                 structured_direction='row',
                 device=None):

        if non_mask_name is None:
            non_mask_name = ["embedding", "norm"]
        if mask_param_name is None:
            mask_param_name = ['sparse']
            
        # Determine device for computation
        if device is None:
            device = next(model.parameters()).device
            
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
        self.device = device
        
        # Pre-identify prunable parameters for faster access
        self.prunable_params = {n: p for n, p in model.named_parameters() 
                               if self.whether_mask_para(n)}
        
        # Initialize importance maps on appropriate device
        for n, p in self.prunable_params.items():
            if n not in self.exp_avg_ipt:
                self.exp_avg_ipt[n] = torch.zeros_like(p, device=self.device)
                if self.beta2 > 0 and self.beta2 != 1:
                    self.exp_avg_unc[n] = torch.zeros_like(p, device=self.device)

    def whether_mask_para(self, n):
        if not self.use_no_mask:
            return any(nd in n for nd in self.mask_param_name)
        else:
            return not any([nd in n for nd in self.non_mask_name])

    def structured_prune(self, is_dict_mat, name):
        num_row, num_col = is_dict_mat.shape
        
        # Use torch operations for efficiency
        if self.structured_direction == 'row_col':
            if any(nd in name for nd in ['q', 'k', 'v']):
                # Row direction
                if self.structured_method == "mean":
                    return torch.mean(is_dict_mat, dim=1, keepdim=True).expand(-1, num_col)
                elif self.structured_method == "sum":
                    return torch.sum(is_dict_mat, dim=1, keepdim=True).expand(-1, num_col)
                elif self.structured_method == "max":
                    return torch.max(is_dict_mat, dim=1, keepdim=True)[0].expand(-1, num_col)
                elif self.structured_method == "min":
                    return torch.min(is_dict_mat, dim=1, keepdim=True)[0].expand(-1, num_col)
            else:
                # Column direction
                if self.structured_method == "mean":
                    return torch.mean(is_dict_mat, dim=0, keepdim=True).expand(num_row, -1)
                elif self.structured_method == "sum":
                    return torch.sum(is_dict_mat, dim=0, keepdim=True).expand(num_row, -1)
                elif self.structured_method == "max":
                    return torch.max(is_dict_mat, dim=0, keepdim=True)[0].expand(num_row, -1)
                elif self.structured_method == "min":
                    return torch.min(is_dict_mat, dim=0, keepdim=True)[0].expand(num_row, -1)
        elif self.structured_direction == 'row':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=1, keepdim=True).expand(-1, num_col)
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=1, keepdim=True).expand(-1, num_col)
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=1, keepdim=True)[0].expand(-1, num_col)
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=1, keepdim=True)[0].expand(-1, num_col)
        elif self.structured_direction == 'col':
            if self.structured_method == "mean":
                return torch.mean(is_dict_mat, dim=0, keepdim=True).expand(num_row, -1)
            elif self.structured_method == "sum":
                return torch.sum(is_dict_mat, dim=0, keepdim=True).expand(num_row, -1)
            elif self.structured_method == "max":
                return torch.max(is_dict_mat, dim=0, keepdim=True)[0].expand(num_row, -1)
            elif self.structured_method == "min":
                return torch.min(is_dict_mat, dim=0, keepdim=True)[0].expand(num_row, -1)
        
        raise ValueError(f"Unsupported: {self.structured_method} with {self.structured_direction}")

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
        local_step = global_step % self.deltaT
        update_step = global_step // self.deltaT
        
        # Process all prunable parameters
        for n, p in self.prunable_params.items():
            # Skip if no gradient
            if p.grad is None:
                continue
                
            # Initialize if not exists (should be already done in __init__)
            if n not in self.ipt:
                self.ipt[n] = (p * p.grad).abs().detach().to(self.device)
            
            # PLATON importance calculation
            if self.pruner_name == 'PLATON':
                # Calculate new importance
                new_ipt = (p * p.grad).abs().detach().to(self.device)
                
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
        # Calculate importance scores more efficiently
        is_dict = {}
        
        # Pre-compute all importance scores at once
        for n, p in self.prunable_params.items():
            if self.pruner_name == 'Magnitude':
                is_dict[n] = p.abs().detach().to(self.device)
            elif self.pruner_name == 'PLATON':
                # Skip if no importance scores
                if n not in self.ipt:
                    continue
                
                # Select appropriate importance metric
                if 0 < self.beta2 < 1:
                    is_dict[n] = (self.ipt[n] * self.exp_avg_unc[n]).to(self.device)
                elif self.beta2 == 1.:
                    is_dict[n] = self.ipt[n].to(self.device)
                elif self.beta2 == 2.:
                    is_dict[n] = (self.ipt[n] * self.exp_avg_unc[n].sqrt()).to(self.device)
                else:
                    is_dict[n] = (self.ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()).to(self.device)

                # Apply structured pruning if needed
                if self.structured_method is not None and len(is_dict[n].shape) == 2:
                    is_dict[n] = self.structured_prune(is_dict[n], n)

        # Calculate threshold - use different methods based on data size
        num_elements = sum(s.numel() for s in is_dict.values())
        k = max(1, min(num_elements, int(num_elements * (1 - self.current_threshold))))
        
        # For very large tensors, use approximate method to avoid OOM
        if num_elements > 10000000:  # 10M elements
            # Sample a subset for estimation
            sample_size = min(1000000, num_elements // 10)  # 1M elements or 10% of data
            sampled_values = []
            # Sample from each tensor proportionally to its size
            for is_score in is_dict.values():
                n_elements = is_score.numel()
                if n_elements == 0:
                    continue
                sample_ratio = sample_size * n_elements / num_elements
                indices = torch.randint(0, n_elements, (int(sample_ratio),), device=self.device)
                sampled_values.append(is_score.view(-1)[indices])
            
            if sampled_values:
                # Combine samples and estimate threshold
                sampled_tensor = torch.cat(sampled_values)
                approximate_quantile = float(k) / num_elements
                mask_threshold = torch.quantile(sampled_tensor, approximate_quantile).item()
            else:
                # Fallback if no samples
                mask_threshold = 0.0
        else:
            # For smaller tensors, collect all values and use kthvalue
            all_is = torch.empty(num_elements, device=self.device)
            idx = 0
            for is_score in is_dict.values():
                size = is_score.numel()
                if size > 0:
                    all_is[idx:idx+size] = is_score.view(-1)
                    idx += size
            
            if idx > 0:
                mask_threshold = torch.kthvalue(all_is[:idx], k)[0].item()
            else:
                mask_threshold = 0.0
        
        # Apply masks in a batched manner
        total_weights = 0
        total_pruned = 0
        
        for n, p in self.prunable_params.items():
            if n in is_dict:  # Only process if we have importance scores
                # Use memory-efficient masking
                mask = is_dict[n] < mask_threshold
                p.data.masked_fill_(mask.to(p.device), 0.0)
                
                # Add to statistics
                num_zeros = mask.sum().item()
                total_pruned += num_zeros
                total_weights += p.numel()
        
        return mask_threshold

    def update_and_pruning(self, model, global_step):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        
        # Get the pruning threshold
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        
        # Apply masking if appropriate
        mask_threshold = None
        if mask_ind:
            mask_threshold = self.mask_with_threshold(model, threshold)
            
            # Apply pruning to optimize forward pass
            self._apply_sparse_pruning(model)
        
        return threshold, mask_threshold
        
    def _apply_sparse_pruning(self, model):
        """Apply pruning optimization to all LinearLoSparse layers"""
        for module in model.modules():
            if isinstance(module, LinearLoSparse) and module.has_sparse:
                module.prune_sparse()

