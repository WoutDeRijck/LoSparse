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
        allow_name = ['query', 'key', 'value', 'dense', 'attention']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

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
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % self.deltaT == 0 else False
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
                        is_dict[n] = self.ipt[n] * self.exp_avg_unc[n]  # Use current importance instead of exp_avg_ipt
                    elif self.beta2 == 1.:
                        is_dict[n] = self.ipt[n]  # Use current importance directly
                    elif self.beta2 == 2.:
                        is_dict[n] = self.ipt[n] * self.exp_avg_unc[n].sqrt()
                    else:
                        is_dict[n] = self.ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

                if self.structured_method is not None and len(is_dict[n].shape) == 2:
                    is_dict[n] = self.structured_prune(is_dict[n], n)

        # Calculate statistics and threshold
        all_is = []
        for n, is_score in is_dict.items():
            all_is.append(is_score.view(-1))
        
        all_is = torch.cat(all_is)
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - threshold)))[0].item()
        
        # Mask weights whose importance lower than threshold
        total_weights = 0
        total_pruned = 0
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                num_zeros_before = (p.data == 0).sum().item()
                mask = is_dict[n] < mask_threshold
                p.data.masked_fill_(mask, 0.0)
                num_zeros_after = (p.data == 0).sum().item()
                pruned = num_zeros_after - num_zeros_before
                total_pruned += pruned
                total_weights += p.numel()
        
        return mask_threshold

    def update_and_pruning(self, model, global_step):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold

