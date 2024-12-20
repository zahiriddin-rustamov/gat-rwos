# utils.py

import os
import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim

@staticmethod
def clear_console():
    """Clear the console/terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def set_seed(seed):
    """Set the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    set_torch_seed(seed)

def set_torch_seed(seed):
    """Set the seed for PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(optimizer_name):
    """Get the optimizer based on the name."""
    optimizer_name = optimizer_name.lower()  # Standardize to lowercase
    optimizer_dict = {cls_name.lower(): getattr(optim, cls_name) 
                      for cls_name in dir(optim) 
                      if not cls_name.startswith('_') and isinstance(getattr(optim, cls_name), type)}

    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available options: {list(optimizer_dict.keys())}")
    return optimizer_dict[optimizer_name]
    
def get_scheduler(scheduler_name):
    """Get the scheduler based on the name."""
    scheduler_name = scheduler_name.lower()  # Standardize to lowercase
    scheduler_dict = {cls_name.lower(): getattr(optim.lr_scheduler, cls_name) 
                      for cls_name in dir(optim.lr_scheduler) 
                      if not cls_name.startswith('_') and isinstance(getattr(optim.lr_scheduler, cls_name), type)}

    if scheduler_name not in scheduler_dict:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available options: {list(scheduler_dict.keys())}")
    return scheduler_dict[scheduler_name]

def get_scaler(scaler_name):
    """Get the scaler based on the name."""
    if scaler_name == 'minmax':
        return MinMaxScaler()
    elif scaler_name == 'standard':
        return StandardScaler()
    else:
        raise ValueError("Unsupported scaler. Choose from 'minmax', 'standard'.")

def save_results(df, results_path, file_name):
    """Save the results to a CSV file."""
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df.to_csv(f"{results_path}/{file_name}.csv", index=False)

# def create_attention_matrix(attention_weights, aggregation_method='mean'):
#     """Create an attention matrix from the attention weights."""
#     # Extract source-target relationships
#     node_rel = attention_weights[0][0].detach().cpu().numpy()

#     # Convert tensors to numpy arrays and exclude the last head
#     attention_weights_np = [aw[1].detach().cpu().numpy() for aw in attention_weights[:-1]]

#     if aggregation_method == 'mul':
#         # For multiplication, we need to handle it differently
#         # Start with the first array and multiply with the rest
#         aggregated_attention_weights = attention_weights_np[0]
#         for aw in attention_weights_np[1:]:
#             aggregated_attention_weights = np.multiply(aggregated_attention_weights, aw)
#     else:
#         # For other methods, use numpy's built-in functions with axis parameter
#         if aggregation_method == 'mean':
#             aggregation_function = np.mean
#         elif aggregation_method == 'median':
#             aggregation_function = np.median
#         elif aggregation_method == 'max':
#             aggregation_function = np.max
#         else:
#             raise ValueError("Unsupported aggregation method. Choose from 'mean', 'median', 'max', 'mul'.")
        
#         aggregated_attention_weights = aggregation_function(attention_weights_np, axis=0)

#     # For the final aggregation across dimensions
#     if aggregation_method == 'mul':
#         combined_attn = np.array([np.prod(aggregated_attention_weights[i], axis=0) for i in range(len(aggregated_attention_weights))])
#     else:
#         combined_attn = np.array([aggregation_function(aggregated_attention_weights[i], axis=0) for i in range(len(aggregated_attention_weights))])

#     num_nodes = int(node_rel.max() + 1)
#     attention_matrix = np.zeros((num_nodes, num_nodes))

#     # Fill the attention matrix
#     for i in range(node_rel.shape[1]):
#         source = int(node_rel[0, i])
#         target = int(node_rel[1, i])
#         attention_matrix[source, target] = max(attention_matrix[source, target], combined_attn[i])

#     return attention_matrix

def create_attention_matrix(attention_weights, aggregation_method='mean'):
    """Create an attention matrix from the attention weights"""
    # Extract source-target relationships
    node_rel = attention_weights[0][0].detach().cpu().numpy()

    # Convert tensors to numpy arrays and exclude the last head
    attention_weights_np = [aw[1].detach().cpu().numpy() for aw in attention_weights[:-1]]
    
    if aggregation_method == 'entropy':
        # Calculate entropy for each attention weight matrix
        entropies = []
        for aw in attention_weights_np:
            # Calculate entropy along feature dimension
            entropy = -np.sum(aw * np.log(aw + 1e-10), axis=1)
            entropies.append(entropy)
        
        # Stack and normalize entropies
        entropies = np.stack(entropies, axis=0)  # Shape: [3, 721199]
        weights = softmax(-entropies, axis=0)    # Shape: [3, 721199]
        
        # Weighted sum of attention weights
        aggregated_attention_weights = np.zeros_like(attention_weights_np[0])  # Shape: [721199, 4]
        for i, aw in enumerate(attention_weights_np):
            aggregated_attention_weights += weights[i, :, np.newaxis] * aw

    elif aggregation_method == 'adaptive':
        # Calculate weights based on attention magnitude
        magnitudes = [np.mean(aw, axis=1) for aw in attention_weights_np]  # Shape: [721199] for each
        weights = softmax(np.stack(magnitudes, axis=0), axis=0)  # Shape: [3, 721199]
        
        # Weighted sum of attention weights
        aggregated_attention_weights = np.zeros_like(attention_weights_np[0])
        for i, aw in enumerate(attention_weights_np):
            aggregated_attention_weights += weights[i, :, np.newaxis] * aw

    elif aggregation_method == 'mul':
        # Multiplicative aggregation with stability
        aggregated_attention_weights = attention_weights_np[0]
        for aw in attention_weights_np[1:]:
            aggregated_attention_weights = np.multiply(aggregated_attention_weights + 1e-10, aw + 1e-10)
        # Normalize
        aggregated_attention_weights = aggregated_attention_weights / (aggregated_attention_weights.sum(axis=1, keepdims=True) + 1e-10)

    else:
        # Standard aggregation methods
        if aggregation_method == 'mean':
            aggregation_function = np.mean
        elif aggregation_method == 'median':
            aggregation_function = np.median
        elif aggregation_method == 'max':
            aggregation_function = np.max
        else:
            raise ValueError("Unsupported aggregation method. Choose from 'mean', 'median', 'max', 'mul', 'entropy', 'adaptive'")
        
        aggregated_attention_weights = aggregation_function(attention_weights_np, axis=0)

    # For final attention matrix creation, take mean across features
    combined_attn = np.mean(aggregated_attention_weights, axis=1)

    # Create and fill the attention matrix
    num_nodes = int(node_rel.max() + 1)
    attention_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(node_rel.shape[1]):
        source = int(node_rel[0, i])
        target = int(node_rel[1, i])
        attention_matrix[source, target] = max(attention_matrix[source, target], combined_attn[i])

    return attention_matrix

def softmax(x, axis=0):
    """Compute softmax values along specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def infer_categorical_columns(X):
    """Get the categorical columns based on the unique values."""
    categorical_columns = []
    _, n_features = X.shape
    
    for col_idx in range(n_features):
        unique_values = np.unique(X[:, col_idx])
        # Check if column values are sequential integers
        if np.array_equal(unique_values, np.arange(unique_values.min(), unique_values.max() + 1)):
            categorical_columns.append(col_idx)
    
    return categorical_columns