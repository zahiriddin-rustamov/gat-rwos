# interpolation.py

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import logging

from .random_walk import biased_random_walk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def interpolation_step(
    X_train, y_train, 
    minority_nodes, minority_attention, 
    strong_connections, 
    num_synthetic_samples, 
    categorical_columns,
    num_steps, p, q, num_interpolations,
    min_alpha, max_alpha, variability,
    fail_switch_limit=2000,
    random_state=None
):
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        max_seeds_needed = fail_switch_limit * 2
        all_seeds = rng.randint(0, 2**32-1, size=max_seeds_needed, dtype=np.int64)
    else:
        rng = np.random
    
    new_features = []
    minority_index_map = {i: minority_nodes[i] for i in range(len(minority_nodes))}
    reverse_index_map = {v: k for k, v in minority_index_map.items()}

    random_walk_time_total = 0
    interpolation_time_total = 0
    fail_switch_counter = 0

    loop_time = time.perf_counter()
    logger.info(f"Generating {num_synthetic_samples} synthetic samples...")
    with tqdm(total=num_synthetic_samples, desc="Generating synthetic samples", unit="sample") as pbar:
        while len(new_features) < num_synthetic_samples:
            if fail_switch_counter >= fail_switch_limit:
                logger.error(f"Failed to generate synthetic samples after {fail_switch_limit} attempts.")
                break

            # Set different seeds for each iteration to maintain controlled randomness
            if random_state is not None:
                current_walk_seed = all_seeds[fail_switch_counter * 2]
                current_interp_seed = all_seeds[fail_switch_counter * 2 + 1]
            else:
                current_walk_seed = None
                current_interp_seed = None

            pair_idx = rng.choice(len(strong_connections), size=1, replace=False)
            node1, node2 = strong_connections[pair_idx][0]
            start_node = strong_connections[pair_idx][0][0]

            random_walk_time = time.perf_counter()
            path = biased_random_walk(
                start_node, 
                np.array(minority_attention), 
                minority_nodes, 
                minority_index_map, 
                num_steps=num_steps, 
                p=p, 
                q=q,
                random_state=current_walk_seed
            )
            
            if len(path) < 2:
                logger.debug(f"Failed to generate a path for node: {start_node}")
                fail_switch_counter += 1
                continue
                
            random_walk_time = time.perf_counter() - random_walk_time

            interpolation_time = time.perf_counter()
            continuous_columns_ = []
            synthetic_features = random_walk_interpolation_with_attention(
                path,
                X_train,
                np.array(minority_attention),
                categorical_columns,
                continuous_columns_,
                reverse_index_map,
                num_interpolations=num_interpolations,
                min_alpha=min_alpha,
                max_alpha=max_alpha,
                variability=variability,
                random_state=current_interp_seed
            )
            interpolation_time = time.perf_counter() - interpolation_time

            random_walk_time_total += random_walk_time
            interpolation_time_total += interpolation_time
            
            if synthetic_features is None:
                logger.debug(f"Failed to generate synthetic samples for path: {path}")
                fail_switch_counter += 1
                continue

            for synthetic_feature in synthetic_features:
                new_features.append(synthetic_feature)
                pbar.update(1)  # Update the progress bar
                if len(new_features) >= num_synthetic_samples:
                    logger.info(f"We have reached the desired number of synthetic samples.")
                    break
    
    loop_time = time.perf_counter() - loop_time

    if len(new_features) == 0:
        logger.error(f"Failed to generate synthetic samples...")
        return None, None

    new_features = np.array(new_features)
    X_synthetic = pd.DataFrame(new_features, columns=X_train.columns)
    y_synthetic = pd.Series([1] * len(X_synthetic), name='class')
    X_train_balanced = pd.concat([X_train, X_synthetic], axis=0)
    y_train_balanced = pd.concat([y_train, y_synthetic], axis=0)

    logger.info(f"Generated {len(new_features)} synthetic samples in {loop_time:.2f} seconds.")
    return X_train_balanced, y_train_balanced

def random_walk_interpolation_with_attention(path, 
                                           features, 
                                           attention_matrix, 
                                           categorical_indices, 
                                           continuous_columns, 
                                           reverse_index_map,
                                           num_interpolations,
                                           min_alpha,
                                           max_alpha,
                                           variability,
                                           random_state=None):
    # if random_state is not None:
    #     set_seed(random_state)
        
    interpolated_features = []
    path = remove_consecutive_repeats(path)
    if len(path) < 2:
        return None

    path_length = len(path)

    # Generate all random seeds at once if using random_state
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        alpha_seeds = rng.randint(0, 2**32-1, size=num_interpolations, dtype=np.int64)
    else:
        rng = np.random
        alpha_seeds = [None] * num_interpolations

    for i in range(num_interpolations):            
        attention_weights = np.array([attention_matrix[reverse_index_map[path[j]], 
                                    reverse_index_map[path[j+1]]] 
                                    for j in range(path_length - 1)])
        adaptive_weights = attention_weights / np.sum(attention_weights)
        exp_weights = np.exp(adaptive_weights)
        interpolation_weights = exp_weights / np.sum(exp_weights)

        weighted_features = np.zeros_like(features.iloc[0])
        for j in range(path_length - 1):
            start_features = features.iloc[path[j]]
            end_features = features.iloc[path[j+1]]
            weight = interpolation_weights[j]
            interpolated_feature = guided_alpha_interpolation(
                node1_features=start_features, 
                node2_features=end_features,
                attention_weight=weight,
                categorical_indices=categorical_indices, 
                continuous_indices=continuous_columns,
                min_alpha=min_alpha, 
                max_alpha=max_alpha, 
                variability=variability,
                random_state=alpha_seeds[i]
            )
            weighted_features += interpolated_feature * interpolation_weights[j]

        interpolated_features.append(weighted_features)

    return np.array(interpolated_features)

def remove_consecutive_repeats(path):
    """Remove consecutive repeated nodes in the path."""
    unique_path = [path[0]]
    for node in path[1:]:
        if node != unique_path[-1]:
            unique_path.append(node)
    return unique_path

def guided_alpha_interpolation(
        node1_features,
        node2_features,
        attention_weight,
        categorical_indices,
        continuous_indices,
        min_alpha=0.1,
        max_alpha=0.5,
        variability=0.05,
        random_state=None
    ):
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        alpha_seed = rng.randint(0, 2**32-1, dtype=np.int64)
        categorical_seed = rng.randint(0, 2**32-1, dtype=np.int64)
        alpha_rng = np.random.RandomState(alpha_seed)
        categorical_rng = np.random.RandomState(categorical_seed)
    else:
        alpha_rng = np.random
        categorical_rng = np.random
    
    # Calculate alpha based on the attention weight
    alpha = min_alpha + (max_alpha - min_alpha) * attention_weight

    # Introduce a minor variability in alpha
    alpha += alpha_rng.uniform(-variability, variability)
    
    # Clip alpha to stay within the bounds [min_alpha, max_alpha]
    alpha = np.clip(alpha, min_alpha, max_alpha)

    if len(continuous_indices) < 3:
        # Convert indices to a set for faster lookup
        categorical_indices = set(categorical_indices)
        # Create a mask for categorical features
        categorical_mask = np.array([i in categorical_indices for i in range(len(node1_features))])
        
        # Generate all random choices for categorical features at once
        rand_choice_mask = categorical_rng.rand(len(node1_features)) > alpha
        
        # Interpolated features
        synthetic_feature = np.where(categorical_mask, 
                                    np.where(rand_choice_mask, node1_features, node2_features),
                                    alpha * node1_features + (1 - alpha) * node2_features)    
    else:
        # Calculate similarity between the two nodes
        similarity = compute_similarity(node1_features, node2_features, continuous_indices)
        synthetic_feature = np.zeros_like(node1_features)
        
        for idx in range(len(node1_features)):
            if idx in categorical_indices:
                # Compute average similarity for each category
                category_similarity = compute_average_similarity_per_category(
                    np.vstack((node1_features, node2_features)), idx, continuous_indices)
                
                # Select the best category based on the similarity
                synthetic_feature[idx] = select_best_category(similarity, category_similarity)
            else:
                # Interpolate continuous features
                synthetic_feature[idx] = alpha * node1_features[idx] + (1 - alpha) * node2_features[idx]
        
    return synthetic_feature

def select_best_category(similarity, category_similarity):
    best_category = None
    min_diff = float('inf')
    
    for category, avg_similarity in category_similarity.items():
        diff = abs(similarity - avg_similarity)
        if diff < min_diff:
            min_diff = diff
            best_category = category
            
    return best_category

def compute_average_similarity_per_category(X, categorical_index, continuous_indices):
    categories = np.unique(X[:, categorical_index])
    category_similarity = {}
    
    for category in categories:
        indices = np.where(X[:, categorical_index] == category)[0]
        if len(indices) < 2:
            category_similarity[category] = 0  # Assign 0 similarity if there's less than 2 samples
            continue
        
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarity = compute_similarity(X[indices[i]], X[indices[j]], continuous_indices)
                similarities.append(similarity)
        
        category_similarity[category] = np.mean(similarities) if similarities else 0
    
    return category_similarity

def compute_similarity(node1_features, node2_features, continuous_indices):
    node1_continuous = node1_features[continuous_indices]
    node2_continuous = node2_features[continuous_indices]

    # Ensure there are continuous features to compare
    # if node1_continuous.shape[0] == 0 or node2_continuous.shape[0] == 0:
    #     raise ValueError("One of the nodes has no continuous features to compare.")

    similarity = cosine_similarity([node1_continuous], [node2_continuous])[0, 0]
    return similarity