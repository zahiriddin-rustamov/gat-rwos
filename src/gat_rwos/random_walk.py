# random_walk.py

import numpy as np
from .utils import set_seed

def biased_random_walk(start_node, attention_matrix, minority_indices, index_map, 
                      num_steps=10, p=1, q=1, random_state=None):        
    path = [start_node]
    current_node = start_node

    # Generate all random seeds upfront if using random_state
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        step_seeds = rng.randint(0, 2**32-1, size=num_steps-1, dtype=np.int64)  # only need num_steps-1 seeds
    else:
        step_seeds = None

    for step in range(1, num_steps):
        if random_state is not None:
            step_rng = np.random.RandomState(step_seeds[step-1])  # use step-1 to start from first seed
        else:
            step_rng = np.random

        neighbors = np.intersect1d(np.nonzero(attention_matrix[current_node])[0], 
                                 np.arange(len(minority_indices)))
                                 
        if len(neighbors) == 0:
            break

        if len(path) > 1:
            prev_node = path[-2]
        else:
            prev_node = None

        probabilities = []
        for neighbor in neighbors:
            weight = attention_matrix[current_node, neighbor]

            if neighbor == prev_node:
                weight /= p
            elif neighbor not in path or path.count(neighbor) == 1:
                weight /= q

            probabilities.append(weight)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        next_node = step_rng.choice(neighbors, p=probabilities)
        path.append(next_node)
        current_node = next_node

    original_path = [index_map[node] for node in path]
    return original_path