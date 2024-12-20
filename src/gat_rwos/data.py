# data.py

import os
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances, jaccard_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

def load_datasets(data_path: str, dataset_names: Union[str, List[str]] = None) -> List[Tuple[str, pd.DataFrame]]:
    """Load datasets from the specified path."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        
    data_files = []
    if dataset_names:
        for name in dataset_names:
            for ext in ['.csv', '.parquet']:
                if os.path.exists(os.path.join(data_path, name + ext)):
                    data_files.append((name, name + ext))
                    break
                else:
                    raise FileNotFoundError(f"Dataset {name} not found in the specified path.")
    else:
        for file in os.listdir(data_path):
            if file.endswith(('.csv', '.parquet')):
                name = os.path.splitext(file)[0]
                data_files.append((name, file))
    
    datasets = []
    for name, file in data_files:
        file_path = os.path.join(data_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:  # parquet
            df = pd.read_parquet(file_path)
        datasets.append((name, df))
        
    return datasets

def preprocess(dataset, val_size, test_size, scaler, random_state):
    """Preprocess the dataset by splitting it into training, validation, and test sets."""
    # we expect a column called class to indicate the target variable
    if 'class' not in dataset.columns:
        raise ValueError("column 'class' not found in the dataset. Please ensure that the target variable is labeled as 'class'.")
    
    # Filtering classes with more than 5 instances
    # dataset = dataset.groupby('class').filter(lambda x: len(x) > 5)
    X = dataset.drop('class', axis=1)  # Features
    y = dataset['class']  # Target variable

    # Splitting the dataset
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)

    # Resetting index
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Standardize classes to numerical format before splitting to ensure consistency across splits
    y_train = standardize_classes(y_train)
    y_val = standardize_classes(y_val)
    y_test = standardize_classes(y_test)

    # Scaling the features
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrame
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_val = pd.DataFrame(X_val, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize_classes(y):
    """Standardize classes so that the minority class is labeled as 1 and the majority class is labeled as 0."""
    # Make the minority class as 1 and majority class as 0
    minority_class = y.value_counts().idxmin()
    return y.apply(lambda x: 1 if x == minority_class else 0)

def create_graph_data(X_train, y_train, method='distance', similarity_method='euclidean', similarity_threshold=0.5, n_neighbors=5):
    """Create a graph data structure from the training data."""
    node_features = torch.tensor(X_train.values, dtype=torch.float)
    labels = torch.tensor(y_train, dtype=torch.long)
    
    if method == 'knn':
        # Compute k-nearest neighbors graph
        knn_graph = kneighbors_graph(X_train, n_neighbors=n_neighbors, metric=similarity_method, mode='connectivity', include_self=False)
        edge_index = np.array(knn_graph.nonzero())
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    else:
        # Compute pairwise distances based on the selected method
        if similarity_method in ['euclidean', 'manhattan', 'cosine']:
            distance_matrix = pairwise_distances(node_features, metric=similarity_method)
        elif similarity_method == 'jaccard':
            distance_matrix = pairwise_distances(X_train.values, metric=lambda u, v: 1 - jaccard_score(u, v))
        elif similarity_method == 'hamming':
            distance_matrix = pairwise_distances(node_features, metric='hamming')
        elif similarity_method == 'mahalanobis':
            VI = np.linalg.inv(np.cov(X_train.values.T)).T  # Inverse of covariance matrix
            distance_matrix = pairwise_distances(node_features, metric='mahalanobis', VI=VI)
        elif similarity_method == 'chebyshev':
            distance_matrix = squareform(pdist(X_train.values, metric='chebyshev'))
        elif similarity_method == 'canberra':
            distance_matrix = squareform(pdist(X_train.values, metric='canberra'))
        elif similarity_method == 'braycurtis':
            distance_matrix = squareform(pdist(X_train.values, metric='braycurtis'))
        else:
            raise ValueError(f"Unsupported similarity method: {similarity_method}")

        # Convert distances to similarity scores
        if similarity_method == "cosine":
            similarity_matrix = 1 - distance_matrix
        else:
            similarity_matrix = 1 / (1 + distance_matrix)

        # Determine the cutoff similarity value based on the percentile
        similarity_percentile = np.percentile(similarity_matrix, similarity_threshold * 100)

        # Extract edges where similarity is above the percentile
        edge_index = np.array(np.where(similarity_matrix > similarity_percentile))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Use all data points for training
    mask = torch.ones(node_features.shape[0], dtype=torch.bool)

    # Create the graph data structure
    data = Data(x=node_features, edge_index=edge_index, y=labels, mask=mask)
    return data