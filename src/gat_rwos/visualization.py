# visualizations.py

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_original_vs_oversampled(X_train, y_train, X_train_balanced, y_train_balanced, figsize=(15, 5)):
    """
    Creates a side-by-side PCA visualization comparing original and oversampled datasets.
    Original points are shown as circles, synthetic minority points as thick plus markers.
    
    Parameters:
    -----------
    X_train : DataFrame or array-like
        Original features
    y_train : Series or array-like
        Original labels
    X_train_balanced : DataFrame or array-like
        Oversampled features (including both original and synthetic samples)
    y_train_balanced : Series or array-like
        Oversampled labels
    figsize : tuple, optional
        Figure size for the plot (default: (15, 5))
    """
    # Convert inputs to numpy arrays if they aren't already
    X_train = np.array(X_train)
    X_train_balanced = np.array(X_train_balanced)
    y_train = np.array(y_train)
    y_train_balanced = np.array(y_train_balanced)
    
    # Combine data for PCA fitting
    X_combined = np.vstack([X_train, X_train_balanced])
    
    # Fit PCA on the combined dataset
    pca = PCA(n_components=2)
    X_combined_pca = pca.fit_transform(X_combined)
    
    # Split back into original and oversampled
    X_original_pca = X_combined_pca[:len(X_train)]
    X_oversampled_pca = X_combined_pca[len(X_train):]
    
    # Color mapping
    colors = {'majority': '#b5c6e0', 'minority': '#cc2936'}
    
    # Convert labels to majority/minority
    y_train_str = np.where(y_train == 1, 'minority', 'majority')
    y_balanced_str = np.where(y_train_balanced == 1, 'minority', 'majority')
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#f0f0f0')  # Set grey background
    
    # Plot original data
    for label in ['majority', 'minority']:
        mask = y_train_str == label
        axes[0].scatter(X_original_pca[mask, 0], X_original_pca[mask, 1],
                       c=colors[label], marker='o', s=50, label=label)
    axes[0].set_title('Original Data', fontsize=18)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Plot oversampled data
    # First plot all original points as circles
    original_indices = list(range(len(X_train)))  # Indices of original samples
    synthetic_indices = list(range(len(X_train), len(X_train_balanced)))  # Indices of synthetic samples
    
    # Plot original majority points
    mask_maj = y_balanced_str[original_indices] == 'majority'
    axes[1].scatter(X_oversampled_pca[original_indices][mask_maj, 0],
                   X_oversampled_pca[original_indices][mask_maj, 1],
                   c=colors['majority'], marker='o', s=50, label='majority')
    
    # Plot original minority points
    mask_min = y_balanced_str[original_indices] == 'minority'
    axes[1].scatter(X_oversampled_pca[original_indices][mask_min, 0],
                   X_oversampled_pca[original_indices][mask_min, 1],
                   c=colors['minority'], marker='o', s=50, label='minority (original)')
    
    # Plot synthetic minority points with thick plus
    mask_syn = y_balanced_str[synthetic_indices] == 'minority'
    axes[1].scatter(X_oversampled_pca[synthetic_indices][mask_syn, 0],
                   X_oversampled_pca[synthetic_indices][mask_syn, 1],
                   c=colors['minority'], marker='+', s=100, linewidths=2,
                   label='minority (synthetic)')
    
    axes[1].set_title('GAT-RWOS', fontsize=18)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(fontsize=14)
    
    plt.tight_layout()
    return fig, axes
