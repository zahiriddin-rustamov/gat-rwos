# training.py

import os
import time
import logging

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm import tqdm

from .data import create_graph_data
from .models import initialize_gat_model
from .utils import (
    get_optimizer,
    get_scheduler,
    create_attention_matrix,
    infer_categorical_columns
)
from .interpolation import interpolation_step
from .evaluation import test_classification_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    graph_method: str
    similarity_method: str
    similarity_threshold: float
    n_neighbors: int
    optimizer_name: str
    optimizer_parms: dict
    scheduler_name: str
    scheduler_parms: dict
    hid: int
    in_head: int
    out_head: int
    dropout_rate: float
    num_hidden_layers: int
    return_attention: bool
    num_epochs: int
    patience: int
    device: str
    aggregation_method: str
    attention_threshold: float
    num_steps: int
    p: float
    q: float
    num_interpolations: int
    tune: bool
    min_alpha: float
    max_alpha: float
    variability: float
    n_trials_attention: int
    n_trials_interpolation: int
    random_state: int

class InterpolationOptimizer:
    """Handles the optimization of interpolation parameters."""
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int,
        random_state: Optional[int] = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials_interpolation = n_trials
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.random_state = random_state or 1

    def interpolation_objective(self, trial: optuna.Trial, minority_nodes: np.ndarray,
                              minority_attention: np.ndarray, strong_connections: np.ndarray,
                              num_synthetic_samples: int, categorical_columns: list) -> float:
        """Optimization objective for interpolation parameters."""
        params = {
            "num_steps": trial.suggest_int("num_steps", 1, 15),
            "p": trial.suggest_float("p", 0.1, 2),
            "q": trial.suggest_float("q", 0.1, 2),
            "num_interpolations": trial.suggest_int("num_interpolations", 1, 10),
            "min_alpha": trial.suggest_float("min_alpha", 0.1, 0.4),
            "max_alpha": trial.suggest_float("max_alpha", 0.5, 1),
            "variability": trial.suggest_float("variability", 0.01, 0.5),
            "random_state": self.random_state
        }

        X_train_balanced, y_train_balanced = interpolation_step(
            self.X_train, self.y_train,
            minority_nodes, minority_attention,
            strong_connections,
            num_synthetic_samples,
            categorical_columns,
            **params
        )

        if X_train_balanced is None:
            return 0

        results = test_classification_models(
            X_train_balanced, y_train_balanced,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            RANDOM_STATE=self.random_state
        )

        return results['Validation G-Mean'].max()

class AttentionOptimizer:
    """Handles the optimization of attention parameters."""
    
    def __init__(self, interpolation_optimizer: InterpolationOptimizer):
        self.interpolation_optimizer = interpolation_optimizer
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

    def attention_objective(self, trial: optuna.Trial, attention_weights: torch.Tensor,
                          y_train: np.ndarray, X_train: np.ndarray) -> float:
        """Optimization objective for attention parameters."""
        attention_threshold = trial.suggest_float("attention_threshold", 0.3, 0.999)
        aggregation_method = trial.suggest_categorical("aggregation_method", ["mean", "median", "max", "mul"])

        # Process attention matrix
        combined_attention_matrix = create_attention_matrix(attention_weights, aggregation_method=aggregation_method)
        np.fill_diagonal(combined_attention_matrix, 0)
        minority_nodes = np.where(y_train == 1)[0]
        minority_attention = combined_attention_matrix[np.ix_(minority_nodes, minority_nodes)]
        
        attention_percentile = np.percentile(minority_attention, attention_threshold * 100)
        strong_connections = np.argwhere(minority_attention > attention_percentile)

        if len(strong_connections) < 5:
            logger.warning("Not enough strong connections found.")
            return 0

        # Calculate synthetic samples
        majority_count = np.sum(y_train == 0)
        minority_count = np.sum(y_train == 1)
        num_synthetic_samples = abs(majority_count - minority_count)
        categorical_columns = infer_categorical_columns(X_train.to_numpy())

        # Create and optimize interpolation study
        interpolation_study = optuna.create_study(direction="maximize", study_name="Interpolation Study")
        objective = lambda trial: self.interpolation_optimizer.interpolation_objective(
            trial, minority_nodes, minority_attention, strong_connections,
            num_synthetic_samples, categorical_columns
        )
        
        interpolation_study.optimize(objective, n_trials=self.interpolation_optimizer.n_trials_interpolation)
        self.interpolation_optimizer.best_params = interpolation_study.best_params
        return interpolation_study.best_value

def update_scheduler(
    scheduler: Optional[Any],
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    scheduler_name: str
) -> None:
    """Update the learning rate scheduler."""
    if scheduler is None:
        return
    
    before_lr = optimizer.param_groups[0]['lr']
    if scheduler_name.lower() == 'reducelronplateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()
    after_lr = optimizer.param_groups[0]['lr']
    if before_lr != after_lr:
        logger.info(f"Learning rate adjusted from {before_lr} => {after_lr}")

def train_epoch(
    model: torch.nn.Module,
    train_data: Any,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(train_data)
    confidences = out.exp()
    _, predictions = torch.max(out.data, 1)
    confidence_scores = confidences[range(len(train_data.y[train_data.mask])), predictions[train_data.mask]]
    
    # Focal loss
    focal_loss = F.nll_loss(out[train_data.mask], train_data.y[train_data.mask], 
                           weight=class_weights, reduction='none')
    focal_loss = (focal_loss * (1 - confidence_scores) ** 2).mean()
    
    # Minority class-specific regularization
    minority_mask = train_data.y[train_data.mask] == 0
    minority_loss = F.nll_loss(out[train_data.mask][minority_mask], 
                              train_data.y[train_data.mask][minority_mask])
    
    # Total loss
    loss = focal_loss + 0.1 * minority_loss
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validate_epoch(
    model: torch.nn.Module,
    val_data: Any,
    class_weights: torch.Tensor,
    device: str
) -> Tuple[float, float]:
    """Validate the model for one epoch."""
    model.eval()
    val_mask = val_data.mask
    
    with torch.no_grad():
        val_out = model(val_data)
        val_confidences = val_out.exp()
        _, val_predictions = torch.max(val_out.data, 1)
        val_confidence_scores = val_confidences[range(len(val_data.y[val_mask])), 
                                              val_predictions[val_mask]]
        
        # Focal loss
        val_focal_loss = F.nll_loss(val_out[val_mask], val_data.y[val_mask], 
                                   weight=class_weights, reduction='none')
        val_focal_loss = (val_focal_loss * (1 - val_confidence_scores) ** 2).mean()
        
        # Minority loss
        val_minority_mask = val_data.y[val_mask] == 0
        val_minority_loss = F.nll_loss(val_out[val_mask][val_minority_mask], 
                                     val_data.y[val_mask][val_minority_mask])
        
        val_loss = val_focal_loss + 0.1 * val_minority_loss
        
        # Calculate F1 score
        val_true = val_data.y[val_mask].cpu().numpy()
        val_pred = val_predictions[val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_pred, average='macro')
        
    return val_loss.item(), val_f1

def generate_balanced_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    attention_weights: torch.Tensor, 
    config: TrainingConfig
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Generate a balanced dataset using attention-based interpolation."""
    combined_attention_matrix = create_attention_matrix(attention_weights, 
                                                      aggregation_method=config.aggregation_method)
    np.fill_diagonal(combined_attention_matrix, 0)
    minority_nodes = np.where(y_train == 1)[0]
    
    # Process minority attention
    minority_attention = combined_attention_matrix[np.ix_(minority_nodes, minority_nodes)]
    attention_percentile = np.percentile(minority_attention, config.attention_threshold * 100)
    strong_connections = np.argwhere(minority_attention > attention_percentile)
    logger.info(f"Found {len(strong_connections)} strong connections")
    
    if len(strong_connections) < 5:
        logger.warning("No strong connections found. Please adjust the attention threshold.")
        return None, None
    
    # Calculate number of synthetic samples needed
    majority_count = np.sum(y_train == 0)
    minority_count = np.sum(y_train == 1)
    num_synthetic_samples = abs(majority_count - minority_count)
    
    categorical_columns = infer_categorical_columns(X_train.to_numpy())
    
    X_train_balanced, y_train_balanced = interpolation_step(
        X_train, y_train,
        minority_nodes, minority_attention,
        strong_connections,
        num_synthetic_samples,
        categorical_columns,
        config.num_steps,
        config.p,
        config.q,
        config.num_interpolations,
        config.min_alpha,
        config.max_alpha,
        config.variability,
        random_state=config.random_state
    )

    return X_train_balanced, y_train_balanced

def train_model_focal_loss(
    train_data: Any,
    val_data: Any,
    config: TrainingConfig,
    model: torch.nn.Module
) -> Tuple[Any, Any, float, Optional[torch.Tensor]]:
    """Train the model using focal loss."""
    if model is None:
        raise ValueError("Model must be provided")

    val_mask = val_data.mask
    optimizer = get_optimizer(config.optimizer_name)(model.parameters(), **config.optimizer_parms)
    scheduler = get_scheduler(config.scheduler_name)(optimizer, **config.scheduler_parms)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_data.y.cpu().numpy()),
        y=train_data.y.cpu().numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model = None
    start_time = time.perf_counter()

    for epoch in tqdm(range(config.num_epochs), desc="Training Progress"):
        loss = train_epoch(model, train_data, optimizer, class_weights, config.device)
        val_loss, val_f1 = validate_epoch(model, val_data, class_weights, config.device)
        
        # Update scheduler
        if scheduler:
            update_scheduler(scheduler, optimizer, val_loss, config.scheduler_name)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model = model
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    training_time = time.perf_counter() - start_time
    
    attention_weights = None
    if config.return_attention and best_model is not None:
        _, attention_weights = best_model(train_data, return_attention=True)
    
    return best_model, model(train_data), training_time, attention_weights

def trainer(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: TrainingConfig
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
    """Main training function."""
    # Create graph data
    train_data = create_graph_data(X_train, y_train, config.graph_method,
                                 config.similarity_method, config.similarity_threshold,
                                 config.n_neighbors)
    val_data = create_graph_data(X_val, y_val, config.graph_method,
                               config.similarity_method, config.similarity_threshold,
                               config.n_neighbors)

    # Initialize and move model to device
    try:
        model = initialize_gat_model(
            num_features=X_train.shape[1],
            num_classes=y_train.nunique(),
            hid=config.hid,
            in_head=config.in_head,
            out_head=config.out_head,
            dropout_rate=config.dropout_rate,
            num_hidden_layers=config.num_hidden_layers
        ).to(config.device)
    except Exception as e:
        logger.error("Failed to initialize the model: %s", e)
        raise

    train_data = train_data.to(config.device)
    val_data = val_data.to(config.device)

    # Train model
    best_model, out, training_time, attention_weights = train_model_focal_loss(
        train_data, val_data, config, model
    )

    if config.tune:
        # Initialize optimizers
        interpolation_optimizer = InterpolationOptimizer(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_trials=config.n_trials_interpolation,
            random_state=config.random_state
        )
        attention_optimizer = AttentionOptimizer(interpolation_optimizer)
        
        # Create and run study
        study = optuna.create_study(direction="maximize", study_name="Attention Study")
        objective = lambda trial: attention_optimizer.attention_objective(
            trial, attention_weights, y_train, X_train
        )
        study.optimize(objective, n_trials=config.n_trials_attention)

        if study.trials and any(trial.value > 0 for trial in study.trials):
            print(f"Best attention study value: {study.best_value}")
            # Update config with best parameters
            config.aggregation_method = study.best_params['aggregation_method']
            config.attention_threshold = study.best_params['attention_threshold']
            config.num_steps = interpolation_optimizer.best_params['num_steps']
            config.p = interpolation_optimizer.best_params['p']
            config.q = interpolation_optimizer.best_params['q']
            config.num_interpolations = interpolation_optimizer.best_params['num_interpolations']
            config.min_alpha = interpolation_optimizer.best_params['min_alpha']
            config.max_alpha = interpolation_optimizer.best_params['max_alpha']
            config.variability = interpolation_optimizer.best_params['variability']
        else:
            logger.warning("No successful optimization trials. Using default parameters.")

    # Generate balanced dataset
    X_train_balanced, y_train_balanced = generate_balanced_dataset(
        X_train, y_train, attention_weights, config
    )

    if X_train_balanced is None:
        logger.error("Balanced dataset generation failed.")
        return None, None, None

    # Test final results
    results = test_classification_models(
        X_train_balanced, y_train_balanced,
        X_val, y_val,
        X_test, y_test,
        RANDOM_STATE=config.random_state
    )
    results['GAT Training Time'] = training_time

    return results, X_train_balanced, y_train_balanced