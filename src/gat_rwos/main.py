import os
from dataclasses import dataclass
import argparse
from typing import Dict, Any, List
from collections import Counter

import pandas as pd
import yaml
import torch
import optuna
import logging

from .utils import get_scaler, save_results
from .data import load_datasets, preprocess
from .training import trainer, TrainingConfig
from .evaluation import test_classification_models
from .visualization import plot_original_vs_oversampled

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters."""
    random_state: int
    data_folder: str
    results_folder: str
    scaler: str
    val_size: float
    test_size: float
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any], override_random_state: int = None) -> 'ExperimentConfig':
        random_state = override_random_state if override_random_state is not None else config['random_state']
        return cls(
            random_state=random_state,
            data_folder=config['data_folder'],
            results_folder=config['results_folder'],
            scaler=config.get('scaler', 'none'),
            val_size=config.get('val_size', 0.1),
            test_size=config.get('test_size', 0.1)
        )

class ExperimentRunner:
    """Handles the execution of experiments on datasets."""
    
    def __init__(self, config_path: str, datasets: List[str], tune: bool, random_state: int = None):
        self.config_path = config_path
        self.datasets = datasets
        self.tune = tune
        self.config = self._load_config()
        self.exp_config = ExperimentConfig.from_dict(self.config['defaults'], override_random_state=random_state)
        self.optuna_ranges = self.config['tuning']['optuna']['ranges']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise

    def _create_training_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Create training configuration from parameters."""
        defaults = self.config.get('defaults', {})
        return TrainingConfig(
            graph_method=params.get('graph_method', defaults.get('graph_method', 'distance')),
            similarity_method=params.get('similarity_method', defaults.get('similarity_method', 'cosine')),
            similarity_threshold=params.get('similarity_threshold', defaults.get('similarity_threshold', 0.8)),
            n_neighbors=params.get('n_neighbors', defaults.get('n_neighbors', 10)),
            optimizer_name=defaults.get('optimizer', {}).get('name', 'adam'),
            optimizer_parms=defaults.get('optimizer', {}).get('params', {}),
            scheduler_name=defaults.get('scheduler', {}).get('name', 'none'),
            scheduler_parms=defaults.get('scheduler', {}).get('params', {}),
            hid=params.get('hid', defaults.get('hid', 16)),
            in_head=params.get('in_head', defaults.get('in_head', 4)),
            out_head=params.get('out_head', defaults.get('out_head', 1)),
            dropout_rate=params.get('dropout_rate', defaults.get('dropout_rate', 0.5)),
            num_hidden_layers=defaults.get('num_hidden_layers', 2),
            return_attention=defaults.get('return_attention', False),
            num_epochs=defaults.get('num_epochs', 100),
            patience=defaults.get('patience', 10),
            device=str(self.device),
            aggregation_method=params.get('aggregation_method', defaults.get('aggregation_method', 'mean')),
            attention_threshold=params.get('attention_threshold', defaults.get('attention_threshold', 0.5)),
            num_steps=params.get('num_steps', defaults.get('num_steps', 10)),
            p=params.get('p', defaults.get('p', 1)),
            q=params.get('q', defaults.get('q', 1)),
            num_interpolations=params.get('num_interpolations', defaults.get('num_interpolations', 5)),
            min_alpha=params.get('min_alpha', defaults.get('min_alpha', 0.1)),
            max_alpha=params.get('max_alpha', defaults.get('max_alpha', 0.9)),
            variability=params.get('variability', defaults.get('variability', 0.1)),
            tune=self.tune,
            n_trials_attention=defaults.get('n_trials_attention', 2),
            n_trials_interpolation=defaults.get('n_trials_interpolation', 5),
            random_state=self.exp_config.random_state
        )

    def _create_optuna_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create parameters for Optuna trial."""

        ranges = self.optuna_ranges
        params = {
            'graph_method': 'distance', # we only use distance for now
            'similarity_method': trial.suggest_categorical("similarity_method", ranges['similarity_methods']),
            'hid': trial.suggest_int("hid", ranges['hid']['min'], ranges['hid']['max']),
            'in_head': trial.suggest_int("in_head", ranges['in_head']['min'], ranges['in_head']['max']),
            'out_head': trial.suggest_int("out_head", ranges['out_head']['min'], ranges['out_head']['max']),
            'dropout_rate': trial.suggest_float("dropout_rate", ranges['dropout_rate']['min'], ranges['dropout_rate']['max']),
        }
        
        if params['graph_method'] == "knn":
            params['n_neighbors'] = trial.suggest_int("n_neighbors", ranges['n_neighbors']['min'], ranges['n_neighbors']['max'])
            params['similarity_threshold'] = None
        else:
            params['similarity_threshold'] = trial.suggest_float(
                "similarity_threshold", ranges['similarity_threshold']['min'], ranges['similarity_threshold']['max']
            )
            params['n_neighbors'] = None
        return params

    def _process_dataset(self, dataset_name: str, dataset: pd.DataFrame) -> None:
        """Process a single dataset."""
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Preprocess data
        scaler = get_scaler(self.exp_config.scaler) if self.exp_config.scaler != 'none' else None
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(
            dataset=dataset,
            val_size=self.exp_config.val_size,
            test_size=self.exp_config.test_size,
            scaler=scaler,
            random_state=self.exp_config.random_state
        )
        
        # Print dataset statistics
        self._print_dataset_stats(y_train, y_val, y_test)
        
        if self.tune:
            self._run_hyperparameter_tuning(dataset_name, X_train, X_val, X_test, y_train, y_val, y_test)
            return

        # Run regular training
        config = self._create_training_config({})
        results, X_train_balanced, y_train_balanced = trainer(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            config=config
        )
        
        self._save_results(
            dataset_name, results, X_train, y_train,
            X_train_balanced, y_train_balanced,
            X_val, y_val, X_test, y_test
        )

    def _print_dataset_stats(self, y_train, y_val, y_test) -> None:
        """Print dataset statistics."""
        counts = {
            "train": Counter(y_train),
            "val": Counter(y_val),
            "test": Counter(y_test)
        }
        for key, count in counts.items():
            logger.info(f"{key.capitalize()}: {count}")

    def _run_hyperparameter_tuning(self, dataset_name: str, X_train, X_val, X_test, y_train, y_val, y_test) -> None:
        """Run hyperparameter tuning using Optuna and then full training with best parameters."""
        def objective(trial):
            params = self._create_optuna_trial_params(trial)
            config = self._create_training_config(params)
            
            results, _, _ = trainer(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                config=config
            )

            # if results is not instance of pd.DataFrame, return 0.0
            if not isinstance(results, pd.DataFrame):
                return 0.0
            return results['Validation G-Mean'].max()
        
        # Run hyperparameter optimization
        study = optuna.create_study(direction="maximize", study_name=dataset_name)
        study.optimize(objective, n_trials=self.config['defaults']['n_trials_main'])
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best trial: {study.best_trial}")
        
        if study.best_value == 0.0:
            logger.warning("No results found during hyperparameter tuning. Please set higher number of trials.")
            return
        
        # Run full training with best parameters
        best_params = study.best_params
        best_config = self._create_training_config(best_params)
        best_config.tune = False  # Set tune to False for final training
        
        results, X_train_balanced, y_train_balanced = trainer(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            config=best_config
        )
        
        # Save results from the best model
        self._save_results(
            dataset_name,
            results,
            X_train, y_train,
            X_train_balanced, y_train_balanced,
            X_val, y_val, X_test, y_test
        )

    def _save_results(self, dataset_name: str, results: pd.DataFrame,
                     X_train, y_train, X_train_balanced, y_train_balanced,
                     X_val, y_val, X_test, y_test) -> None:
        """Save results and visualizations."""
        # Create results directory if it doesn't exist
        dataset_results_path = os.path.join(self.exp_config.results_folder, dataset_name)
        os.makedirs(dataset_results_path, exist_ok=True)
        
        # Generate and save visualizations
        fig, _ = plot_original_vs_oversampled(
            X_train=X_train, y_train=y_train,
            X_train_balanced=X_train_balanced, y_train_balanced=y_train_balanced,
            figsize=(15, 5)
        )
        fig_path = os.path.join(dataset_results_path, f"{dataset_name}_original_vs_oversampled.png")
        fig.savefig(fig_path)
        logger.info(f"Saved visualization to {fig_path}")
        
        # Save results
        balanced_data = pd.concat([X_train_balanced, y_train_balanced], axis=1)
        original_results = test_classification_models(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            RANDOM_STATE=self.exp_config.random_state
        )
        
        save_results(df=results, results_path=self.exp_config.results_folder,
                    file_name=f'{dataset_name}/{dataset_name}_results')
        save_results(df=original_results, results_path=self.exp_config.results_folder,
                    file_name=f'{dataset_name}/{dataset_name}_original_results')
        save_results(df=balanced_data, results_path=self.exp_config.results_folder,
                    file_name=f'{dataset_name}/{dataset_name}_balanced')
        logger.info(f"Saved results for dataset: {dataset_name}")

    def run(self) -> None:
        """Run experiments on all datasets."""
        datasets = load_datasets(self.exp_config.data_folder, self.datasets)
        for dataset_name, dataset in datasets:
            self._process_dataset(dataset_name, dataset)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GAT-RWOS: Graph Attention Network-based Random Walk Oversampling')
    parser.add_argument('--datasets', nargs='+', help='Names of datasets to process (without extension)')
    parser.add_argument('--tune', action='store_true', help='Whether to perform hyperparameter tuning')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility. If specified, overwrites the one in the configuration file.')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main entry point of the program."""
    args = parse_args()
    runner = ExperimentRunner(args.config, args.datasets, args.tune, args.random_state)
    runner.run()

if __name__ == '__main__':
    main()
