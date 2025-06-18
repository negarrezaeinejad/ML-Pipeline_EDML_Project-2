#!/usr/bin/env python3
"""
Improved ML Pipeline Demo - Demonstrating key improvements
This script shows how to implement some of the recommended improvements
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedMLPipeline:
    """
    Improved ML Pipeline with better practices
    """
    
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or self._default_config()
        self.models = {}
        self.best_model = None
        self.scaler = None
        
    def _default_config(self):
        """Default configuration"""
        return {
            'data': {
                'n_samples_a': 1000,
                'n_samples_b': 500,
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 5],
                        'learning_rate': [0.1, 0.2]
                    }
                },
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 5, None]
                    }
                }
            },
            'evaluation': {
                'cv_folds': 5,
                'scoring': 'neg_mean_squared_error'
            }
        }
    
    def validate_data(self, df, required_columns=None):
        """Validate input data"""
        if required_columns is None:
            required_columns = ['age', 'temperature', 'vibration', 'current_draw']
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Data contains missing values")
        
        # Check for outliers (simple IQR method)
        for col in required_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                logger.warning(f"Column '{col}' contains {len(outliers)} potential outliers")
        
        logger.info(f"Data validation passed. Shape: {df.shape}")
        return True
    
    def generate_data(self, n_samples, model_type='A', seed=42):
        """Generate synthetic data with validation"""
        try:
            np.random.seed(seed)
            age = np.random.uniform(0, 10, n_samples)
            
            if model_type == 'A':
                temperature = 60 + 10 * age + np.random.normal(0, 2, n_samples)
                vibration = 5 + 2 * age + np.random.normal(0, 1.5, n_samples)
                current_draw = 10 + 0.5 * age + np.random.normal(0, 0.5, n_samples)
            elif model_type == 'B':
                temperature = 55 + 8 * age + np.random.normal(0, 1.5, n_samples)
                vibration = 4 + 1.5 * age + np.random.normal(0, 1, n_samples)
                current_draw = 9 + 0.4 * age + np.random.normal(0, 4, n_samples)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            rul = 100 - (10 * age + 0.2 * temperature + 0.5 * vibration + 0.1 * current_draw) + np.random.normal(0, 0.5, n_samples)
            rul = np.clip(rul, 0, 100)
            
            data = pd.DataFrame({
                'age': age,
                'temperature': temperature,
                'vibration': vibration,
                'current_draw': current_draw,
                'rul': rul
            })
            
            self.validate_data(data)
            logger.info(f"Generated {n_samples} samples for model type {model_type}")
            return data
            
        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            raise
    
    def train_models(self, X_train, y_train):
        """Train multiple models with hyperparameter tuning"""
        logger.info("Starting model training and hyperparameter tuning...")
        
        best_score = float('-inf')
        
        for model_name, model_config in self.config['models'].items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_config['model'])
                ])
                
                # Hyperparameter tuning
                param_grid = {}
                for param, values in model_config['params'].items():
                    param_grid[f'model__{param}'] = values
                
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=self.config['evaluation']['cv_folds'],
                    scoring=self.config['evaluation']['scoring'],
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store results
                self.models[model_name] = {
                    'pipeline': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
                logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
                
                # Track best model
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    self.best_model = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Best model: {self.best_model} with CV score: {best_score:.4f}")
    
    def evaluate_comprehensive(self, X_test, y_test, model_name=None):
        """Comprehensive model evaluation"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            pipeline = self.models[model_name]['pipeline']
            y_pred = pipeline.predict(X_test)
            
            # Calculate multiple metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE (avoiding division by zero)
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
            
            logger.info(f"Evaluation results for {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics, y_pred
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, filepath, model_name=None):
        """Save the best model"""
        if model_name is None:
            model_name = self.best_model
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            dump(self.models[model_name]['pipeline'], filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def run_demo(self):
        """Run the improved pipeline demo"""
        logger.info("Starting Improved ML Pipeline Demo")
        
        try:
            # Generate data
            data_a = self.generate_data(
                self.config['data']['n_samples_a'], 
                model_type='A'
            )
            data_b = self.generate_data(
                self.config['data']['n_samples_b'], 
                model_type='B', 
                seed=101
            )
            
            # Split Model A data for training
            X = data_a.drop('rul', axis=1)
            y = data_a['rul']
            X_train, X_test_a, y_train, y_test_a = train_test_split(
                X, y, 
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )
            
            # Prepare Model B test data
            X_test_b = data_b.drop('rul', axis=1)
            y_test_b = data_b['rul']
            
            # Train models
            self.train_models(X_train, y_train)
            
            # Evaluate on both test sets
            logger.info("\n" + "="*50)
            logger.info("EVALUATION ON MODEL A TEST SET")
            logger.info("="*50)
            metrics_a, pred_a = self.evaluate_comprehensive(X_test_a, y_test_a)
            
            logger.info("\n" + "="*50)
            logger.info("EVALUATION ON MODEL B TEST SET")
            logger.info("="*50)
            metrics_b, pred_b = self.evaluate_comprehensive(X_test_b, y_test_b)
            
            # Calculate performance degradation
            rmse_degradation = ((metrics_b['RMSE'] - metrics_a['RMSE']) / metrics_a['RMSE']) * 100
            logger.info(f"\nPerformance Degradation (Model A → Model B):")
            logger.info(f"RMSE Degradation: {rmse_degradation:.1f}%")
            
            # Save best model
            os.makedirs('models_improved', exist_ok=True)
            self.save_model('models_improved/best_model.joblib')
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise

def main():
    """Main function to run the demo"""
    try:
        pipeline = ImprovedMLPipeline()
        pipeline.run_demo()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main())