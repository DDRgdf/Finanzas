import optuna
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class HyperparameterOptimizer:
    def __init__(self, model_builder):
        self.model_builder = model_builder
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """Optimización bayesiana de hiperparámetros"""
        try:
            def objective(trial):
                # ... (código de optimización)
                return min(history.history['val_loss'])
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30)
            return study.best_params
            
        except Exception as e:
            print(f"Error en la optimización de hiperparámetros: {str(e)}")
            raise