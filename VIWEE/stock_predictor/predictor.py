import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# Importaciones relativas dentro del paquete
from .models import ModelBuilder
from .data_processor import DataProcessor
from .optimizer import HyperparameterOptimizer

class AdvancedStockPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.best_window = None
        self.best_features = None
        self.model_builder = ModelBuilder()
        self.data_processor = DataProcessor()
        self.optimizer = HyperparameterOptimizer(self.model_builder)