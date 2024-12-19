import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

class DataProcessor:
    @staticmethod
    def prepare_features(data):
        """Preparar características avanzadas"""
        try:
            df = pd.DataFrame(index=data.index)
            # ... (código de preparación de características)
            return df
        except Exception as e:
            print(f"Error en la preparación de características: {str(e)}")
            raise
    
    @staticmethod
    def create_sequences(data, window_size, features):
        """Crear secuencias de datos para entrenamiento"""
        try:
            values = data[features].astype('float32').values
            # ... (código de creación de secuencias)
            return X, y
        except Exception as e:
            print(f"Error al crear secuencias: {str(e)}")
            raise