# Importaciones de TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Conv1D, BatchNormalization,
    Activation, Bidirectional, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    @staticmethod
    def create_hybrid_model(input_shape, learning_rate=0.001):
        """Crear modelo híbrido CNN-LSTM con atención"""
        try:
            inputs = Input(shape=input_shape)
            
            # CNN para patrones locales
            conv1 = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            conv1 = Dropout(0.2)(conv1)
            
            # ... (resto del código del modelo)
            
            return model
            
        except Exception as e:
            print(f"Error al crear el modelo: {str(e)}")
            raise