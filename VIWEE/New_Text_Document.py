# Importar librerías necesarias
import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, roc_curve, auc
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import queue
import scipy.stats as stats
import streamlit as st
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, 
    LSTM, Bidirectional, MultiHeadAttention, LayerNormalization,
    Dense, Dropout, GlobalAveragePooling1D, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

def obtener_input_usuario():
    """Función para obtener input del usuario"""
    print("\n=== Configuración del Análisis ===")
    print("Símbolos populares: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon)")
    simbolo = input("Ingrese el símbolo de la acción: ").upper()
    
    print("\nSeleccione el horizonte temporal:")
    print("1. Diario")
    print("2. Semanal")
    print("3. Mensual")
    opcion = input("Ingrese su opción (1-3): ")
    
    horizonte = {
        '1': 'diario',
        '2': 'semanal',
        '3': 'mensual'
    }.get(opcion, 'diario')
    
    return simbolo, horizonte

def obtener_datos_historicos(simbolo, horizonte):
    """Función para descargar datos históricos de Yahoo Finance"""
    intervalos = {
        'diario': '1d',
        'semanal': '1wk',
        'mensual': '1mo'
    }
    
    try:
        accion = yf.Ticker(simbolo)
        datos = accion.history(period='1y', interval=intervalos[horizonte])
        print(f"\nDatos históricos obtenidos exitosamente para {simbolo}")
        return datos
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return None

def calcular_indicadores(datos):
    """Función para calcular indicadores técnicos"""
    if datos is None:
        return None
    
    datos['SMA10'] = SMAIndicator(close=datos['Close'], window=10).sma_indicator()
    datos['SMA50'] = SMAIndicator(close=datos['Close'], window=50).sma_indicator()
    datos['EMA10'] = EMAIndicator(close=datos['Close'], window=10).ema_indicator()
    datos['RSI'] = RSIIndicator(close=datos['Close'], window=14).rsi()
    
    macd = MACD(close=datos['Close'])
    datos['MACD'] = macd.macd()
    datos['MACD_Signal'] = macd.macd_signal()
    
    return datos

def visualizar_precios(datos, simbolo):
    """Función para visualizar precios y volumen"""
    if datos is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(datos.index, datos['Close'], label='Precio de Cierre')
    ax1.plot(datos.index, datos['SMA10'], label='SMA 10', alpha=0.7)
    ax1.plot(datos.index, datos['SMA50'], label='SMA 50', alpha=0.7)
    ax1.plot(datos.index, datos['EMA10'], label='EMA 10', alpha=0.7)
    
    ax1.set_title(f'Precio histórico de {simbolo}')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio')
    ax1.legend()
    ax1.grid(True)
    
    ax2.bar(datos.index, datos['Volume'], alpha=0.7)
    ax2.set_title('Volumen de transacciones')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Volumen')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def mostrar_resumen_estadistico(datos):
    """Función para mostrar estadísticas descriptivas"""
    if datos is None:
        return
    
    print("\nResumen Estadístico:")
    print(f"Precio promedio: {datos['Close'].mean():.2f}")
    print(f"Precio máximo: {datos['Close'].max():.2f}")
    print(f"Precio mínimo: {datos['Close'].min():.2f}")
    print(f"Desviación estándar: {datos['Close'].std():.2f}")

def mapa_correlaciones(datos):
    """Función para generar mapa de calor de correlaciones"""
    if datos is None:
        return
    
    cols_correlacion = ['Close', 'Volume', 'SMA10', 'SMA50', 'EMA10', 'RSI', 'MACD']
    correlation_matrix = datos[cols_correlacion].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Mapa de Correlaciones')
    plt.tight_layout()
    plt.show()

def mostrar_datos(datos):
    """Función para mostrar los datos en consola"""
    if datos is not None:
        print("\nÚltimos 5 registros de precios:")
        print(datos[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

def actualizar_datos_tiempo_real(simbolo):
    """Función para actualizar datos en tiempo real"""
    while True:
        accion = yf.Ticker(simbolo)
        datos_actuales = accion.history(period='1d', interval='1m').tail(1)
        
        print("\nDatos en tiempo real:")
        print(datos_actuales[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        time.sleep(60)  # Esperar 1 minuto antes de la siguiente actualización

def crear_variables_derivadas(datos):
    """Función para crear variables derivadas y objetivo"""
    if datos is None:
        return None
    
    datos['Retorno'] = datos['Close'].pct_change()
    datos['Precio_Siguiente'] = datos['Close'].shift(-1)
    datos['Sube_Baja'] = (datos['Close'].shift(-1) > datos['Close']).astype(int)
    
    datos = datos.dropna()
    
    return datos

def preparar_datos(datos):
    """Función para preparar datos para Machine Learning"""
    if datos is None:
        return None, None, None, None, None
    
    try:
        # Asegurarnos que Close y Volume son numéricos
        datos['Close'] = pd.to_numeric(datos['Close'], errors='coerce')
        datos['Volume'] = pd.to_numeric(datos['Volume'], errors='coerce')
        
        # Indicadores técnicos básicos
        datos['SMA10'] = datos['Close'].rolling(window=10).mean()
        datos['SMA50'] = datos['Close'].rolling(window=50).mean()
        datos['EMA10'] = datos['Close'].ewm(span=10, adjust=False).mean()
        
        # RSI
        datos['RSI'] = RSIIndicator(close=datos['Close']).rsi()
        
        # MACD
        macd = MACD(close=datos['Close'])
        datos['MACD'] = macd.macd()
        datos['MACD_Signal'] = macd.macd_signal()
        
        # Variable objetivo
        datos['Target'] = datos['Close'].shift(-1)
        
        # Seleccionar features
        features = ['Close', 'Volume', 'SMA10', 'SMA50', 'EMA10', 'RSI', 'MACD']
        
        # Eliminar filas con valores nulos
        datos_clean = datos.dropna()
        
        if datos_clean.empty:
            raise ValueError("No hay suficientes datos después de limpiar")
        
        X = datos_clean[features].astype('float32')  # Convertir a float32
        y = datos_clean['Target'].astype('float32')  # Convertir a float32
        
        # División temporal de datos
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Escalado de datos
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        print(f"Error en preparación de datos: {str(e)}")
        return None, None, None, None, None

def verificar_datos(datos, X_train, y_train):
    """Función para verificar la calidad de los datos preparados"""
    if datos is None or X_train is None or y_train is None:
        return
    
    print("\nVerificación de datos:")
    print("\nEstadísticas de retornos:")
    print(datos['Retorno'].describe())
    
    print("\nDistribución de Sube/Baja:")
    print(datos['Sube_Baja'].value_counts(normalize=True))
    
    correlaciones = pd.DataFrame({
        'Feature': X_train.columns,
        'Correlación con Precio': X_train.corrwith(y_train)
    }).sort_values('Correlación con Precio', ascending=False)
    
    print("\nCorrelaciones con el precio objetivo:")
    print(correlaciones)

def crear_modelo_lstm_regresion(X_train):
    """Función para crear modelo LSTM para regresión"""
    modelo = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

def crear_modelo_lstm_clasificacion(X_train):
    """Función para crear modelo LSTM para clasificación"""
    modelo = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

def modelo_regresion(X_train, X_test, y_train, y_test):
    """Función para entrenar y evaluar modelos de regresión"""
    resultados = {}
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    resultados['Regresión Lineal'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_lr)),
        'MAE': mean_absolute_error(y_test, pred_lr),
        'predicciones': pred_lr
    }
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_rf)),
        'MAE': mean_absolute_error(y_test, pred_rf),
        'predicciones': pred_rf
    }
    
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    resultados['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_xgb)),
        'MAE': mean_absolute_error(y_test, pred_xgb),
        'predicciones': pred_xgb
    }
    
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    modelo_lstm = crear_modelo_lstm_regresion(X_train)
    modelo_lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    pred_lstm = modelo_lstm.predict(X_test_lstm).flatten()
    
    resultados['LSTM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_lstm)),
        'MAE': mean_absolute_error(y_test, pred_lstm),
        'predicciones': pred_lstm
    }
    
    return resultados

def modelo_clasificacion(X_train, X_test, y_train, y_test):
    """Función para entrenar y evaluar modelos de clasificación"""
    resultados = {}
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    resultados['Regresión Logística'] = {
        'reporte': classification_report(y_test, pred_lr),
        'confusion_matrix': confusion_matrix(y_test, pred_lr),
        'predicciones': pred_lr
    }
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    resultados['Random Forest'] = {
        'reporte': classification_report(y_test, pred_rf),
        'confusion_matrix': confusion_matrix(y_test, pred_rf),
        'predicciones': pred_rf
    }
    
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    resultados['XGBoost'] = {
        'reporte': classification_report(y_test, pred_xgb),
        'confusion_matrix': confusion_matrix(y_test, pred_xgb),
        'predicciones': pred_xgb
    }
    
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    modelo_lstm = crear_modelo_lstm_clasificacion(X_train)
    modelo_lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
    pred_lstm = (modelo_lstm.predict(X_test_lstm) > 0.5).flatten()
    
    resultados['LSTM'] = {
        'reporte': classification_report(y_test, pred_lstm),
        'confusion_matrix': confusion_matrix(y_test, pred_lstm),
        'predicciones': pred_lstm
    }
    
    return resultados

def visualizar_predicciones(y_test, resultados_regresion):
    """Función para visualizar predicciones vs valores reales"""
    plt.figure(figsize=(15, 10))
    
    plt.plot(y_test.index, y_test.values, label='Valores Reales', color='black')
    
    colores = ['blue', 'green', 'red', 'purple']
    for (nombre_modelo, resultado), color in zip(resultados_regresion.items(), colores):
        plt.plot(y_test.index, resultado['predicciones'], 
                label=f'Predicciones {nombre_modelo}', 
                alpha=0.7, 
                color=color)
    
    plt.title('Predicciones vs Valores Reales')
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluar_modelos(resultados_regresion, resultados_clasificacion):
    """Función para mostrar evaluación comparativa de modelos"""
    print("\nResultados de Regresión:")
    print("-" * 50)
    for nombre_modelo, metricas in resultados_regresion.items():
        print(f"\n{nombre_modelo}:")
        print(f"RMSE: {metricas['RMSE']:.4f}")
        print(f"MAE: {metricas['MAE']:.4f}")
    
    print("\nResultados de Clasificación:")
    print("-" * 50)
    for nombre_modelo, metricas in resultados_clasificacion.items():
        print(f"\n{nombre_modelo}:")
        print(metricas['reporte'])
        print("Matriz de Confusión:")
        print(metricas['confusion_matrix'])

def optimizar_random_forest(X_train, y_train, tipo='regresion'):
    """Función para optimizar hiperparámetros de Random Forest"""
    param_dist = {
        'n_estimators': stats.randint(100, 500),
        'max_depth': [None] + list(range(10, 50, 10)),
        'min_samples_split': stats.randint(2, 20),
        'min_samples_leaf': stats.randint(1, 10)
    }
    
    modelo = RandomForestRegressor(random_state=42) if tipo == 'regresion' else RandomForestClassifier(random_state=42)
    scoring = 'neg_mean_squared_error' if tipo == 'regresion' else 'f1'
    
    random_search = RandomizedSearchCV(
        modelo, param_distributions=param_dist,
        n_iter=20, cv=5, scoring=scoring,
        random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"\nMejores parámetros Random Forest ({tipo}):")
    print(random_search.best_params_)
    
    return random_search.best_estimator_

def optimizar_xgboost(X_train, y_train, tipo='regresion'):
    """Función para optimizar hiperparámetros de XGBoost"""
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    modelo = XGBRegressor(random_state=42) if tipo == 'regresion' else XGBClassifier(random_state=42)
    scoring = 'neg_mean_squared_error' if tipo == 'regresion' else 'f1'
    
    grid_search = GridSearchCV(
        modelo, param_grid, cv=5,
        scoring=scoring, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"\nMejores parámetros XGBoost ({tipo}):")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_

def optimizar_lstm(X_train, y_train, X_val, y_val, tipo='regresion'):
    """Función para optimizar la red LSTM"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_lstm = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    modelo = Sequential([
        LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.3),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid' if tipo == 'clasificacion' else None)
    ])
    
    modelo.compile(optimizer='adam', loss='mse' if tipo == 'regresion' else 'binary_crossentropy', metrics=['accuracy'] if tipo == 'clasificacion' else [])
    
    modelo.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_lstm, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    return modelo

def validar_modelos(modelos, X, y, cv=5):
    """Función para realizar validación cruzada de los modelos"""
    resultados = {}
    
    for nombre, modelo in modelos.items():
        if nombre != 'LSTM':  # Solo validar modelos scikit-learn
            try:
                scores_rmse = cross_val_score(modelo, X, y, cv=cv, scoring='neg_root_mean_squared_error')
                scores_r2 = cross_val_score(modelo, X, y, cv=cv, scoring='r2')
                
                resultados[nombre] = {
                    'RMSE_mean': -scores_rmse.mean(),
                    'RMSE_std': scores_rmse.std(),
                    'R2_mean': scores_r2.mean(),
                    'R2_std': scores_r2.std()
                }
            except Exception as e:
                print(f"Error al validar modelo {nombre}: {str(e)}")
                resultados[nombre] = {'error': str(e)}
    
    return resultados

def visualizar_roc_curva(y_test, y_pred_proba, titulo='Curva ROC'):
    """Función para visualizar la curva ROC"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.show()

def entrenar_modelos_optimizados(X_train, X_test, y_reg_train, y_reg_test):
    """Función principal para entrenar y evaluar modelos optimizados"""
    resultados = {}
    modelos_sklearn = {}
    
    try:
        print("\nOptimizando Random Forest...")
        rf_reg = optimizar_random_forest(X_train, y_reg_train, tipo='regresion')
        modelos_sklearn['Random Forest'] = rf_reg
    except Exception as e:
        print(f"Error en Random Forest: {str(e)}")
    
    try:
        print("\nOptimizando XGBoost...")
        xgb_reg = optimizar_xgboost(X_train, y_reg_train, tipo='regresion')
        modelos_sklearn['XGBoost'] = xgb_reg
    except Exception as e:
        print(f"Error en XGBoost: {str(e)}")
    
    print("\nRealizando validación cruzada...")
    resultados_validacion = validar_modelos(modelos_sklearn, X_train, y_reg_train)
    
    try:
        print("\nEntrenando LSTM...")
        X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        modelo_lstm = crear_modelo_lstm_regresion(X_train)
        modelo_lstm.fit(X_train_lstm, y_reg_train, epochs=50, batch_size=32, verbose=0)
        modelos_sklearn['LSTM'] = modelo_lstm
    except Exception as e:
        print(f"Error en LSTM: {str(e)}")
    
    return modelos_sklearn, resultados_validacion

def main():
    """Función principal del programa"""
    print("=== Programa de Análisis de Acciones ===")
    
    simbolo, horizonte = obtener_input_usuario()
    datos = obtener_datos_historicos(simbolo, horizonte)
    datos = calcular_indicadores(datos)
    datos = crear_variables_derivadas(datos)
    
    X_train, X_test, y_reg_train, y_reg_test, scaler = preparar_datos(datos)
    modelos, resultados_validacion = entrenar_modelos_optimizados(X_train, X_test, y_reg_train, y_reg_test)
    
    print("\nResultados de Validación:")
    print(pd.DataFrame(resultados_validacion))
    
    predicciones = {}
    for nombre, modelo in modelos.items():
        try:
            if nombre == 'LSTM':
                X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
                pred = modelo.predict(X_test_lstm).flatten()
            else:
                pred = modelo.predict(X_test)
            predicciones[nombre] = {'predicciones': pred}
        except Exception as e:
            print(f"Error al predecir con {nombre}: {str(e)}")
    
    if predicciones:
        visualizar_predicciones(y_reg_test, predicciones)
    
    mostrar_datos(datos)
    mostrar_resumen_estadistico(datos)
    visualizar_precios(datos, simbolo)
    mapa_correlaciones(datos)
    
    print("\nAnálisis completado.")

def crear_grafico_interactivo():
    """Función para crear un gráfico interactivo con Plotly"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
    return fig

def actualizar_grafico(fig, datos, predicciones, simbolo):
    """Función para actualizar el gráfico interactivo"""
    fig.data = []
    
    fig.add_trace(go.Scatter(x=datos.index, y=datos['Close'], name='Precio de Cierre', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['SMA10'], name='SMA 10', line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['SMA50'], name='SMA 50', line=dict(color='green', dash='dash')), row=1, col=1)
    fig.add_trace(go.Bar(x=datos.index, y=datos['Volume'], name='Volumen', marker_color='gray'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos.index, y=datos['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    
    if predicciones is not None:
        ultima_fecha = datos.index[-1]
        fechas_prediccion = [ultima_fecha + timedelta(days=i) for i in range(1, len(predicciones)+1)]
        fig.add_trace(go.Scatter(x=fechas_prediccion, y=predicciones, name='Predicción', line=dict(color='red', dash='dot')), row=1, col=1)
    
    fig.update_layout(title=f'Análisis en tiempo real - {simbolo}', height=800, showlegend=True)
    return fig

def procesar_datos_tiempo_real(datos_cola, stop_event):
    """Función para procesar datos en tiempo real"""
    while not stop_event.is_set():
        try:
            datos = datos_cola.get(timeout=1)
            if datos is not None:
                datos = calcular_indicadores(datos)
                datos = crear_variables_derivadas(datos)
                return datos
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error procesando datos: {e}")
    return None

def actualizar_predicciones(modelos, X_nuevo):
    """Función para actualizar predicciones con nuevos datos"""
    predicciones = {}
    
    for nombre, modelo in modelos.items():
        try:
            if nombre == 'LSTM':
                X_nuevo_lstm = X_nuevo.values.reshape((X_nuevo.shape[0], X_nuevo.shape[1], 1))
                pred = modelo.predict(X_nuevo_lstm, verbose=0).flatten()
            else:
                pred = modelo.predict(X_nuevo)
            predicciones[nombre] = pred
        except Exception as e:
            print(f"Error al predecir con {nombre}: {str(e)}")
            continue
    
    return predicciones

def app_streamlit():
    """Función principal para la aplicación Streamlit"""
    st.title('Predicción de Acciones en Tiempo Real')
    
    st.sidebar.header('Configuración')
    simbolo = st.sidebar.text_input('Símbolo de la acción:', 'AAPL')
    horizonte = st.sidebar.selectbox('Horizonte temporal:', ['diario', 'semanal', 'mensual'])
    
    if st.sidebar.button('Iniciar análisis'):
        datos = obtener_datos_historicos(simbolo, horizonte)
        if datos is None:
            st.error('Error al obtener datos')
            return
        
        datos = calcular_indicadores(datos)
        datos = crear_variables_derivadas(datos)
        X_train, X_test, y_reg_train, y_reg_test, scaler = preparar_datos(datos)
        
        modelos, _ = entrenar_modelos_optimizados(X_train, X_test, y_reg_train, y_reg_test)
        
        fig = crear_grafico_interactivo()
        grafico_placeholder = st.empty()
        
        datos_cola = queue.Queue()
        stop_event = threading.Event()
        
        try:
            while True:
                nuevos_datos = obtener_datos_historicos(simbolo, horizonte)
                if nuevos_datos is not None:
                    datos_cola.put(nuevos_datos)
                    datos_procesados = procesar_datos_tiempo_real(datos_cola, stop_event)
                    if datos_procesados is not None:
                        X_nuevo = preparar_datos_prediccion(datos_procesados)
                        predicciones = actualizar_predicciones(modelos, X_nuevo)
                        fig = actualizar_grafico(fig, datos_procesados, predicciones.get('Random Forest'), simbolo)
                        grafico_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(60)  # Actualizar cada minuto
                
        except KeyboardInterrupt:
            stop_event.set()
            st.warning('Análisis detenido por el usuario')

def preparar_datos_prediccion(datos):
    """Función para preparar datos para predicción"""
    features = ['Close', 'Volume', 'SMA10', 'SMA50', 'EMA10', 'RSI', 'MACD', 'Retorno']
    return datos[features].tail(1)

def actualizar_grafico_tiempo_real(datos, predicciones, simbolo):
    """Función para actualizar el gráfico en tiempo real usando matplotlib"""
    plt.clf()  # Limpiar figura actual
    
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    ax1.plot(datos.index, datos['Close'], label='Precio Real', color='#2E86C1', linewidth=2)
    ax1.plot(datos.index, datos['SMA10'], label='SMA 10', color='#F39C12', alpha=0.7)
    ax1.plot(datos.index, datos['SMA50'], label='SMA 50', color='#27AE60', alpha=0.7)
    
    if predicciones is not None:
        ultima_fecha = datos.index[-1]
        fechas_prediccion = [ultima_fecha + timedelta(days=i) for i in range(1, len(predicciones)+1)]
        ax1.plot(fechas_prediccion, predicciones, label='Predicción', color='#E74C3C', linestyle='--', linewidth=2)
    
    ax1.set_title(f'Análisis en Tiempo Real - {simbolo}', fontsize=12, pad=10)
    ax1.set_xlabel('Fecha', fontsize=10)
    ax1.set_ylabel('Precio ($)', fontsize=10)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(datos.index, datos['Volume'], color='#34495E', alpha=0.7)
    ax2.set_ylabel('Volumen', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(datos.index, datos['RSI'], color='#8E44AD', label='RSI', linewidth=2)
    ax3.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='#27AE60', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def ejecutar_analisis_tiempo_real():
    """Función principal para ejecutar el análisis en tiempo real"""
    print("\n=== Programa de Análisis y Predicción de Acciones ===")
    print("Desarrollado por [Tu Nombre]")
    print("Versión 1.0")
    print("\nEste programa analiza y predice el comportamiento de acciones en tiempo real.")
    
    try:
        simbolo, horizonte = obtener_input_usuario()
        
        plt.style.use('seaborn')
        plt.ion()  # Modo interactivo
        
        print("\nObteniendo datos iniciales...")
        datos = obtener_datos_historicos(simbolo, horizonte)
        if datos is None:
            print("Error: No se pudieron obtener los datos.")
            return
        
        print("Preparando modelos de predicción...")
        datos = calcular_indicadores(datos)
        datos = crear_variables_derivadas(datos)
        X_train, X_test, y_reg_train, y_reg_test, scaler = preparar_datos(datos)
        
        print("Entrenando modelos...")
        modelos, resultados_validacion = entrenar_modelos_optimizados(X_train, X_test, y_reg_train, y_reg_test)
        
        print("\nIniciando análisis en tiempo real...")
        print("Presione Ctrl+C para detener el programa")
        
        while True:
            try:
                nuevos_datos = obtener_datos_historicos(simbolo, horizonte)
                if nuevos_datos is not None:
                    nuevos_datos = calcular_indicadores(nuevos_datos)
                    nuevos_datos = crear_variables_derivadas(nuevos_datos)
                    X_nuevo = preparar_datos_prediccion(nuevos_datos)
                    
                    predicciones = actualizar_predicciones(modelos, X_nuevo)
                    actualizar_grafico_tiempo_real(nuevos_datos, predicciones.get('Random Forest'), simbolo)
                    
                    ultimo_precio = nuevos_datos['Close'].iloc[-1]
                    print(f"\nÚltima actualización: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"Precio actual: ${ultimo_precio:.2f}")
                    print("Predicciones para el siguiente período:")
                    for nombre, pred in predicciones.items():
                        if len(pred) > 0:
                            print(f"- {nombre}: ${pred[-1]:.2f}")
                
                print("\nEsperando nueva actualización...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\nPrograma detenido por el usuario.")
                break
            except Exception as e:
                print(f"\nError durante la actualización: {str(e)}")
                continue
        
    except Exception as e:
        print(f"\nError general: {str(e)}")
    finally:
        plt.ioff()
        plt.close('all')

class AdvancedStockPredictor:
    def __init__(self):
        self.scaler = RobustScaler()  # Cambio a RobustScaler para mejor manejo de outliers
        self.models = {}  # Diccionario para almacenar múltiples modelos
        self.feature_importance = {}
        self.best_window = None
        self.best_features = None
        
    def prepare_features(self, data):
        """Preparar características para el modelo"""
        try:
            # Convertir DataFrame a numpy array y forzar 1D donde sea necesario
            data = data.copy()
            
            # Forzar 1D en las columnas principales
            for col in ['Close', 'High', 'Low', 'Volume']:
                if isinstance(data[col], pd.DataFrame):
                    data[col] = data[col].values.ravel()
                elif isinstance(data[col], np.ndarray) and data[col].ndim > 1:
                    data[col] = data[col].ravel()
                data[col] = pd.Series(data[col], index=data.index)

            # Calcular indicadores con datos ya limpios
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # Medias móviles simples
            data['SMA20'] = close.rolling(window=20).mean().values
            data['SMA50'] = close.rolling(window=50).mean().values
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs)).values
            
            # MACD manual para evitar problemas de dimensionalidad
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            data['MACD'] = (exp1 - exp2).values
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean().values
            
            # ADX manual
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            data['ADX'] = tr.rolling(window=14).mean().values
            
            # Asegurar que todas las columnas sean 1D
            for col in data.columns:
                if isinstance(data[col], pd.DataFrame):
                    data[col] = data[col].values.ravel()
                elif isinstance(data[col], np.ndarray) and data[col].ndim > 1:
                    data[col] = data[col].ravel()
                data[col] = pd.Series(data[col], index=data.index)
            
            # Definir características finales
            self.best_features = ['Close', 'Volume', 'SMA20', 'SMA50', 
                                'RSI', 'MACD', 'Signal_Line', 'ADX']
            
            # Eliminar NaN y verificar dimensiones finales
            data = data.dropna()
            
            # Verificación final
            for feature in self.best_features:
                if isinstance(data[feature], np.ndarray) and data[feature].ndim > 1:
                    data[feature] = data[feature].ravel()
                elif isinstance(data[feature], pd.Series) and isinstance(data[feature].values, np.ndarray) and data[feature].values.ndim > 1:
                    data[feature] = pd.Series(data[feature].values.ravel(), index=data[feature].index)
            
            return data
            
        except Exception as e:
            print(f"Error en la preparación de características: {str(e)}")
            raise

    def create_hybrid_model(self, input_shape, learning_rate=0.001):
        """Crear modelo híbrido CNN-LSTM con atención"""
        try:
            inputs = Input(shape=input_shape)
            
            # CNN para patrones locales
            conv1 = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            conv1 = Dropout(0.2)(conv1)
            
            conv2 = Conv1D(filters=128, kernel_size=3, padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            conv2 = Dropout(0.2)(conv2)
            
            # LSTM bidireccional para dependencias temporales
            lstm1 = Bidirectional(LSTM(100, return_sequences=True))(conv2)
            lstm1 = BatchNormalization()(lstm1)
            lstm1 = Dropout(0.2)(lstm1)
            
            # Mecanismo de atención
            attention = MultiHeadAttention(num_heads=4, key_dim=50)(lstm1, lstm1)
            attention = LayerNormalization(epsilon=1e-6)(attention + lstm1)
            
            # Capa densa con regularización
            dense1 = Dense(64, activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(attention)
            dense1 = Dropout(0.2)(dense1)
            
            # Pooling global
            gap = GlobalAveragePooling1D()(dense1)
            
            # Salida
            outputs = Dense(1)(gap)
            
            model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='huber',
                        metrics=['mae', 'mape'])
            
            return model
            
        except Exception as e:
            print(f"Error al crear el modelo: {str(e)}")
            raise

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimización bayesiana de hiperparámetros con Optuna"""
        try:
            def objective(trial):
                # Parámetros a optimizar
                params = {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                    'lstm_units': trial.suggest_int('lstm_units', 50, 200),
                    'conv_filters': trial.suggest_int('conv_filters', 32, 128),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
                }
                
                # Asegurar que los datos sean float32
                X_train_float = X_train.astype(np.float32)
                y_train_float = y_train.astype(np.float32)
                X_val_float = X_val.astype(np.float32)
                y_val_float = y_val.astype(np.float32)
                
                # Crear y entrenar modelo
                model = self.create_hybrid_model(
                    input_shape=X_train.shape[1:],
                    learning_rate=params['learning_rate']
                )
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train_float, y_train_float,
                    validation_data=(X_val_float, y_val_float),
                    epochs=50,
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                return min(history.history['val_loss'])
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=1)
            
            return study.best_params
            
        except Exception as e:
            print(f"Error en la optimización de hiperparámetros: {str(e)}")
            raise

    def train_model(self, symbol, prediction_days):
        """Entrenar modelo con validación temporal robusta"""
        try:
            print(f"\nObteniendo datos históricos para {symbol}...")
            
            # Obtener datos históricos
            end_date = datetime.now()
            start_date = end_date - timedelta(days=prediction_days + 365*2)
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError("No se pudieron obtener datos para este símbolo")
            
            print("\nPreparando características...")
            data = self.prepare_features(data)
            
            # Asegurar que todas las columnas sean numéricas
            for column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')
            
            # Eliminar filas con valores NaN
            data = data.dropna()
            
            # Crear ventanas temporales de diferentes tamaños
            windows = [30, 60, 90]
            models = {}
            scores = {}
            
            for window in windows:
                print(f"\nProbando ventana de {window} días...")
                X, y = self.create_sequences(data, window)
                
                # División temporal de datos
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size+val_size]
                y_val = y[train_size:train_size+val_size]
                X_test = X[train_size+val_size:]
                y_test = y[train_size+val_size:]
                
                # Optimizar hiperparámetros
                best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
                
                # Entrenar modelo
                model = self.create_hybrid_model(
                    input_shape=(window, len(self.best_features)),
                    learning_rate=best_params['learning_rate']
                )
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=best_params['batch_size'],
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluar modelo
                score = model.evaluate(X_test, y_test, verbose=0)
                scores[window] = score[1]  # MAE
                models[window] = model
            
            # Seleccionar mejor ventana
            self.best_window = min(scores, key=scores.get)
            self.model = models[self.best_window]
            
            print(f"\nMejor ventana temporal: {self.best_window} días")
            print(f"MAE: {scores[self.best_window]:.4f}")
            
            return data
            
        except Exception as e:
            print(f"Error en el entrenamiento del modelo: {str(e)}")
            return None

    def create_sequences(self, data, window_size):
        """Crear secuencias de datos para entrenamiento"""
        try:
            # Convertir a array numpy y asegurar tipo float32
            values = data[self.best_features].astype('float32').values
            X = []
            y = []
            
            for i in range(window_size, len(values)):
                X.append(values[i-window_size:i])
                y.append(data['Close'].iloc[i])
            
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        except Exception as e:
            print(f"Error al crear secuencias: {str(e)}")
            raise

    def predict_future(self, data, days):
        """Predecir precios futuros con intervalos de confianza"""
        try:
            if self.model is None:
                raise ValueError("El modelo no ha sido entrenado")
            
            # Preparar última secuencia
            last_sequence = data[self.best_features].values[-self.best_window:]
            last_sequence = last_sequence.reshape((1, self.best_window, -1))
            
            # Generar predicciones
            predictions = []
            confidence_intervals = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predicción
                pred = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(pred)
                
                # Calcular intervalo de confianza
                std = np.std(current_sequence[0, :, 0])
                confidence_intervals.append([
                    pred - 1.96 * std,
                    pred + 1.96 * std
                ])
                
                # Actualizar secuencia
                new_sequence = np.roll(current_sequence[0], -1, axis=0)
                new_sequence[-1] = pred
                current_sequence = new_sequence.reshape((1, self.best_window, -1))
            
            return np.array(predictions), np.array(confidence_intervals)
            
        except Exception as e:
            print(f"Error en la predicción: {str(e)}")
            return None, None

    def show_prediction_error(self, error_message):
        """Mostrar error de predicción"""
        try:
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert('1.0', f"Error en la predicción: {error_message}")
            self.prediction_data = {}
        except Exception as local_e:  # Cambiar 'e' por 'local_e' para evitar conflicto de scope
            print(f"Error al mostrar mensaje de error: {str(local_e)}")

class ModelEvaluator:
    """Nueva clase para evaluar múltiples modelos"""
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
        }
        self.best_model = None
        self.best_score = float('inf')
        
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimizar hiperparámetros de los modelos"""
        param_grid = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        }
        
        for name, model in self.models.items():
            grid_search = GridSearchCV(
                model, 
                param_grid[name],
                cv=5, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.models[name] = grid_search.best_estimator_
            
            if -grid_search.best_score_ < self.best_score:
                self.best_score = -grid_search.best_score_
                self.best_model = grid_search.best_estimator_

                def prepare_features(self, data):
    """Preparar características para el modelo"""
    try:
        # Convertir DataFrame a numpy array y forzar 1D donde sea necesario
        data = data.copy()
        
        # Forzar 1D en las columnas principales
        for col in ['Close', 'High', 'Low', 'Volume']:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].values.ravel()
            elif isinstance(data[col], np.ndarray) and data[col].ndim > 1:
                data[col] = data[col].ravel()
            data[col] = pd.Series(data[col], index=data.index)

        # Calcular indicadores con datos ya limpios
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Medias móviles simples
        data['SMA20'] = close.rolling(window=20).mean().values
        data['SMA50'] = close.rolling(window=50).mean().values
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs)).values
        
        # MACD manual para evitar problemas de dimensionalidad
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        data['MACD'] = (exp1 - exp2).values
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean().values
        
        # ADX manual
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        data['ADX'] = tr.rolling(window=14).mean().values
        
        # Asegurar que todas las columnas sean 1D
        for col in data.columns:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].values.ravel()
            elif isinstance(data[col], np.ndarray) and data[col].ndim > 1:
                data[col] = data[col].ravel()
            data[col] = pd.Series(data[col], index=data.index)
        
        # Definir características finales
        self.best_features = ['Close', 'Volume', 'SMA20', 'SMA50', 
                            'RSI', 'MACD', 'Signal_Line', 'ADX']
        
        # Eliminar NaN y verificar dimensiones finales
        data = data.dropna()
        
        # Verificación final
        for feature in self.best_features:
            if isinstance(data[feature], np.ndarray) and data[feature].ndim > 1:
                data[feature] = data[feature].ravel()
            elif isinstance(data[feature], pd.Series) and isinstance(data[feature].values, np.ndarray) and data[feature].values.ndim > 1:
                data[feature] = pd.Series(data[feature].values.ravel(), index=data[feature].index)
        
        return data
        
    except Exception as e:
        print(f"Error en la preparación de características: {str(e)}")
        

def generar_reporte_detallado(datos, predicciones, metricas, symbol):
    """Genera un reporte detallado de análisis"""
    ultimo_precio = datos['Close'].iloc[-1]
    volatilidad = datos['Close'].pct_change().std() * 100
    
    reporte = f"""
=== Reporte Detallado de Análisis para {symbol} ===

Resumen de Mercado:
• Último precio: ${ultimo_precio:.2f}
• Rango 52 semanas: ${datos['Close'].min():.2f} - ${datos['Close'].max():.2f}
• Volatilidad anualizada: {volatilidad * np.sqrt(252):.1f}%

Análisis Técnico:
• RSI (14): {datos['RSI'].iloc[-1]:.1f}
• MACD: {'Positivo' if datos['MACD'].iloc[-1] > datos['Signal_Line'].iloc[-1] else 'Negativo'}
• Tendencia SMA: {'Alcista' if datos['SMA20'].iloc[-1] > datos['SMA50'].iloc[-1] else 'Bajista'}
• Bandas de Bollinger: {'Sobrecomprado' if datos['Close'].iloc[-1] > datos['BB_upper'].iloc[-1] else 'Sobrevendido' if datos['Close'].iloc[-1] < datos['BB_lower'].iloc[-1] else 'Normal'}

Métricas del Modelo:
• Error Absoluto Medio (MAE): ${metricas['mae']:.2f}
• Error Cuadrático Medio (RMSE): ${metricas['rmse']:.2f}
• Coeficiente de Determinación (R²): {metricas.get('r2', 0):.3f}
• Precisión Direccional: {metricas.get('direction_accuracy', 0):.1f}%

Predicciones para los próximos 7 días:
"""
    
    for i, pred in enumerate(predicciones[:7], 1):
        cambio = (pred - ultimo_precio) / ultimo_precio * 100
        reporte += f"Día {i}: ${pred:.2f} ({'+' if cambio > 0 else ''}{cambio:.1f}%)\n"
    
    return reporte

def crear_graficos_interactivos(datos, predicciones, symbol):
    """Crea gráficos interactivos usando Plotly"""
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxis=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Precio y Predicciones', 'Volumen', 'Indicadores Técnicos'))
    
    # Gráfico de precios y predicciones
    fig.add_trace(
        go.Candlestick(
            x=datos.index,
            open=datos['Open'],
            high=datos['High'],
            low=datos['Low'],
            close=datos['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Añadir predicciones
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(start=datos.index[-1], periods=len(predicciones)+1)[1:],
            y=predicciones,
            mode='lines',
            name='Predicción',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Volumen
    fig.add_trace(
        go.Bar(
            x=datos.index,
            y=datos['Volume'],
            name='Volumen'
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=datos.index,
            y=datos['RSI'],
            name='RSI'
        ),
        row=3, col=1
    )
    
    # Actualizar diseño
    fig.update_layout(
        title=f'Análisis Técnico de {symbol}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        height=1000
    )
    
    return fig

def calcular_intervalos_confianza(predicciones, nivel_confianza=0.95):
    """Calcula intervalos de confianza para las predicciones"""
    std = np.std(predicciones)
    z_score = stats.norm.ppf((1 + nivel_confianza) / 2)
    margen_error = z_score * std
    
    return [(pred - margen_error, pred + margen_error) for pred in predicciones]

def main():
    print("=== Predictor Avanzado de Precios de Acciones ===")
    
    symbol = input("\nIngrese el símbolo de la acción (ej: AAPL): ").upper()
    days = int(input("Ingrese el número de días para la predicción: "))
    
    predictor = AdvancedStockPredictor()
    data = predictor.train_model(symbol, days)
    
    if data is not None:
        current_price = data['Close'].iloc[-1]
        predictions, confidence_intervals = predictor.predict_future(data, days)
        
        # Verificar que predictions no esté vacío
        if predictions and len(predictions) > 0:
            future_price = predictions[-1]
            print("\n=== Resultados ===")
            print(f"Precio actual de {symbol}: ${current_price:.2f}")
            print(f"Precio predicho en {days} días: ${future_price:.2f}")
            print(f"Cambio porcentual esperado: {((future_price - current_price) / current_price * 100):.2f}%")
        else:
            print("No se pudo generar una predicción válida")

if __name__ == "__main__":
    main()