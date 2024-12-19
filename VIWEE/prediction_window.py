import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from ta.trend import MACD

class PredictionWindow:
    def __init__(self, symbol=None):
        self.window = tk.Toplevel()
        self.window.title("Análisis Predictivo Avanzado")
        self.window.geometry("1400x900")
        self.window.configure(bg='#1c1c1c')
        
        # Variables de control
        self.symbol = symbol
        self.data = None
        self.predictions = None
        self.symbol_var = tk.StringVar(value=symbol if symbol else "")
        self.scaled_data = None
        
        # Configurar estilo
        self.setup_style()
        
        # Configurar GUI
        self._setup_gui()
        
        # Configurar cierre de ventana
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Si se proporciona un símbolo, cargar datos inmediatamente
        if self.symbol:
            self.load_data()

    def setup_style(self):
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#1c1c1c')
        self.style.configure('Custom.TLabel',
                           background='#1c1c1c',
                           foreground='white',
                           font=('Helvetica', 10))
        self.style.configure('Title.TLabel',
                           background='#1c1c1c',
                           foreground='white',
                           font=('Helvetica', 12, 'bold'))

    def _setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.window, style='Custom.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Panel superior
        self.setup_control_panel()
        
        # Panel de gráficos
        self.setup_charts_panel()
        
        # Panel de predicciones
        self.setup_prediction_panel()
        
        # Panel de métricas
        self.setup_metrics_panel()

    def setup_control_panel(self):
        control_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        control_frame.pack(fill='x', pady=5)
        
        # Días para predecir
        ttk.Label(control_frame, 
                 text="Días a predecir:", 
                 style='Custom.TLabel').pack(side='left', padx=5)
        
        self.days_var = tk.StringVar(value="30")
        self.days_entry = ttk.Entry(control_frame, 
                                  textvariable=self.days_var,
                                  width=5)
        self.days_entry.pack(side='left', padx=5)
        
        # Botón de predicción
        self.predict_button = ttk.Button(control_frame,
                                       text="Generar Predicción",
                                       command=self.generate_prediction)
        self.predict_button.pack(side='left', padx=5)
        
        # Información del modelo
        self.model_info = ttk.Label(control_frame,
                                  text="Modelo: LSTM + Análisis Técnico",
                                  style='Custom.TLabel')
        self.model_info.pack(side='right', padx=10)

    def setup_charts_panel(self):
        charts_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        charts_frame.pack(expand=True, fill='both', pady=5)
        
        self.fig = Figure(figsize=(14, 8), facecolor='#1c1c1c')
        
        # Gráfico principal
        self.ax_main = self.fig.add_subplot(211)
        self.ax_main.set_facecolor('#1c1c1c')
        self.ax_main.tick_params(colors='white')
        
        # Gráfico de confianza
        self.ax_conf = self.fig.add_subplot(212)
        self.ax_conf.set_facecolor('#1c1c1c')
        self.ax_conf.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

    def setup_prediction_panel(self):
        pred_frame = ttk.LabelFrame(self.main_frame,
                                  text="Análisis de Predicciones",
                                  style='Custom.TFrame')
        pred_frame.pack(fill='x', pady=5)
        
        # Grid para predicciones
        self.pred_grid = ttk.Frame(pred_frame, style='Custom.TFrame')
        self.pred_grid.pack(fill='x', padx=5, pady=5)
        
        # Labels para predicciones con más detalles
        self.prediction_labels = {
            'short': {
                'title': ttk.Label(self.pred_grid, 
                                 text="Corto Plazo (7d)",
                                 style='Title.TLabel'),
                'price': ttk.Label(self.pred_grid, 
                                 text="Precio: --",
                                 style='Custom.TLabel'),
                'change': ttk.Label(self.pred_grid,
                                  text="Cambio: --",
                                  style='Custom.TLabel')
            },
            'medium': {
                'title': ttk.Label(self.pred_grid,
                                 text="Medio Plazo (30d)",
                                 style='Title.TLabel'),
                'price': ttk.Label(self.pred_grid,
                                 text="Precio: --",
                                 style='Custom.TLabel'),
                'change': ttk.Label(self.pred_grid,
                                  text="Cambio: --",
                                  style='Custom.TLabel')
            },
            'long': {
                'title': ttk.Label(self.pred_grid,
                                 text="Largo Plazo (90d)",
                                 style='Title.TLabel'),
                'price': ttk.Label(self.pred_grid,
                                 text="Precio: --",
                                 style='Custom.TLabel'),
                'change': ttk.Label(self.pred_grid,
                                  text="Cambio: --",
                                  style='Custom.TLabel')
            }
        }
        
        # Organizar labels en grid
        for i, (period, labels) in enumerate(self.prediction_labels.items()):
            labels['title'].grid(row=0, column=i, padx=10, pady=2)
            labels['price'].grid(row=1, column=i, padx=10, pady=2)
            labels['change'].grid(row=2, column=i, padx=10, pady=2)

    def setup_metrics_panel(self):
        # Guardar la referencia al frame de métricas
        self.metrics_frame = ttk.LabelFrame(self.main_frame,
                                          text="Análisis Técnico y Métricas",
                                          style='Custom.TFrame')
        self.metrics_frame.pack(fill='x', pady=5)
        
        # Frame para métricas principales
        main_metrics = ttk.Frame(self.metrics_frame, style='Custom.TFrame')
        main_metrics.pack(fill='x', padx=5, pady=5)
        
        # Crear variables para métricas
        self.metrics_vars = {
            'confidence': tk.StringVar(value="Confianza: --"),
            'volatility': tk.StringVar(value="Volatilidad: --"),
            'trend': tk.StringVar(value="Tendencia: --"),
            'support': tk.StringVar(value="Soporte: --"),
            'resistance': tk.StringVar(value="Resistencia: --"),
            'rsi': tk.StringVar(value="RSI: --"),
            'macd': tk.StringVar(value="MACD: --")
        }
        
        # Crear y organizar labels con referencias
        self.metric_labels = {}
        for i, (key, var) in enumerate(self.metrics_vars.items()):
            label = ttk.Label(main_metrics, 
                             textvariable=var,
                             style='Custom.TLabel')
            label.grid(row=i//3, column=i%3, padx=10, pady=5)
            self.metric_labels[key] = label

    def set_symbol(self, symbol):
        """Establecer símbolo a analizar"""
        try:
            self.symbol = symbol
            self.symbol_var.set(symbol)
            self.window.title(f"Análisis Predictivo Avanzado - {symbol}")
            
            # Cargar datos inmediatamente cuando se establece el símbolo
            self.load_data()
            
            # Verificar si los datos se cargaron correctamente
            if self.scaled_data is None:
                raise ValueError("No se pudieron cargar los datos del símbolo")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al configurar símbolo: {str(e)}")

    def load_data(self):
        """Cargar datos históricos"""
        try:
            if not self.symbol:
                raise ValueError("Símbolo no especificado")
            
            stock = yf.Ticker(self.symbol)
            data = stock.history(period='2y')
            
            if data.empty:
                raise ValueError("No se encontraron datos para el símbolo")
            
            self.data = data
            self.prepare_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datos: {str(e)}")
            self.data = None
            self.scaled_data = None

    def prepare_data(self):
        """Preparar datos para el modelo"""
        try:
            if self.data is None or self.data.empty:
                raise ValueError("No hay datos disponibles para preparar")
            
            # Indicadores básicos y rápidos
            self.data['SMA10'] = self.data['Close'].rolling(window=10).mean()
            self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
            self.data['RSI'] = self.calculate_rsi(self.data['Close'])
            
            # MACD simplificado
            macd = MACD(close=self.data['Close'])
            self.data['MACD'] = macd.macd()
            self.data['MACD_Signal'] = macd.macd_signal()
            
            # Verificar datos necesarios
            required_columns = ['Close', 'Volume', 'SMA10', 'SMA50', 'RSI', 'MACD']
            data_for_scaling = self.data[required_columns].dropna()
            
            if data_for_scaling.empty:
                raise ValueError("No hay suficientes datos después de calcular indicadores")
            
            # Normalizar datos
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(data_for_scaling)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error preparando datos: {str(e)}")
            self.scaled_data = None

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_prediction(self):
        """Generar predicción"""
        try:
            # Verificar que tenemos datos
            if self.scaled_data is None:
                # Intentar cargar datos si no están disponibles
                self.load_data()
                if self.scaled_data is None:
                    raise ValueError("No hay datos escalados disponibles. Por favor, asegúrese de que el símbolo es válido.")
            
            days = int(self.days_var.get())
            if days <= 0:
                raise ValueError("Los días deben ser un número positivo")
                
            # Crear y entrenar modelo LSTM
            self.train_model()
            
            # Generar predicciones
            self.make_predictions(days)
            
            # Actualizar gráficos y métricas
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")

    def train_model(self):
        """Entrenar modelo LSTM (versión rápida para pruebas)"""
        X, y = self.prepare_sequences()
        
        # Modelo más ligero para pruebas
        self.model = Sequential([
            LSTM(50, input_shape=(60, 6)),  # 6 features
            Dropout(0.1),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Entrenamiento rápido
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def prepare_sequences(self):
        """Preparar secuencias para LSTM"""
        if self.scaled_data is None:
            raise ValueError("No hay datos escalados disponibles")
        
        sequence_length = 60
        
        # Verificar que tenemos suficientes datos
        if len(self.scaled_data) < sequence_length:
            raise ValueError("No hay suficientes datos para crear secuencias")
        
        X = []
        y = []
        
        for i in range(sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-sequence_length:i])
            y.append(self.scaled_data[i, 0])
        
        # Verificar que tenemos datos
        if not X or not y:
            raise ValueError("No se pudieron crear secuencias de entrenamiento")
        
        return np.array(X), np.array(y)

    def make_predictions(self, days):
        """Realizar predicciones (versión rápida)"""
        sequence_length = 60
        last_sequence = self.scaled_data[-sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            next_pred = self.model.predict(
                current_sequence.reshape(1, sequence_length, 6),  # 6 features
                verbose=0
            )
            
            predictions.append(next_pred[0, 0])
            
            # Actualizar secuencia de manera simplificada
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred[0, 0]
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Convertir predicciones a precios reales
        pred_array = np.zeros((len(predictions), 6))
        pred_array[:, 0] = predictions
        self.predictions = self.scaler.inverse_transform(pred_array)[:, 0]

    def update_display(self):
        """Actualizar visualización"""
        if self.predictions is None:
            return
            
        # Actualizar gráfico principal
        self.ax_main.clear()
        self.ax_main.set_facecolor('#1c1c1c')
        self.ax_main.tick_params(colors='white')
        
        # Graficar datos históricos
        self.ax_main.plot(self.data.index[-100:],
                         self.data['Close'].iloc[-100:],
                         color='white',
                         label='Histórico')
        
        # Graficar predicciones
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=len(self.predictions),
            freq='D'
        )
        
        self.ax_main.plot(future_dates,
                         self.predictions,
                         color='lime' if self.predictions[-1] > self.data['Close'].iloc[-1] else 'red',
                         linestyle='--',
                         label='Predicción')
        
        self.ax_main.legend()
        self.ax_main.set_title('Predicción de Precios', color='white')
        
        # Actualizar panel de predicciones
        self.update_predictions_display()
        
        # Actualizar métricas
        self.update_metrics()
        
        self.fig.tight_layout()
        self.canvas.draw()

    def update_metrics(self):
        """Actualizar métricas de predicción"""
        if self.predictions is None:
            return
        
        confidence = self.calculate_confidence()
        volatility = np.std(self.predictions) / np.mean(self.predictions) * 100
        trend = "Alcista" if self.predictions[-1] > self.data['Close'].iloc[-1] else "Bajista"
        support = min(self.predictions)
        resistance = max(self.predictions)
        
        # Actualizar variables de métricas
        self.metrics_vars['confidence'].set(f"Confianza: {confidence:.1f}%")
        self.metrics_vars['volatility'].set(f"Volatilidad: {volatility:.1f}%")
        self.metrics_vars['trend'].set(f"Tendencia: {trend}")
        self.metrics_vars['support'].set(f"Soporte: ${support:.2f}")
        self.metrics_vars['resistance'].set(f"Resistencia: ${resistance:.2f}")
        self.metrics_vars['rsi'].set(f"RSI: {self.data['RSI'].iloc[-1]:.1f}")
        self.metrics_vars['macd'].set(f"MACD: {'Positivo' if self.data['MACD'].iloc[-1] > 0 else 'Negativo'}")
        
        # Actualizar colores según tendencia
        trend_color = 'lime' if trend == "Alcista" else 'red'
        self.metric_labels['trend'].configure(foreground=trend_color)

    def calculate_confidence(self):
        """Calcular confianza del modelo basada en tendencias"""
        if self.predictions is None or len(self.predictions) < 2:
            return 0.0
        
        # Tendencia de precios
        price_trend = np.diff(self.predictions)
        trend_consistency = np.mean(price_trend > 0) if np.mean(self.predictions) > self.data['Close'].iloc[-1] else np.mean(price_trend < 0)
        
        # RSI actual
        current_rsi = self.data['RSI'].iloc[-1]
        rsi_confidence = 1.0 if (current_rsi > 70 and np.mean(price_trend) > 0) or (current_rsi < 30 and np.mean(price_trend) < 0) else 0.5
        
        # MACD
        macd_signal = 1.0 if self.data['MACD'].iloc[-1] > self.data['MACD_Signal'].iloc[-1] else 0.0
        
        # Confianza final
        confidence = (trend_consistency * 0.5 + rsi_confidence * 0.3 + macd_signal * 0.2) * 100
        
        return min(max(confidence, 0), 100)  # Asegurar que está entre 0 y 100

    def update_predictions_display(self):
        """Actualizar panel de predicciones"""
        if self.predictions is None:
            return
        
        current_price = self.data['Close'].iloc[-1]
        
        for period, days in [('short', 7), ('medium', 30), ('long', 90)]:
            if len(self.predictions) >= days:
                future_price = self.predictions[days-1]
                change = ((future_price - current_price) / current_price) * 100
                trend = '↑' if change > 0 else '↓'
                color = 'green' if change > 0 else 'red'
                
                self.prediction_labels[period]['price'].config(
                    text=f"Precio: ${future_price:.2f}"
                )
                self.prediction_labels[period]['change'].config(
                    text=f"Cambio: {trend}{abs(change):.2f}%",
                    foreground=color
                )

    def on_closing(self):
        """Manejar cierre de ventana"""
        self.window.destroy()

