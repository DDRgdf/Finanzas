import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class RealTimeWindow:
    def __init__(self):
        self.window = tk.Toplevel()
        self.window.title("Análisis Avanzado en Tiempo Real")
        self.window.geometry("1400x900")
        self.window.configure(bg='#1c1c1c')
        
        # Variables de control
        self.running = True
        self.symbol = None
        self.data = None
        self.update_interval = 5  # segundos
        self.symbol_var = tk.StringVar()
        
        # Configurar estilo
        self.setup_style()
        
        # Configurar GUI
        self._setup_gui()
        
        # Variables para datos técnicos
        self.technical_indicators = {}
        self.alerts = []
        
        # Configurar cierre de ventana
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        self.style.configure('Alert.TLabel',
                           background='#1c1c1c',
                           foreground='#ff4444',
                           font=('Helvetica', 10, 'bold'))

    def _setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.window, style='Custom.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Panel superior
        self.setup_top_panel()
        
        # Panel de gráficos
        self.setup_charts_panel()
        
        # Panel de indicadores
        self.setup_indicators_panel()
        
        # Panel de alertas
        self.setup_alerts_panel()

    def setup_top_panel(self):
        top_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        top_frame.pack(fill='x', pady=5)
        
        # Información básica
        self.symbol_label = ttk.Label(top_frame, text="", style='Title.TLabel')
        self.symbol_label.pack(side='left', padx=10)
        
        self.price_label = ttk.Label(top_frame, text="", style='Title.TLabel')
        self.price_label.pack(side='left', padx=10)
        
        self.change_label = ttk.Label(top_frame, text="", style='Title.TLabel')
        self.change_label.pack(side='left', padx=10)
        
        # Información adicional
        info_frame = ttk.Frame(top_frame, style='Custom.TFrame')
        info_frame.pack(side='right', padx=10)
        
        self.volume_label = ttk.Label(info_frame, text="", style='Custom.TLabel')
        self.volume_label.pack(side='right', padx=5)
        
        self.market_cap_label = ttk.Label(info_frame, text="", style='Custom.TLabel')
        self.market_cap_label.pack(side='right', padx=5)

    def setup_charts_panel(self):
        charts_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        charts_frame.pack(expand=True, fill='both', pady=5)
        
        # Crear figura principal
        self.fig = Figure(figsize=(14, 8), facecolor='#1c1c1c')
        
        # Subplot para precio
        self.ax_price = self.fig.add_subplot(311)
        self.ax_price.set_facecolor('#1c1c1c')
        self.ax_price.tick_params(colors='white')
        
        # Subplot para volumen
        self.ax_volume = self.fig.add_subplot(312)
        self.ax_volume.set_facecolor('#1c1c1c')
        self.ax_volume.tick_params(colors='white')
        
        # Subplot para indicadores
        self.ax_indicators = self.fig.add_subplot(313)
        self.ax_indicators.set_facecolor('#1c1c1c')
        self.ax_indicators.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

    def setup_indicators_panel(self):
        indicators_frame = ttk.LabelFrame(self.main_frame, text="Indicadores Técnicos", 
                                        style='Custom.TFrame')
        indicators_frame.pack(fill='x', pady=5)
        
        # Grid de indicadores
        self.indicators_grid = ttk.Frame(indicators_frame, style='Custom.TFrame')
        self.indicators_grid.pack(fill='x', padx=5, pady=5)
        
        # Labels para indicadores
        self.indicator_labels = {
            'RSI': ttk.Label(self.indicators_grid, text="RSI: --", style='Custom.TLabel'),
            'MACD': ttk.Label(self.indicators_grid, text="MACD: --", style='Custom.TLabel'),
            'BB': ttk.Label(self.indicators_grid, text="Bandas Bollinger: --", style='Custom.TLabel'),
            'SMA': ttk.Label(self.indicators_grid, text="SMA(20): --", style='Custom.TLabel'),
            'EMA': ttk.Label(self.indicators_grid, text="EMA(20): --", style='Custom.TLabel'),
            'VWAP': ttk.Label(self.indicators_grid, text="VWAP: --", style='Custom.TLabel')
        }
        
        # Organizar grid
        row = 0
        col = 0
        for label in self.indicator_labels.values():
            label.grid(row=row, column=col, padx=5, pady=2)
            col += 1
            if col > 2:
                col = 0
                row += 1

    def setup_alerts_panel(self):
        alerts_frame = ttk.LabelFrame(self.main_frame, text="Alertas y Señales", 
                                    style='Custom.TFrame')
        alerts_frame.pack(fill='x', pady=5)
        
        self.alerts_text = tk.Text(alerts_frame, height=4, bg='#1c1c1c', fg='white',
                                 wrap=tk.WORD)
        self.alerts_text.pack(fill='x', padx=5, pady=5)

    def set_symbol(self, symbol):
        """Establecer símbolo a analizar"""
        self.symbol = symbol
        self.symbol_var.set(symbol)
        self.start_tracking()

    def start_tracking(self):
        """Iniciar seguimiento en tiempo real"""
        self.symbol = self.symbol_var.get()
        if not self.symbol:
            return
        
        self.symbol_label.config(text=f"Símbolo: {self.symbol}")
        
        # Iniciar thread de actualización
        if hasattr(self, 'update_thread') and self.update_thread.is_alive():
            return
        
        self.update_thread = threading.Thread(target=self._update_data, daemon=True)
        self.update_thread.start()

    def _update_data(self):
        """Actualizar datos en tiempo real"""
        while self.running:
            try:
                # Obtener datos
                stock = yf.Ticker(self.symbol)
                self.data = stock.history(period='1d', interval='1m')
                
                if not self.data.empty:
                    # Calcular indicadores técnicos
                    self.calculate_technical_indicators()
                    
                    # Actualizar UI
                    self.window.after(0, self.update_display)
                    
                    # Verificar alertas
                    self.check_alerts()
                    
            except Exception as e:
                print(f"Error actualizando datos: {e}")
                
            time.sleep(self.update_interval)

    def calculate_technical_indicators(self):
        """Calcular indicadores técnicos"""
        try:
            # RSI
            rsi = RSIIndicator(self.data['Close'])
            self.technical_indicators['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(self.data['Close'])
            self.technical_indicators['MACD'] = macd.macd()
            
            # Bollinger Bands
            bb = BollingerBands(self.data['Close'])
            self.technical_indicators['BB_upper'] = bb.bollinger_hband()
            self.technical_indicators['BB_lower'] = bb.bollinger_lband()
            
            # SMA y EMA
            self.technical_indicators['SMA'] = SMAIndicator(self.data['Close'], 20).sma_indicator()
            self.technical_indicators['EMA'] = EMAIndicator(self.data['Close'], 20).ema_indicator()
            
            # VWAP
            vwap = VolumeWeightedAveragePrice(high=self.data['High'],
                                             low=self.data['Low'],
                                             close=self.data['Close'],
                                             volume=self.data['Volume'])
            self.technical_indicators['VWAP'] = vwap.volume_weighted_average_price()
            
        except Exception as e:
            print(f"Error calculando indicadores: {e}")

    def update_display(self):
        """Actualizar visualización"""
        try:
            if self.data.empty:
                return
                
            # Actualizar precio actual y cambio
            current_price = self.data['Close'].iloc[-1]
            change = (current_price - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
            
            self.price_label.config(text=f"Precio: ${current_price:.2f}")
            self.change_label.config(
                text=f"Cambio: {'↑' if change >= 0 else '↓'}{abs(change):.2f}%",
                foreground='#00ff00' if change >= 0 else '#ff4444'
            )
            
            # Actualizar volumen y market cap
            volume = self.data['Volume'].iloc[-1]
            self.volume_label.config(text=f"Volumen: {volume:,}")
            
            # Actualizar gráficos
            self.update_charts()
            
            # Actualizar indicadores
            self.update_indicators()
            
        except Exception as e:
            print(f"Error actualizando display: {e}")

    def update_charts(self):
        """Actualizar gráficos"""
        try:
            # Limpiar gráficos
            self.ax_price.clear()
            self.ax_volume.clear()
            self.ax_indicators.clear()
            
            # Configurar colores y estilo
            self.ax_price.set_facecolor('#1c1c1c')
            self.ax_volume.set_facecolor('#1c1c1c')
            self.ax_indicators.set_facecolor('#1c1c1c')
            
            # Graficar precio y Bollinger Bands
            self.ax_price.plot(self.data.index, self.data['Close'], color='white', label='Precio')
            self.ax_price.plot(self.data.index, self.technical_indicators['BB_upper'], 
                             'g--', alpha=0.5, label='BB Superior')
            self.ax_price.plot(self.data.index, self.technical_indicators['BB_lower'], 
                             'r--', alpha=0.5, label='BB Inferior')
            
            # Graficar volumen
            self.ax_volume.bar(self.data.index, self.data['Volume'], color='blue', alpha=0.5)
            
            # Graficar RSI
            self.ax_indicators.plot(self.data.index, self.technical_indicators['RSI'], 
                                  color='yellow', label='RSI')
            self.ax_indicators.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            self.ax_indicators.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            
            # Configurar leyendas y títulos
            self.ax_price.legend()
            self.ax_indicators.legend()
            
            # Ajustar diseño
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")

    def update_indicators(self):
        """Actualizar panel de indicadores"""
        try:
            last_values = {
                'RSI': self.technical_indicators['RSI'].iloc[-1],
                'MACD': self.technical_indicators['MACD'].iloc[-1],
                'BB': f"Superior: {self.technical_indicators['BB_upper'].iloc[-1]:.2f}, "
                      f"Inferior: {self.technical_indicators['BB_lower'].iloc[-1]:.2f}",
                'SMA': self.technical_indicators['SMA'].iloc[-1],
                'EMA': self.technical_indicators['EMA'].iloc[-1],
                'VWAP': self.technical_indicators['VWAP'].iloc[-1]
            }
            
            for indicator, label in self.indicator_labels.items():
                value = last_values[indicator]
                if isinstance(value, float):
                    label.config(text=f"{indicator}: {value:.2f}")
                else:
                    label.config(text=f"{indicator}: {value}")
                    
        except Exception as e:
            print(f"Error actualizando indicadores: {e}")

    def check_alerts(self):
        """Verificar y generar alertas"""
        try:
            alerts = []
            
            # RSI sobrecompra/sobreventa
            rsi = self.technical_indicators['RSI'].iloc[-1]
            if rsi > 70:
                alerts.append(f"RSI indica sobrecompra ({rsi:.2f})")
            elif rsi < 30:
                alerts.append(f"RSI indica sobreventa ({rsi:.2f})")
            
            # Cruce de Bandas Bollinger
            price = self.data['Close'].iloc[-1]
            bb_upper = self.technical_indicators['BB_upper'].iloc[-1]
            bb_lower = self.technical_indicators['BB_lower'].iloc[-1]
            
            if price > bb_upper:
                alerts.append("Precio por encima de la banda superior de Bollinger")
            elif price < bb_lower:
                alerts.append("Precio por debajo de la banda inferior de Bollinger")
            
            # Actualizar panel de alertas
            if alerts:
                self.alerts_text.delete('1.0', tk.END)
                for alert in alerts:
                    self.alerts_text.insert(tk.END, f"⚠ {alert}\n")
                    
        except Exception as e:
            print(f"Error verificando alertas: {e}")

    def on_closing(self):
        """Manejar cierre de ventana"""
        self.running = False
        self.window.destroy()

    def run(self):
        """Iniciar la ventana"""
        if not self.window.winfo_exists():
            return
        self.window.mainloop() 