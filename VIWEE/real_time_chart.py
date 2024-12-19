import tkinter as tk
from tkinter import ttk
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import pandas as pd

class RealTimeWindow:
    def __init__(self):
        # Crear nueva ventana
        self.window = tk.Tk()
        self.window.title("Análisis en Tiempo Real")
        self.window.geometry("1200x800")
        self.window.configure(bg='#2b2b2b')
        
        # Variables de control
        self.running = True
        self.symbol = None
        self.data = None
        
        # Configurar estilo
        self.style = ttk.Style()
        self.style.configure('Custom.TFrame', background='#2b2b2b')
        self.style.configure('Custom.TLabel', 
                           background='#2b2b2b', 
                           foreground='white',
                           font=('Helvetica', 12))
        
        self._setup_gui()
        
    def _setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.window, style='Custom.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Frame superior para entrada y controles
        self.control_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.control_frame.pack(fill='x', pady=5)
        
        # Entrada de símbolo
        ttk.Label(self.control_frame, 
                 text="Símbolo:", 
                 style='Custom.TLabel').pack(side='left', padx=5)
        
        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Entry(self.control_frame, 
                                    textvariable=self.symbol_var,
                                    width=10)
        self.symbol_entry.pack(side='left', padx=5)
        
        # Botón de inicio
        self.start_button = ttk.Button(self.control_frame,
                                     text="Iniciar Seguimiento",
                                     command=self.start_tracking)
        self.start_button.pack(side='left', padx=5)
        
        # Panel de información
        self.info_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.info_frame.pack(fill='x', pady=5)
        
        # Labels para información
        self.price_label = ttk.Label(self.info_frame, 
                                   text="Precio Actual: --",
                                   style='Custom.TLabel')
        self.price_label.pack(side='left', padx=10)
        
        self.change_label = ttk.Label(self.info_frame,
                                    text="Cambio: --",
                                    style='Custom.TLabel')
        self.change_label.pack(side='left', padx=10)
        
        self.prediction_label = ttk.Label(self.info_frame,
                                        text="Predicción: --",
                                        style='Custom.TLabel')
        self.prediction_label.pack(side='left', padx=10)
        
        # Frame para gráficos
        self.chart_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.chart_frame.pack(expand=True, fill='both')
        
        # Configurar gráficos
        self.setup_charts()
        
    def setup_charts(self):
        self.fig = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        
        # Gráfico de precios
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_facecolor('#2b2b2b')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.2)
        
        # Gráfico de predicción
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.2)
        
        # Configurar canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')
        
    def start_tracking(self):
        symbol = self.symbol_var.get().upper()
        if not symbol:
            return
            
        self.symbol = symbol
        self.running = True
        
        # Iniciar thread de actualización
        threading.Thread(target=self.update_data, daemon=True).start()
        
    def update_data(self):
        while self.running:
            try:
                # Obtener datos
                stock = yf.Ticker(self.symbol)
                new_data = stock.history(period='1d', interval='1m')
                
                if not new_data.empty:
                    self.data = new_data
                    self.update_charts()
                    self.update_info()
                    
            except Exception as e:
                print(f"Error actualizando datos: {e}")
                
            time.sleep(60)  # Actualizar cada minuto
            
    def update_charts(self):
        if self.data is None:
            return
            
        try:
            # Limpiar gráficos
            self.ax1.clear()
            self.ax2.clear()
            
            # Configurar colores y estilo
            self.ax1.set_facecolor('#2b2b2b')
            self.ax2.set_facecolor('#2b2b2b')
            self.ax1.grid(True, alpha=0.2)
            self.ax2.grid(True, alpha=0.2)
            
            # Graficar precio
            self.ax1.plot(self.data.index, self.data['Close'], 
                         color='white', label='Precio')
            
            # Calcular y graficar tendencia
            x = np.arange(len(self.data))
            z = np.polyfit(x, self.data['Close'], 1)
            p = np.poly1d(z)
            
            # Extender tendencia
            future_x = np.arange(len(self.data), len(self.data) + 30)
            future_trend = p(future_x)
            
            # Color según tendencia
            trend_color = 'lime' if z[0] > 0 else 'red'
            
            # Graficar proyección
            all_x = np.concatenate([x, future_x])
            all_y = np.concatenate([self.data['Close'], future_trend])
            self.ax1.plot(all_x, all_y, color=trend_color, 
                         linestyle='--', label='Proyección')
            
            # Configurar gráficos
            self.ax1.set_title(f'{self.symbol} - Tiempo Real', color='white')
            self.ax1.legend(facecolor='#2b2b2b', labelcolor='white')
            
            # Graficar indicadores
            self.ax2.plot(self.data.index, self.data['Close'].pct_change(),
                         color='yellow', label='Cambio %')
            self.ax2.axhline(y=0, color='white', linestyle='-', alpha=0.2)
            
            self.ax2.set_title('Cambio Porcentual', color='white')
            self.ax2.legend(facecolor='#2b2b2b', labelcolor='white')
            
            # Actualizar canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")
            
    def update_info(self):
        if self.data is None:
            return
            
        try:
            current_price = self.data['Close'].iloc[-1]
            change = self.data['Close'].pct_change().iloc[-1] * 100
            
            # Predecir próximo valor
            x = np.arange(len(self.data))
            z = np.polyfit(x, self.data['Close'], 1)
            p = np.poly1d(z)
            next_price = p(len(self.data))
            
            # Actualizar labels
            self.price_label.config(
                text=f"Precio Actual: ${current_price:.2f}")
            
            change_text = f"Cambio: {'↑' if change > 0 else '↓'} {abs(change):.2f}%"
            self.change_label.config(
                text=change_text,
                foreground='lime' if change > 0 else 'red')
            
            self.prediction_label.config(
                text=f"Predicción: ${next_price:.2f}")
            
        except Exception as e:
            print(f"Error actualizando información: {e}")
            
    def run(self):
        self.window.mainloop()
        
    def on_closing(self):
        self.running = False
        self.window.destroy()

if __name__ == "__main__":
    app = RealTimeWindow()
    app.window.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.run()
