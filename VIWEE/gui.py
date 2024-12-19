import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import mplfinance as mpf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.figure import Figure
import seaborn as sns
from PIL import Image, ImageTk
import requests
from io import BytesIO
import json
import locale
from ayud_fin import AsistenteFinanciero, PerfilInversion
from real_time_window import RealTimeWindow
from prediction_window import PredictionWindow

# Configurar locale para formato de números
locale.setlocale(locale.LC_ALL, '')

class ModernStockAnalyzerGUI:
    def __init__(self):
        self.setup_main_window()
        self.setup_styles()
        self.initialize_variables()
        self.create_menu()
        self.setup_tabs()
        self.setup_input_frame()
        
        # Modificar la inicialización del predictor
        self.predictor = None
        try:
            from New_Text_Document import AdvancedStockPredictor
            self.predictor = AdvancedStockPredictor()
            print("Predictor inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar el predictor: {str(e)}")
            self.show_error("Error: No se pudo cargar el predictor correctamente")
        
        self.asistente = AsistenteFinanciero()
        
        # Añadir atributo para la ventana de tiempo real
        self.real_time_window = None
        
        self.setup_menu()
        self.setup_prediction_button()
    
    def setup_main_window(self):
        """Configuración de la ventana principal"""
        self.root = tk.Tk()
        self.root.title("Análisis Profesional de Mercados Financieros")
        self.root.state('zoomed')
        
        # Configurar tema oscuro
        self.root.configure(bg='#2b2b2b')
        
        # Icono de la aplicación
        try:
            icon_path = "assets/icon.ico"
            self.root.iconbitmap(icon_path)
        except:
            pass
    
    def setup_styles(self):
        """Configurar estilos personalizados"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores
        style.configure('.',
            background='#2b2b2b',
            foreground='white',
            fieldbackground='#363636')
        
        # Estilo para pestañas
        style.configure('Custom.TNotebook.Tab',
            padding=[12, 4],
            background='#363636',
            foreground='white')
        
        # Estilo para botones
        style.configure('Action.TButton',
            padding=10,
            background='#0066cc',
            foreground='white')
    
    def initialize_variables(self):
        """Inicializar variables"""
        self.current_symbol = None
        self.technical_data = {}
        self.fundamental_data = {}
        self.prediction_data = {}
        
        # Añadir variables para la vista general
        self.company_name_var = tk.StringVar()
        self.sector_var = tk.StringVar()
        self.industry_var = tk.StringVar()
        self.market_cap_var = tk.StringVar()
    
    def create_menu(self):
        """Crear barra de menú"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Guardar Análisis", command=self.save_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)
        
        # Menú Herramientas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Preferencias", command=self.show_preferences)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Documentación", command=self.show_documentation)
        help_menu.add_command(label="Acerca de", command=self.show_about)
    
    def setup_tabs(self):
        """Configurar pestañas principales"""
        self.tab_control = ttk.Notebook(self.root)
        
        # Crear pestañas
        self.overview_tab = ttk.Frame(self.tab_control)
        self.technical_tab = ttk.Frame(self.tab_control)
        self.fundamental_tab = ttk.Frame(self.tab_control)
        self.prediction_tab = ttk.Frame(self.tab_control)
        self.advisor_tab = ttk.Frame(self.tab_control)
        self.realtime_tab = ttk.Frame(self.tab_control)
        self.estimation_tab = ttk.Frame(self.tab_control)  # Nueva pestaña
        
        # Añadir pestañas al control
        self.tab_control.add(self.overview_tab, text='Vista General')
        self.tab_control.add(self.technical_tab, text='Análisis Técnico')
        self.tab_control.add(self.fundamental_tab, text='Análisis Fundamental')
        self.tab_control.add(self.prediction_tab, text='Predicciones')
        self.tab_control.add(self.advisor_tab, text='Ayudante de Finanzas')
        self.tab_control.add(self.realtime_tab, text='Tiempo Real')
        self.tab_control.add(self.estimation_tab, text='Estimaciones')  # Nueva pestaña
        
        self.tab_control.pack(expand=True, fill='both')
        
        # Configurar contenido de cada pestaña
        self.setup_overview_tab()
        self.setup_technical_tab()
        self.setup_fundamental_tab()
        self.setup_prediction_tab()
        self.setup_advisor_tab()
        self.setup_realtime_tab()
        self.setup_estimation_tab()  # Nueva función
    
    def setup_overview_tab(self):
        """Configurar pestaña de vista general"""
        # Frame principal con grid
        main_frame = ttk.Frame(self.overview_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Información general
        info_frame = ttk.LabelFrame(main_frame, text="Información General")
        info_frame.pack(fill='x', padx=5, pady=5)
        
        # Widgets para información básica
        self.company_name_label = ttk.Label(info_frame, text="")
        self.company_name_label.pack(pady=5)
        
        # Gráfico principal
        self.overview_chart_frame = ttk.Frame(main_frame)
        self.overview_chart_frame.pack(expand=True, fill='both', pady=5)
        
        # Resumen de métricas
        metrics_frame = ttk.LabelFrame(main_frame, text="Métricas Clave")
        metrics_frame.pack(fill='x', padx=5, pady=5)
        
        # Grid de métricas
        self.setup_metrics_grid(metrics_frame)
    
    def setup_metrics_grid(self, parent):
        """Configurar grid de métricas principales"""
        metrics = [
            ("Precio Actual", "current_price"),
            ("Cambio %", "price_change"),
            ("Volumen", "volume"),
            ("Cap. Mercado", "market_cap"),
            ("P/E Ratio", "pe_ratio"),
            ("Beta", "beta")
        ]
        
        for i, (label, var_name) in enumerate(metrics):
            ttk.Label(parent, text=label).grid(row=i//3, column=(i%3)*2, padx=5, pady=2)
            setattr(self, f"{var_name}_label", ttk.Label(parent, text="-"))
            getattr(self, f"{var_name}_label").grid(row=i//3, column=(i%3)*2+1, padx=5, pady=2)
    
    def setup_technical_tab(self):
        """Configurar pestaña de análisis técnico"""
        # Frame principal
        main_frame = ttk.Frame(self.technical_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Frame de controles
        controls_frame = ttk.LabelFrame(main_frame, text="Controles")
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        # Periodo
        ttk.Label(controls_frame, text="Periodo:").pack(side='left', padx=5)
        self.period_var = tk.StringVar(value='1y')
        period_combo = ttk.Combobox(controls_frame, 
                                  values=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                                  textvariable=self.period_var,
                                  width=10)
        period_combo.pack(side='left', padx=5)
        
        # Intervalo
        ttk.Label(controls_frame, text="Intervalo:").pack(side='left', padx=5)
        self.interval_var = tk.StringVar(value='1d')
        interval_combo = ttk.Combobox(controls_frame,
                                    values=['1d', '5d', '1wk', '1mo'],
                                    textvariable=self.interval_var,
                                    width=10)
        interval_combo.pack(side='left', padx=5)
        
        # Botón actualizar
        ttk.Button(controls_frame, text="Actualizar Gráfico", 
                  command=self.update_technical_chart).pack(side='left', padx=5)
        
        # Frame para gráfico
        self.chart_frame = ttk.Frame(main_frame)
        self.chart_frame.pack(expand=True, fill='both', pady=5)
        
        # Frame para indicadores
        indicators_frame = ttk.LabelFrame(main_frame, text="Indicadores Técnicos")
        indicators_frame.pack(fill='x', padx=5, pady=5)
        
        self.technical_text = tk.Text(indicators_frame, height=10)
        self.technical_text.pack(expand=True, fill='both', padx=5, pady=5)

    def update_technical_chart(self):
        """Actualizar gráfico técnico"""
        if not self.current_symbol:
            self.show_error("Primero seleccione un símbolo")
            return
            
        try:
            # Limpiar frame actual
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            
            # Obtener datos
            stock = yf.Ticker(self.current_symbol)
            data = stock.history(period=self.period_var.get(),
                               interval=self.interval_var.get())
            
            # Crear figura
            fig = Figure(figsize=(12, 8), dpi=100)
            ax1 = fig.add_subplot(211)  # Gráfico de precios
            ax2 = fig.add_subplot(212)  # Gráfico de volumen
            
            # Configurar gráfico de precios
            ax1.plot(data.index, data['Close'], label='Precio de Cierre')
            ax1.plot(data.index, data['Close'].rolling(window=20).mean(), 
                    label='SMA 20', alpha=0.7)
            ax1.plot(data.index, data['Close'].rolling(window=50).mean(),
                    label='SMA 50', alpha=0.7)
            ax1.set_title(f'Análisis Técnico - {self.current_symbol}')
            ax1.legend()
            ax1.grid(True)
            
            # Configurar gráfico de volumen
            ax2.bar(data.index, data['Volume'], alpha=0.7)
            ax2.set_title('Volumen')
            ax2.grid(True)
            
            # Ajustar diseño
            fig.tight_layout()
            
            # Mostrar gráfico
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill='both')
            
            # Actualizar indicadores
            self.update_technical_indicators(data)
            
        except Exception as e:
            self.show_error(f"Error al actualizar gráfico: {str(e)}")

    def update_technical_indicators(self, data):
        """Actualizar indicadores técnicos"""
        try:
            # Calcular indicadores
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            rsi = self.calculate_rsi(data['Close']).iloc[-1]
            
            # Calcular tendencia
            trend = "Alcista" if sma_20 > sma_50 else "Bajista"
            
            # Calcular volatilidad
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Anualizada
            
            # Mostrar resultados
            self.technical_text.delete('1.0', tk.END)
            self.technical_text.insert(tk.END,
                f"=== Indicadores Técnicos ===\n\n"
                f"Último Precio: ${data['Close'].iloc[-1]:.2f}\n"
                f"SMA 20: ${sma_20:.2f}\n"
                f"SMA 50: ${sma_50:.2f}\n"
                f"RSI (14): {rsi:.2f}\n"
                f"Tendencia: {trend}\n"
                f"Volatilidad Anual: {volatility:.2f}%\n"
                f"Volumen Promedio (10d): {data['Volume'].rolling(10).mean().iloc[-1]:,.0f}\n"
            )
            
        except Exception as e:
            self.show_error(f"Error al calcular indicadores: {str(e)}")

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def setup_fundamental_tab(self):
        """Configurar pestaña de análisis fundamental"""
        # Canvas con scroll
        canvas = tk.Canvas(self.fundamental_tab)
        scrollbar = ttk.Scrollbar(self.fundamental_tab, orient="vertical", 
                                command=canvas.yview)
        
        self.fundamental_frame = ttk.Frame(canvas)
        self.fundamental_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.fundamental_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configurar scroll con mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Secciones del análisis fundamental
        self.setup_fundamental_sections()
    
    def setup_fundamental_sections(self):
        """Configurar secciones del análisis fundamental"""
        sections = [
            ("Información General", self.general_info_fields),
            ("Métricas de Valoración", self.valuation_fields),
            ("Métricas Financieras", self.financial_fields),
            ("Dividendos", self.dividend_fields),
            ("Análisis de Crecimiento", self.growth_fields),
            ("Ratios de Eficiencia", self.efficiency_fields)
        ]
        
        for title, fields in sections:
            frame = ttk.LabelFrame(self.fundamental_frame, text=title)
            frame.pack(fill='x', padx=5, pady=5)
            
            for label, key in fields():
                row = ttk.Frame(frame)
                row.pack(fill='x', padx=5, pady=2)
                ttk.Label(row, text=f"{label}:").pack(side='left', padx=5)
                label_widget = ttk.Label(row, text="-")
                label_widget.pack(side='left', padx=5)
                setattr(self, f"{key}_label", label_widget)
    
    def general_info_fields(self):
        return [
            ("Nombre", "longName"),
            ("Sector", "sector"),
            ("Industria", "industry"),
            ("País", "country"),
            ("Empleados", "fullTimeEmployees"),
            ("Descripción", "longBusinessSummary")
        ]
    
    def valuation_fields(self):
        return [
            ("Capitalización de Mercado", "marketCap"),
            ("Valor Empresa", "enterpriseValue"),
            ("P/E Ratio (TTM)", "trailingPE"),
            ("P/E Ratio Forward", "forwardPE"),
            ("PEG Ratio", "pegRatio"),
            ("Precio/Ventas", "priceToSalesTrailing12Months"),
            ("Precio/Valor Libro", "priceToBook"),
            ("Valor Empresa/EBITDA", "enterpriseToEbitda")
        ]
    
    def financial_fields(self):
        return [
            ("Ingresos (TTM)", "totalRevenue"),
            ("Beneficio Bruto", "grossProfits"),
            ("EBITDA", "ebitda"),
            ("Margen Neto", "profitMargins"),
            ("ROE", "returnOnEquity"),
            ("ROA", "returnOnAssets"),
            ("Deuda Total", "totalDebt"),
            ("Ratio Deuda/Capital", "debtToEquity"),
            ("Efectivo", "totalCash"),
            ("Flujo de Caja Libre", "freeCashflow")
        ]
    
    def dividend_fields(self):
        return [
            ("Rendimiento Dividendo", "dividendYield"),
            ("Ratio de Pago", "payoutRatio"),
            ("Dividendo por Acción", "dividendRate"),
            ("Fecha Ex-Dividendo", "exDividendDate")
        ]
    
    def growth_fields(self):
        return [
            ("Crecimiento Ingresos (YoY)", "revenueGrowth"),
            ("Crecimiento Beneficios (YoY)", "earningsGrowth"),
            ("Crecimiento EPS (YoY)", "earningsQuarterlyGrowth")
        ]
    
    def efficiency_fields(self):
        return [
            ("Rotación de Activos", "assetTurnover"),
            ("Rotación de Inventario", "inventoryTurnover"),
            ("Días Cobro", "daysReceivable"),
            ("Días Pago", "daysPayable")
        ]
    
    def setup_prediction_tab(self):
        """Configurar pestaña de predicciones"""
        # Frame principal
        main_frame = ttk.Frame(self.prediction_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Configuración de predicción
        config_frame = ttk.LabelFrame(main_frame, text="Configuración de Predicción")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Frame para los controles
        controls_frame = ttk.Frame(config_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        # Parámetros
        ttk.Label(controls_frame, text="Horizonte de predicción (días):").pack(side='left', padx=5)
        self.days_var = tk.StringVar(value="30")
        ttk.Entry(controls_frame, textvariable=self.days_var, width=10).pack(side='left', padx=5)
        
        # Botón de actualización
        ttk.Button(controls_frame, 
                  text="Actualizar Predicción",
                  command=lambda: self.update_predictions(self.current_symbol),
                  style='Action.TButton').pack(side='left', padx=20)
        
        # Gráfico de predicci��n
        self.prediction_chart_frame = ttk.Frame(main_frame)
        self.prediction_chart_frame.pack(expand=True, fill='both', pady=5)
        
        # Resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de la Predicción")
        results_frame.pack(fill='x', padx=5, pady=5)
        
        self.prediction_text = tk.Text(results_frame, height=10)
        self.prediction_text.pack(expand=True, fill='both', padx=5, pady=5)
    
    def setup_input_frame(self):
        """Configurar frame principal de entrada"""
        # Frame principal de entrada
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Frame de búsqueda
        search_frame = ttk.LabelFrame(input_frame, text="Búsqueda de Acciones")
        search_frame.pack(fill='x', padx=5, pady=5)
        
        # Símbolo
        symbol_frame = ttk.Frame(search_frame)
        symbol_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(symbol_frame, text="Símbolo:").pack(side='left', padx=5)
        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Entry(symbol_frame, textvariable=self.symbol_var, width=15)
        self.symbol_entry.pack(side='left', padx=5)
        
        # Ejemplos de símbolos
        ttk.Label(symbol_frame, 
                 text="(Ejemplos: AAPL, MSFT, GOOGL, AMZN)",
                 foreground='gray').pack(side='left', padx=5)
        
        # Frame de opciones
        options_frame = ttk.Frame(search_frame)
        options_frame.pack(fill='x', padx=5, pady=5)
        
        # Periodo
        ttk.Label(options_frame, text="Periodo:").pack(side='left', padx=5)
        self.period_var = tk.StringVar(value='1y')
        period_combo = ttk.Combobox(options_frame, 
                                  values=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                                  textvariable=self.period_var,
                                  width=8)
        period_combo.pack(side='left', padx=5)
        
        # Intervalo
        ttk.Label(options_frame, text="Intervalo:").pack(side='left', padx=5)
        self.interval_var = tk.StringVar(value='1d')
        interval_combo = ttk.Combobox(options_frame,
                                    values=['1d', '5d', '1wk', '1mo'],
                                    textvariable=self.interval_var,
                                    width=8)
        interval_combo.pack(side='left', padx=5)
        
        # Botón de análisis
        analyze_button = ttk.Button(options_frame, 
                                  text="Analizar",
                                  command=self.analyze_stock,
                                  style='Action.TButton')
        analyze_button.pack(side='left', padx=20)
        
        # Estado del análisis
        self.status_var = tk.StringVar()
        status_label = ttk.Label(options_frame, 
                               textvariable=self.status_var,
                               foreground='gray')
        status_label.pack(side='left', padx=5)
        
        # Frame de información rápida
        quick_info_frame = ttk.Frame(input_frame)
        quick_info_frame.pack(fill='x', padx=5, pady=5)
        
        # Labels para información rápida
        self.price_var = tk.StringVar()
        self.change_var = tk.StringVar()
        self.volume_var = tk.StringVar()
        
        ttk.Label(quick_info_frame, textvariable=self.price_var,
                 font=('Helvetica', 12, 'bold')).pack(side='left', padx=20)
        ttk.Label(quick_info_frame, textvariable=self.change_var,
                 font=('Helvetica', 12)).pack(side='left', padx=20)
        ttk.Label(quick_info_frame, textvariable=self.volume_var,
                 font=('Helvetica', 12)).pack(side='left', padx=20)
        
        # Separador
        ttk.Separator(input_frame, orient='horizontal').pack(fill='x', pady=5)

    def update_quick_info(self, stock_data):
        """Actualizar información rápida"""
        if stock_data is not None and not stock_data.empty:
            last_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2]
            change = last_price - prev_price
            change_pct = (change / prev_price) * 100
            volume = stock_data['Volume'].iloc[-1]
            
            self.price_var.set(f"Precio: ${last_price:.2f}")
            
            if change >= 0:
                self.change_var.set(f"▲ +${change:.2f} (+{change_pct:.2f}%)")
                # Cambiar color a verde
            else:
                self.change_var.set(f"▼ ${change:.2f} ({change_pct:.2f}%)")
                # Cambiar color a rojo
                
            self.volume_var.set(f"Vol: {volume:,.0f}")
            
        else:
            self.price_var.set("")
            self.change_var.set("")
            self.volume_var.set("")

    def analyze_stock(self):
        """Iniciar análisis de la acción"""
        symbol = self.symbol_var.get().upper()
        if not symbol:
            self.show_error("Por favor, ingrese un símbolo válido")
            return
        
        self.status_var.set("Analizando...")
        self.current_symbol = symbol
        
        # Iniciar análisis en thread separado
        self.analysis_thread = threading.Thread(
            target=self._perform_analysis,
            args=(symbol,),
            daemon=True
        )
        self.analysis_thread.start()
    
    def _perform_analysis(self, symbol):
        """Realizar análisis completo"""
        try:
            # Obtener datos
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Obtener datos históricos
            hist_data = stock.history(period=self.period_var.get(), 
                                    interval=self.interval_var.get())
            
            if hist_data.empty:
                raise ValueError("No se pudieron obtener datos históricos")
            
            # Actualizar cada sección en el hilo principal
            self.root.after(0, lambda: self.update_quick_info(hist_data))
            self.root.after(0, lambda: self.update_overview(info))
            self.root.after(0, lambda: self.update_technical_analysis(symbol))
            self.root.after(0, lambda: self.update_fundamental_analysis(info))
            self.root.after(0, lambda: self.update_predictions(symbol))
            self.root.after(0, lambda: self.generate_advice())
            
            self.root.after(0, lambda: self.status_var.set("Análisis completado"))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error en el análisis: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error en el análisis"))
    
    def format_number(self, value):
        """Formatear números para mejor visualización"""
        if isinstance(value, (int, float)):
            if value > 1e9:
                return f"{value/1e9:.2f}B"
            elif value > 1e6:
                return f"{value/1e6:.2f}M"
            elif value > 1e3:
                return f"{value/1e3:.2f}K"
            else:
                return f"{value:.2f}"
        return str(value)
    
    def run(self):
        """Iniciar aplicación"""
        self.root.mainloop()

    def save_analysis(self):
        """Guardar análisis actual"""
        if not self.current_symbol:
            self.show_error("No hay análisis para guardar")
            return
        
        try:
            # Crear diccionario con todos los datos del análisis
            analysis_data = {
                "symbol": self.current_symbol,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "technical_analysis": self.get_technical_data(),
                "fundamental_analysis": self.get_fundamental_data(),
                "predictions": self.get_prediction_data()
            }
            
            # Guardar en archivo
            filename = f"analysis_{self.current_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            
            messagebox.showinfo("Éxito", f"Análisis guardado en {filename}")
            
        except Exception as e:
            self.show_error(f"Error al guardar: {str(e)}")

    def show_preferences(self):
        """Mostrar ventana de preferencias"""
        preferences_window = tk.Toplevel(self.root)
        preferences_window.title("Preferencias")
        preferences_window.geometry("400x300")
        
        # Crear pestañas de preferencias
        pref_notebook = ttk.Notebook(preferences_window)
        pref_notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Pestaña General
        general_tab = ttk.Frame(pref_notebook)
        pref_notebook.add(general_tab, text='General')
        
        # Pestaña Gráficos
        charts_tab = ttk.Frame(pref_notebook)
        pref_notebook.add(charts_tab, text='Gráficos')
        
        # Añadir opciones
        ttk.Label(general_tab, text="Tema:").pack(padx=5, pady=5)
        theme_combo = ttk.Combobox(general_tab, values=['Claro', 'Oscuro'])
        theme_combo.pack(padx=5, pady=5)
        
        ttk.Label(charts_tab, text="Estilo de gráficos:").pack(padx=5, pady=5)
        chart_style_combo = ttk.Combobox(charts_tab, values=['Candlestick', 'OHLC', 'Line'])
        chart_style_combo.pack(padx=5, pady=5)

    def show_documentation(self):
        """Mostrar documentación"""
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentación")
        doc_window.geometry("600x400")
        
        text = tk.Text(doc_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(expand=True, fill='both')
        
        # Añadir contenido de documentación
        documentation = """
        === Análisis Profesional de Mercados Financieros ===
        
        1. Vista General
        - Muestra información general de la empresa
        - Resumen de precios y volumen
        - Noticias relevantes
        
        2. Análisis Técnico
        - Gráficos interactivos
        - Múltiples indicadores técnicos
        - Patrones de velas
        
        3. Análisis Fundamental
        - Métricas financieras
        - Ratios importantes
        - Análisis de crecimiento
        
        4. Predicciones
        - Modelo de IA avanzado
        - Múltiples horizontes temporales
        - Métricas de precisión
        """
        text.insert('1.0', documentation)
        text.config(state='disabled')

    def show_about(self):
        """Mostrar ventana Acerca de"""
        about_window = tk.Toplevel(self.root)
        about_window.title("Acerca de")
        about_window.geometry("400x300")
        
        about_text = """
        Análisis Profesional de Mercados Financieros
        Versión 1.0
        
        Desarrollado por: Didier Guisland
        
        Esta aplicación proporciona análisis avanzado
        de mercados financieros utilizando técnicas
        de machine learning y análisis técnico/fundamental.
        
        © 2024 Todos los derechos reservados
        """
        
        label = ttk.Label(about_window, text=about_text, justify='center')
        label.pack(expand=True, padx=20, pady=20)

    def get_technical_data(self):
        """Obtener datos del análisis técnico actual"""
        if not hasattr(self, 'technical_data'):
            return {}
        return self.technical_data

    def get_fundamental_data(self):
        """Obtener datos del análisis fundamental actual"""
        if not hasattr(self, 'fundamental_data'):
            return {}
        return self.fundamental_data

    def get_prediction_data(self):
        """Obtener datos de predicciones actuales"""
        if not hasattr(self, 'prediction_data'):
            return {}
        return self.prediction_data

    def show_loading_message(self):
        """Mostrar mensaje de carga"""
        if hasattr(self, 'technical_text'):
            self.technical_text.delete('1.0', tk.END)
            self.technical_text.insert('1.0', "Cargando análisis...")
        if hasattr(self, 'prediction_text'):
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert('1.0', "Cargando predicciones...")

    def show_error(self, message):
        """Mostrar mensaje de error"""
        messagebox.showerror("Error", message)

    def update_technical_analysis(self, symbol):
        """Actualizar análisis técnico"""
        try:
            # Obtener datos históricos
            stock = yf.Ticker(symbol)
            data = stock.history(period=self.period_var.get(), 
                               interval=self.interval_var.get())
            
            if data.empty:
                raise ValueError("No se pudieron obtener datos para este símbolo")
            
            # Calcular indicadores técnicos
            # SMA
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['SMA200'] = data['Close'].rolling(window=200).mean()
            
            # RSI
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (std * 2)
            data['BB_lower'] = data['BB_middle'] - (std * 2)
            
            # Actualizar gráfico técnico
            self.update_technical_chart()
            
            # Preparar análisis
            last_close = data['Close'].iloc[-1]
            sma20 = data['SMA20'].iloc[-1]
            sma50 = data['SMA50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            
            # Análisis de tendencia
            trend = "ALCISTA" if sma20 > sma50 else "BAJISTA"
            strength = "FUERTE" if abs(sma20 - sma50) / sma50 > 0.02 else "MODERADA"
            
            # RSI análisis
            rsi_signal = "SOBRECOMPRADO" if rsi > 70 else "SOBREVENDIDO" if rsi < 30 else "NEUTRAL"
            
            # MACD análisis
            macd_signal = "COMPRA" if macd > signal else "VENTA"
            
            # Actualizar texto de análisis
            self.technical_text.delete('1.0', tk.END)
            self.technical_text.insert(tk.END, 
                f"=== ANÁLISIS TÉCNICO DE {symbol} ===\n\n"
                f"Precio Actual: ${last_close:.2f}\n\n"
                f"TENDENCIA PRINCIPAL: {trend} ({strength})\n"
                f"- SMA 20: ${sma20:.2f}\n"
                f"- SMA 50: ${sma50:.2f}\n\n"
                f"INDICADORES:\n"
                f"- RSI (14): {rsi:.2f} - {rsi_signal}\n"
                f"- MACD: {macd_signal}\n"
                f"  * MACD Line: {macd:.3f}\n"
                f"  * Signal Line: {signal:.3f}\n\n"
                f"VOLATILIDAD:\n"
                f"- Bandas de Bollinger:\n"
                f"  * Superior: ${data['BB_upper'].iloc[-1]:.2f}\n"
                f"  * Media: ${data['BB_middle'].iloc[-1]:.2f}\n"
                f"  * Inferior: ${data['BB_lower'].iloc[-1]:.2f}\n\n"
                f"VOLUMEN:\n"
                f"- Último: {data['Volume'].iloc[-1]:,.0f}\n"
                f"- Promedio (20d): {data['Volume'].rolling(20).mean().iloc[-1]:,.0f}\n"
            )
            
            # Guardar datos para uso posterior
            self.technical_data = {
                'last_close': last_close,
                'sma20': sma20,
                'sma50': sma50,
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'trend': trend
            }
            
        except Exception as e:
            self.show_error(f"Error en análisis técnico: {str(e)}")
            self.technical_text.delete('1.0', tk.END)
            self.technical_text.insert('1.0', "Error al realizar el análisis técnico")

    def update_overview(self, info):
        """Actualizar vista general con información de la empresa"""
        try:
            # Actualizar información general
            self.company_name_var.set(info.get('longName', 'N/A'))
            self.sector_var.set(f"Sector: {info.get('sector', 'N/A')}")
            self.industry_var.set(f"Industria: {info.get('industry', 'N/A')}")
            
            # Formatear capitalización de mercado
            market_cap = info.get('marketCap', 0)
            market_cap_str = self.format_number(market_cap)
            self.market_cap_var.set(f"Capitalización: {market_cap_str}")
            
            # Actualizar métricas clave
            self.update_key_metrics(info)
            
            # Actualizar gráfico de resumen
            self.update_overview_chart()
            
            # Actualizar noticias
            self.update_news(info.get('symbol', ''))
            
            self.status_var.set("Análisis completado")
            
        except Exception as e:
            self.show_error(f"Error al actualizar vista general: {str(e)}")

    def update_key_metrics(self, info):
        """Actualizar métricas clave"""
        try:
            # Actualizar precio actual
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
            if hasattr(self, 'current_price_label'):
                self.current_price_label.config(text=f"${current_price}" if current_price != 'N/A' else '-')

            # Actualizar cambio porcentual
            change_pct = info.get('regularMarketChangePercent', 'N/A')
            if hasattr(self, 'price_change_label'):
                if change_pct != 'N/A':
                    change_text = f"{change_pct:+.2f}%" if change_pct != 'N/A' else '-'
                else:
                    change_text = '-'
                self.price_change_label.config(text=change_text)

            # Actualizar volumen
            volume = info.get('regularMarketVolume', 'N/A')
            if hasattr(self, 'volume_label'):
                self.volume_label.config(text=self.format_number(volume) if volume != 'N/A' else '-')

            # Actualizar capitalización de mercado
            market_cap = info.get('marketCap', 'N/A')
            if hasattr(self, 'market_cap_label'):
                self.market_cap_label.config(text=self.format_number(market_cap) if market_cap != 'N/A' else '-')

            # Actualizar P/E Ratio
            pe_ratio = info.get('trailingPE', 'N/A')
            if hasattr(self, 'pe_ratio_label'):
                self.pe_ratio_label.config(text=f"{pe_ratio:.2f}" if pe_ratio != 'N/A' else '-')

            # Actualizar Beta
            beta = info.get('beta', 'N/A')
            if hasattr(self, 'beta_label'):
                self.beta_label.config(text=f"{beta:.2f}" if beta != 'N/A' else '-')

        except Exception as e:
            self.show_error(f"Error al actualizar métricas clave: {str(e)}")
            # En caso de error, establecer todos los valores a '-'
            for label in ['current_price_label', 'price_change_label', 'volume_label', 
                         'market_cap_label', 'pe_ratio_label', 'beta_label']:
                if hasattr(self, label):
                    getattr(self, label).config(text='-')

    def update_news(self, symbol):
        """Actualizar noticias de la empresa"""
        try:
            # Obtener noticias
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if hasattr(self, 'news_text'):
                self.news_text.delete('1.0', tk.END)
                
                if not news:
                    self.news_text.insert('1.0', "No hay noticias disponibles")
                    return
                
                # Mostrar últimas 5 noticias
                for i, article in enumerate(news[:5], 1):
                    title = article.get('title', 'Sin título')
                    date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    date_str = date.strftime('%Y-%m-%d %H:%M')
                    
                    self.news_text.insert(tk.END, f"{i}. {title}\n")
                    self.news_text.insert(tk.END, f"   {date_str}\n\n")
                    
                    
        except Exception as e:
            if hasattr(self, 'news_text'):
                self.news_text.insert(tk.END, "Error al cargar noticias\n")

    def update_predictions(self, symbol):
        """Actualizar predicciones para el símbolo dado"""
        try:
            # Mostrar mensaje de carga
            if hasattr(self, 'prediction_text'):
                self.prediction_text.delete('1.0', tk.END)
                self.prediction_text.insert('1.0', "Calculando predicciones...\nEsto puede tardar varios minutos...")
                self.prediction_text.update()

            # Obtener datos históricos para predicción
            stock = yf.Ticker(symbol)
            data = stock.history(period='2y')
            
            if data.empty:
                raise ValueError("No hay suficientes datos para realizar predicciones")

            if self.predictor is not None:
                # Ejecutar predicción en un hilo separado
                def run_prediction():
                    try:
                        days = int(self.days_var.get())
                        prediction = self.predictor.predict(data, days)
                        
                        # Actualizar UI en el hilo principal
                        self.root.after(0, lambda: self.update_prediction_results(prediction, symbol, data))
                    except Exception as e:
                        self.root.after(0, lambda: self.show_prediction_error(str(e)))
                
                # Iniciar predicción en hilo separado
                threading.Thread(target=run_prediction, daemon=True).start()
            else:
                raise ImportError("Predictor no inicializado correctamente")

        except Exception as e:
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert('1.0', f"Error al generar predicciones: {str(e)}")
            self.prediction_data = {}

    def update_prediction_results(self, prediction, symbol, data):
        """Actualizar resultados de predicción en la UI"""
        try:
            self.prediction_text.delete('1.0', tk.END)
            self.prediction_text.insert(tk.END, 
                f"=== PREDICCIÓN PARA {symbol} ===\n\n"
                f"Horizonte: {self.days_var.get()} días\n"
                f"Último precio: ${data['Close'].iloc[-1]:.2f}\n"
                f"Precio predicho: ${prediction['predicted_price']:.2f}\n"
                f"Cambio esperado: {prediction['expected_change']:.2f}%\n\n"
                f"Confianza: {prediction['confidence']:.2f}%\n"
                f"Tendencia: {prediction['trend']}\n"
            )
            
            self.prediction_data = prediction
        except Exception as e:
            self.show_prediction_error(str(e))

    def show_prediction_error(self, error_message):
        """Mostrar error de predicción"""
        self.prediction_text.delete('1.0', tk.END)
        self.prediction_text.insert('1.0', f"Error en la predicción: {error_message}")
        self.prediction_data = {}

    def update_fundamental_analysis(self, info):
        """Actualizar análisis fundamental"""
        try:
            # Actualizar cada sección de análisis fundamental
            sections = {
                'general_info_fields': self.general_info_fields(),
                'valuation_fields': self.valuation_fields(),
                'financial_fields': self.financial_fields(),
                'dividend_fields': self.dividend_fields(),
                'growth_fields': self.growth_fields(),
                'efficiency_fields': self.efficiency_fields()
            }
            
            for section_fields in sections.values():
                for label, key in section_fields:
                    value = info.get(key, 'N/A')
                    
                    # Formatear valores numéricos
                    if isinstance(value, (int, float)):
                        if key in ['marketCap', 'enterpriseValue', 'totalRevenue', 'grossProfits', 
                                 'totalCash', 'totalDebt', 'freeCashflow']:
                            value = self.format_number(value)
                        elif key in ['dividendYield', 'profitMargins', 'returnOnEquity', 'returnOnAssets']:
                            value = f"{value * 100:.2f}%"
                        else:
                            value = f"{value:.2f}"
                    
                    # Formatear fechas
                    elif key == 'exDividendDate' and value != 'N/A':
                        try:
                            value = pd.to_datetime(value).strftime('%Y-%m-%d')
                        except:
                            value = 'N/A'
                    
                    # Actualizar label correspondiente
                    label_widget = getattr(self, f"{key}_label", None)
                    if label_widget:
                        label_widget.config(text=str(value))
            
            # Guardar datos para uso posterior
            self.fundamental_data = {
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'eps': info.get('trailingEps', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'profit_margin': info.get('profitMargins', 'N/A'),
                'revenue_growth': info.get('revenueGrowth', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'return_on_equity': info.get('returnOnEquity', 'N/A')
            }
            
        except Exception as e:
            self.show_error(f"Error al actualizar análisis fundamental: {str(e)}")
            # Limpiar todos los campos en caso de error
            for section_fields in sections.values():
                for _, key in section_fields:
                    label_widget = getattr(self, f"{key}_label", None)
                    if label_widget:
                        label_widget.config(text="-")

    def update_overview_chart(self):
        """Actualizar gráfico de vista general"""
        try:
            # Limpiar el frame del gráfico
            for widget in self.overview_chart_frame.winfo_children():
                widget.destroy()
            
            if not self.current_symbol:
                return
            
            # Obtener datos históricos
            stock = yf.Ticker(self.current_symbol)
            data = stock.history(period='1y')
            
            if data.empty:
                return
            
            # Crear figura
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Graficar precio de cierre
            ax.plot(data.index, data['Close'], label='Precio', color='#0066cc')
            
            # Añadir media móvil de 50 días
            sma_50 = data['Close'].rolling(window=50).mean()
            ax.plot(data.index, sma_50, label='SMA 50', color='#ff6600', alpha=0.7)
            
            # Configurar gráfico
            ax.set_title(f'Evolución del Precio - {self.current_symbol}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rotar etiquetas del eje x
            ax.tick_params(axis='x', rotation=45)
            
            # Ajustar márgenes
            fig.tight_layout()
            
            # Mostrar gráfico
            canvas = FigureCanvasTkAgg(fig, self.overview_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill='both')
            
        except Exception as e:
            self.show_error(f"Error al actualizar gráfico de vista general: {str(e)}")

    def setup_advisor_tab(self):
        """Configurar pestaña del ayudante de finanzas"""
        main_frame = ttk.Frame(self.advisor_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Frame para seleccionar días
        days_frame = ttk.LabelFrame(main_frame, text="Seleccionar Horizonte de Predicción")
        days_frame.pack(fill='x', padx=5, pady=5)
        
        # Spinbox para seleccionar días
        ttk.Label(days_frame, text="Días:").pack(side='left', padx=5)
        self.days_var = tk.IntVar(value=30)  # Valor por defecto
        days_spinbox = ttk.Spinbox(days_frame, from_=1, to=365, textvariable=self.days_var, width=5)
        days_spinbox.pack(side='left', padx=5)
        
        # Vincular el Spinbox a la función de actualización
        self.days_var.trace_add("write", lambda *args: self.update_advice())

        # Frame para recomendaciones
        recommendations_frame = ttk.LabelFrame(main_frame, text="Recomendaciones")
        recommendations_frame.pack(expand=True, fill='both', padx=5, pady=5)
        
        self.advisor_text = tk.Text(recommendations_frame, height=15)
        self.advisor_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Botón para generar recomendaciones
        ttk.Button(main_frame, text="Generar Recomendaciones", 
                   command=self.generate_advice).pack(pady=10)

    def update_advice(self):
        """Actualizar recomendaciones cuando cambian los días seleccionados"""
        if self.current_symbol:  # Asegurarse de que hay un símbolo seleccionado
            self.generate_advice()

    def generate_advice(self):
        """Generar recomendaciones basadas en el análisis"""
        try:
            # Obtener datos de análisis
            technical_data = self.get_technical_data()
            fundamental_data = self.get_fundamental_data()
            prediction_data = self.get_prediction_data()
            
            # Obtener recomendación del asistente
            dias = self.days_var.get()
            perfil = PerfilInversion.MODERADO  # Ahora PerfilInversion está definido
            
            recomendacion = self.asistente.analizar_recomendacion(
                technical_data,
                fundamental_data,
                prediction_data,
                dias,
                perfil
            )
            
            # Mostrar recomendaciones
            self.advisor_text.delete('1.0', tk.END)
            self.advisor_text.insert(tk.END, recomendacion)
        
        except Exception as e:
            self.show_error(f"Error al generar recomendaciones: {str(e)}")

    def setup_menu(self):
        """Configurar menú y barra de herramientas"""
        # Crear menú principal si no existe
        if not hasattr(self, 'menu_bar'):
            self.menu_bar = tk.Menu(self.root)
            self.root.config(menu=self.menu_bar)

        # Crear toolbar frame si no existe
        if not hasattr(self, 'toolbar_frame'):
            self.toolbar_frame = ttk.Frame(self.root)
            self.toolbar_frame.pack(fill='x', padx=5, pady=5)

        # Botón para análisis en tiempo real
        self.realtime_button = ttk.Button(
            self.toolbar_frame,
            text="Análisis en Tiempo Real",
            command=self.show_real_time_window
        )
        self.realtime_button.pack(side='left', padx=5)

        # Añadir botón de estimaciones junto al de tiempo real
        self.estimation_button = ttk.Button(
            self.toolbar_frame,
            text="Estimaciones",
            command=self.show_estimation_window
        )
        self.estimation_button.pack(side='left', padx=5)

    def show_real_time_window(self):
        """Abrir ventana de análisis en tiempo real"""
        try:
            # Verificar si hay un símbolo seleccionado
            if not self.current_symbol:
                messagebox.showwarning(
                    "Advertencia",
                    "Por favor, seleccione primero un símbolo para analizar."
                )
                return

            # Crear nueva instancia de la ventana de tiempo real
            real_time_app = RealTimeWindow()
            
            # Establecer el símbolo actual
            real_time_app.set_symbol(self.current_symbol)
            
            # Mantener una referencia para evitar que se cierre
            self.real_time_window = real_time_app
            
        except Exception as e:
            self.show_error(f"Error al abrir ventana de tiempo real: {str(e)}")

    def show_estimation_window(self):
        """Abrir ventana de estimaciones"""
        try:
            if not self.current_symbol:
                messagebox.showwarning(
                    "Advertencia",
                    "Por favor, seleccione primero un símbolo para analizar."
                )
                return

            # Crear nueva instancia de la ventana de predicción pasando el símbolo
            estimation_app = PredictionWindow(symbol=self.current_symbol)
            
            # Mantener una referencia
            self.estimation_window = estimation_app
            
        except Exception as e:
            self.show_error(f"Error al abrir ventana de estimaciones: {str(e)}")

    def setup_realtime_tab(self):
        """Configurar pestaña de análisis en tiempo real"""
        main_frame = ttk.Frame(self.realtime_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Frame superior para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Botón para abrir ventana de tiempo real
        ttk.Button(
            control_frame,
            text="Abrir Análisis en Tiempo Real",
            command=self.show_real_time_window
        ).pack(side='left', padx=5)
        
        # Frame para información
        info_frame = ttk.LabelFrame(main_frame, text="Información")
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Texto informativo
        info_text = tk.Text(info_frame, height=10, wrap=tk.WORD)
        info_text.pack(fill='both', expand=True, padx=5, pady=5)
        info_text.insert('1.0', """
El análisis en tiempo real proporciona:

• Gráfico de precios actualizado cada minuto
• Indicadores técnicos en tiempo real
• Predicciones de tendencia
• Alertas de cambios significativos
• Visualización de soporte y resistencia
• Análisis de momentum
• Señales de compra/venta

Para comenzar, seleccione un símbolo y haga clic en "Abrir Análisis en Tiempo Real"
        """)
        info_text.config(state='disabled')
        
        # Frame para últimas actualizaciones
        updates_frame = ttk.LabelFrame(main_frame, text="Últimas Actualizaciones")
        updates_frame.pack(fill='x', padx=5, pady=5)
        
        # Labels para información en tiempo real
        self.rt_price_var = tk.StringVar(value="Precio Actual: --")
        self.rt_change_var = tk.StringVar(value="Cambio: --")
        self.rt_volume_var = tk.StringVar(value="Volumen: --")
        
        ttk.Label(updates_frame, textvariable=self.rt_price_var).pack(side='left', padx=10)
        ttk.Label(updates_frame, textvariable=self.rt_change_var).pack(side='left', padx=10)
        ttk.Label(updates_frame, textvariable=self.rt_volume_var).pack(side='left', padx=10)

    def setup_prediction_button(self):
        """Añadir botón de predicción a la barra de herramientas existente"""
        if hasattr(self, 'toolbar_frame'):
            self.prediction_button = ttk.Button(
                self.toolbar_frame,
                text="Análisis Predictivo",
                command=self.show_prediction_window
            )
            self.prediction_button.pack(side='left', padx=5)

    def show_prediction_window(self):
        """Abrir ventana de predicción"""
        try:
            if not self.current_symbol:
                messagebox.showwarning(
                    "Advertencia",
                    "Por favor, seleccione primero un símbolo para analizar."
                )
                return

            # Crear nueva instancia de la ventana de predicción
            prediction_app = PredictionWindow()
            
            # Configurar el símbolo actual
            prediction_app.symbol_var.set(self.current_symbol)
            
            # Mantener una referencia
            self.prediction_window = prediction_app
            
        except Exception as e:
            self.show_error(f"Error al abrir ventana de predicción: {str(e)}")

    def setup_estimation_tab(self):
        """Configurar pestaña de estimaciones"""
        # Frame principal
        main_frame = ttk.Frame(self.estimation_tab)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Botón para abrir ventana de estimaciones
        ttk.Button(
            main_frame,
            text="Abrir Análisis de Estimaciones",
            command=self.show_estimation_window
        ).pack(pady=20)
        
        # Texto informativo
        info_text = tk.Text(main_frame, height=10, width=50)
        info_text.pack(pady=10)
        info_text.insert('1.0', """
El análisis de estimaciones proporciona:

• Proyecciones de precios a corto y largo plazo
• Análisis de tendencias futuras
• Indicadores predictivos avanzados
• Niveles de soporte y resistencia proyectados
• Análisis de patrones históricos
• Señales de trading anticipadas

Para comenzar, haga clic en "Abrir Análisis de Estimaciones"
        """)
        info_text.config(state='disabled')

if __name__ == "__main__":
    app = ModernStockAnalyzerGUI()
    app.run() 
    
    