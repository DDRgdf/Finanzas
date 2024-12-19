import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from scipy import stats
import ta  # Technical Analysis library

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerfilInversion(Enum):
    CONSERVADOR = "conservador"
    MODERADO = "moderado"
    AGRESIVO = "agresivo"

@dataclass
class MetricasRiesgo:
    volatilidad: float
    var_95: float  # Value at Risk al 95%
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

@dataclass
class AnalisisFundamental:
    pe_ratio: float
    precio_libro: float
    margen_beneficio: float
    roe: float
    roa: float
    deuda_capital: float
    crecimiento_ingresos: float

class AsistenteFinanciero:
    """
    Asistente financiero avanzado que proporciona análisis y recomendaciones personalizadas.
    Utiliza análisis técnico, fundamental y predictivo para generar insights accionables.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_levels = {
            PerfilInversion.CONSERVADOR: {
                'rsi_max': 65, 'rsi_min': 35, 
                'volatility_threshold': 0.15,
                'var_limit': 0.02,
                'max_drawdown_limit': 0.10
            },
            PerfilInversion.MODERADO: {
                'rsi_max': 70, 'rsi_min': 30,
                'volatility_threshold': 0.25,
                'var_limit': 0.035,
                'max_drawdown_limit': 0.15
            },
            PerfilInversion.AGRESIVO: {
                'rsi_max': 75, 'rsi_min': 25,
                'volatility_threshold': 0.35,
                'var_limit': 0.05,
                'max_drawdown_limit': 0.25
            }
        }
        self.market_sentiment = None
        self.macro_indicators = None

    def analizar_recomendacion(self, technical_data: dict, fundamental_data: dict, 
                             prediction_data: dict, dias: int, 
                             perfil: PerfilInversion = PerfilInversion.MODERADO) -> str:
        """
        Genera una recomendación personalizada basada en múltiples factores de análisis.
        
        Args:
            technical_data: Datos de análisis técnico
            fundamental_data: Datos fundamentales de la empresa
            prediction_data: Datos de predicción
            dias: Horizonte temporal para el análisis
            perfil: Perfil de inversión del usuario
        
        Returns:
            str: Recomendación detallada y personalizada
        """
        try:
            self.logger.info(f"Iniciando análisis para perfil {perfil.value} a {dias} días")
            
            # Análisis completo
            analisis = self._realizar_analisis_completo(
                technical_data, fundamental_data, prediction_data, dias, perfil
            )
            
            # Generar recomendación personalizada
            recomendacion = self._generar_recomendacion_personalizada(analisis, perfil)
            
            self.logger.info("Análisis completado exitosamente")
            return recomendacion
            
        except Exception as e:
            self.logger.error(f"Error en análisis: {str(e)}")
            return self._generar_mensaje_error(str(e))

    def _realizar_analisis_completo(self, technical_data, fundamental_data, 
                                  prediction_data, dias, perfil) -> dict:
        """Realiza un análisis completo de todos los factores relevantes."""
        
        # Análisis de riesgo
        metricas_riesgo = self._calcular_metricas_riesgo(technical_data)
        
        # Análisis técnico avanzado
        señales_tecnicas = self._analisis_tecnico_avanzado(technical_data)
        
        # Análisis fundamental profundo
        analisis_fundamental = self._analisis_fundamental_profundo(fundamental_data)
        
        # Análisis predictivo
        analisis_predictivo = self._analisis_predictivo_avanzado(prediction_data, dias)
        
        # Análisis de mercado y contexto
        contexto_mercado = self._analizar_contexto_mercado()
        
        return {
            'riesgo': metricas_riesgo,
            'tecnico': señales_tecnicas,
            'fundamental': analisis_fundamental,
            'predictivo': analisis_predictivo,
            'contexto': contexto_mercado
        }

    def _generar_recomendacion_personalizada(self, analisis: dict, 
                                           perfil: PerfilInversion) -> str:
        """
        Genera una recomendación personalizada con formato de asistente virtual.
        """
        # Obtener nombre aleatorio para el asistente
        nombres = ["NOVA", "AIDA", "FINN", "SAGE"]
        asistente_nombre = np.random.choice(nombres)
        
        # Iniciar mensaje personalizado
        mensaje = f"""
🤖 Asistente Virtual {asistente_nombre} - Análisis Financiero Personalizado
════���══════════════════════════════════════════════════

👋 ¡Hola! Soy {asistente_nombre}, tu asistente financiero personal.
He analizado detalladamente los datos disponibles y preparé este informe 
especialmente para tu perfil {perfil.value}.

📊 RESUMEN DE ANÁLISIS
──────────────────

🔍 Análisis Técnico:
• Tendencia principal: {analisis['tecnico']['tendencia']}
• RSI actual: {analisis['tecnico']['rsi']:.2f}
• Señales de momentum: {analisis['tecnico']['señales_momentum']}

📈 Métricas de Riesgo:
• Volatilidad: {analisis['riesgo'].volatilidad:.2%}
• Ratio Sharpe: {analisis['riesgo'].sharpe_ratio:.2f}
• Máximo drawdown: {analisis['riesgo'].max_drawdown:.2%}

💰 Salud Financiera:
• ROE: {analisis['fundamental'].roe:.2%}
• Margen de beneficio: {analisis['fundamental'].margen_beneficio:.2%}
• Nivel de deuda: {'Alto ⚠️' if analisis['fundamental'].deuda_capital > 1 else 'Saludable ✅'}

🎯 RECOMENDACIÓN PERSONALIZADA
──────────────────────────

{self._generar_recomendacion_principal(analisis, perfil)}

🔮 PERSPECTIVAS FUTURAS
─────────────────────
{analisis['predictivo']['mensaje']}

⚡ ACCIONES SUGERIDAS
────────────────────
{self._generar_acciones_sugeridas(analisis, perfil)}

⚠️ ADVERTENCIAS Y CONSIDERACIONES
───────────────────────────────
• Recuerda que todo análisis tiene un margen de error
• Diversifica tu portafolio para minimizar riesgos
• Considera consultar con un asesor financiero profesional

🤖 {asistente_nombre} siempre a tu servicio. ¿Necesitas algo más?
═══════════════════════════════════════════════════════
"""
        return mensaje

    def _generar_recomendacion_principal(self, analisis: dict, 
                                       perfil: PerfilInversion) -> str:
        """Genera la recomendación principal basada en el análisis."""
        if self._es_recomendacion_compra(analisis, perfil):
            return """
✅ RECOMENDACIÓN: COMPRA
Basado en múltiples indicadores positivos:
• Tendencia técnica favorable
• Fundamentos sólidos
• Riesgo dentro de parámetros aceptables
• Perspectivas de crecimiento positivas"""
        elif self._es_recomendacion_venta(analisis, perfil):
            return """
🛑 RECOMENDACIÓN: VENTA
Factores que sugieren precaución:
• Señales técnicas negativas
• Deterioro en fundamentos
• Riesgo por encima de niveles aceptables
• Perspectivas inciertas"""
        else:
            return """
⚠️ RECOMENDACIÓN: MANTENER/NEUTRAL
Situación mixta que sugiere:
• Mantener posiciones existentes
• Esperar mejores puntos de entrada
• Monitorear cambios en tendencia
• Reevaluar en próximos días"""

    def _generar_acciones_sugeridas(self, analisis: dict, 
                                   perfil: PerfilInversion) -> str:
        """Genera sugerencias de acciones específicas para el usuario."""
        acciones = []
        
        if perfil == PerfilInversion.CONSERVADOR:
            acciones.append("• Establecer stop loss en -5% para proteger capital")
            acciones.append("• Considerar inversión gradual (dollar-cost averaging)")
        elif perfil == PerfilInversion.AGRESIVO:
            acciones.append("• Monitorear niveles de soporte/resistencia para entradas")
            acciones.append("• Considerar estrategias de trading más dinámicas")
        
        acciones.append(f"• Establecer alertas de precio en {analisis['tecnico']['niveles_clave']}")
        acciones.append("• Revisar diversificación del portafolio")
        
        return "\n".join(acciones)

    def _es_recomendacion_compra(self, analisis: dict, perfil: PerfilInversion) -> bool:
        """Determina si se debe recomendar compra basado en el análisis."""
        return (analisis['tecnico']['tendencia'] == 'ALCISTA' and
                analisis['riesgo'].sharpe_ratio > 1 and
                analisis['fundamental'].roe > 0.15)

    def _es_recomendacion_venta(self, analisis: dict, perfil: PerfilInversion) -> bool:
        """Determina si se debe recomendar venta basado en el análisis."""
        return (analisis['tecnico']['tendencia'] == 'BAJISTA' and
                analisis['riesgo'].max_drawdown > self.risk_levels[perfil]['max_drawdown_limit'])

    def _generar_mensaje_error(self, error: str) -> str:
        """Genera un mensaje de error amigable."""
        return f"""
🤖 Asistente Virtual - Reporte de Error
═══════════════════════════════════

❌ Lo siento, he encontrado un problema al analizar los datos:
{error}

Sugerencias:
• Verifica que los datos proporcionados sean correctos
• Intenta con un período de análisis diferente
• Contacta soporte si el problema persiste

¿Necesitas ayuda adicional?
"""

    def _calcular_metricas_riesgo(self, technical_data: dict) -> MetricasRiesgo:
        """
        Calcula las métricas de riesgo basadas en los datos técnicos.
        
        Args:
            technical_data: Diccionario con datos técnicos del activo
            
        Returns:
            MetricasRiesgo: Objeto con las métricas de riesgo calculadas
        """
        try:
            # Extraer datos necesarios
            precios = technical_data.get('precios_cierre', [])
            if not precios:
                raise ValueError("No hay datos de precios disponibles")
            
            # Convertir a numpy array si no lo es
            precios = np.array(precios)
            retornos = np.diff(precios) / precios[:-1]
            
            # Calcular volatilidad
            volatilidad = np.std(retornos) * np.sqrt(252)  # Anualizada
            
            # Calcular VaR al 95%
            var_95 = np.percentile(retornos, 5) * np.sqrt(252)
            
            # Calcular Sharpe Ratio (asumiendo tasa libre de riesgo = 2%)
            rf = 0.02
            retorno_medio = np.mean(retornos) * 252
            sharpe_ratio = (retorno_medio - rf) / volatilidad if volatilidad != 0 else 0
            
            # Calcular Sortino Ratio
            retornos_negativos = retornos[retornos < 0]
            downside_vol = np.std(retornos_negativos) * np.sqrt(252) if len(retornos_negativos) > 0 else 1
            sortino_ratio = (retorno_medio - rf) / downside_vol if downside_vol != 0 else 0
            
            # Calcular Maximum Drawdown
            cum_returns = np.cumprod(1 + retornos)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            return MetricasRiesgo(
                volatilidad=volatilidad,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            self.logger.error(f"Error al calcular métricas de riesgo: {str(e)}")
            # Retornar valores por defecto en caso de error
            return MetricasRiesgo(
                volatilidad=0.0,
                var_95=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0
            )

    def _analisis_tecnico_avanzado(self, technical_data: dict) -> dict:
        """
        Realiza un análisis técnico avanzado de los datos.
        """
        try:
            return {
                'tendencia': technical_data.get('trend', 'NEUTRAL'),
                'rsi': technical_data.get('rsi', 50),
                'señales_momentum': technical_data.get('momentum_signals', 'NEUTRAL'),
                'niveles_clave': technical_data.get('key_levels', '---')
            }
        except Exception as e:
            self.logger.error(f"Error en análisis técnico: {str(e)}")
            return {
                'tendencia': 'NEUTRAL',
                'rsi': 50,
                'señales_momentum': 'NEUTRAL',
                'niveles_clave': '---'
            }

    def _analisis_fundamental_profundo(self, fundamental_data: dict) -> AnalisisFundamental:
        """
        Realiza un análisis fundamental profundo.
        """
        try:
            return AnalisisFundamental(
                pe_ratio=fundamental_data.get('pe_ratio', 0.0),
                precio_libro=fundamental_data.get('precio_libro', 0.0),
                margen_beneficio=fundamental_data.get('profit_margin', 0.0),
                roe=fundamental_data.get('return_on_equity', 0.0),
                roa=fundamental_data.get('return_on_assets', 0.0),
                deuda_capital=fundamental_data.get('debt_to_equity', 0.0),
                crecimiento_ingresos=fundamental_data.get('revenue_growth', 0.0)
            )
        except Exception as e:
            self.logger.error(f"Error en análisis fundamental: {str(e)}")
            return AnalisisFundamental(
                pe_ratio=0.0,
                precio_libro=0.0,
                margen_beneficio=0.0,
                roe=0.0,
                roa=0.0,
                deuda_capital=0.0,
                crecimiento_ingresos=0.0
            )

    def _analisis_predictivo_avanzado(self, prediction_data: dict, dias: int) -> dict:
        """
        Realiza un análisis predictivo avanzado.
        """
        try:
            return {
                'mensaje': f"Análisis predictivo para {dias} días:\n" +
                         f"• Tendencia esperada: {prediction_data.get('trend', 'NEUTRAL')}\n" +
                         f"• Confianza: {prediction_data.get('confidence', 0):.1f}%"
            }
        except Exception as e:
            self.logger.error(f"Error en análisis predictivo: {str(e)}")
            return {
                'mensaje': "No hay suficientes datos para realizar predicciones confiables."
            }

    def _analizar_contexto_mercado(self) -> dict:
        """
        Analiza el contexto general del mercado.
        """
        return {
            'sentimiento': self.market_sentiment or 'NEUTRAL',
            'condicion_mercado': 'NORMAL',
            'volatilidad_mercado': 'MODERADA'
        }

def mostrar_recomendaciones(recomendaciones):
    """
    Muestra las recomendaciones de manera estructurada y visualmente atractiva.
    """
    for recomendacion in recomendaciones:
        print(f"Recomendación: {recomendacion['tipo']}")
        print(f"Confianza: {recomendacion['confianza']:.1f}%")
        print(f"Razones: {', '.join(recomendacion['razones'])}")
        print("-" * 40)

def validar_datos(data):
    """
    Valida la calidad de los datos y lanza advertencias si son insuficientes.
    """
    if data.isnull().any().any():
        logging.warning("Datos faltantes detectados. Se recomienda revisar la fuente de datos.")
