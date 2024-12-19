import sys
from pathlib import Path

# Añadir el directorio padre al path de Python
sys.path.append(str(Path(__file__).parent))

from stock_predictor import AdvancedStockPredictor

def main():
    print("=== Predictor Avanzado de Precios de Acciones ===")
    
    symbol = input("\nIngrese el símbolo de la acción (ej: AAPL): ").upper()
    days = int(input("Ingrese el número de días para la predicción: "))
    
    predictor = AdvancedStockPredictor()
    data = predictor.train_model(symbol, days)
    
    if data is not None:
        predictions = predictor.predict_future(data, days)
        # ... (código de visualización de resultados)

if __name__ == "__main__":
    main()