# ma-crossover-backtester
Backtester en Python para estrategias de cruce de medias moviles, con calculo de metricas de rendimiento (Sharpe, drawdown, win rate) y simulacion de comisiones/slippage. Soporta acciones y forex via yfinance.
# MA Crossover Backtester

Backtester en Python para estrategias de cruce de medias moviles (moving average crossover).

## Que hace
- Descarga datos historicos de acciones o forex (via yfinance)
- Calcula senales de compra/venta en el cruce de dos medias moviles
- Simula el rendimiento con y sin comisiones/slippage
- Calcula metricas: retorno total, drawdown maximo, Sharpe ratio, win rate
- Genera un grafico con las senales marcadas y la curva de capital

## Como usarlo
1. Instalar dependencias: `pip install yfinance pandas matplotlib numpy`
2. Ajustar los parametros en la seccion CONFIG (ticker, medias, comisiones)
3. Correr: `python3 backtester.py`

## Ejemplo de resultado
Backtest de AAPL (2018-2026), cruce 50/200:
- Retorno total: 104.92% (vs 618.73% buy & hold)
- Drawdown maximo: -43.51%
- Sharpe ratio: 0.46
