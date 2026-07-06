"""
Backtester de Cruce de Medias Moviles
--------------------------------------
Descarga datos historicos de un activo (accion o forex), calcula dos medias
moviles, genera senales de compra/venta en los cruces, simula las operaciones
y calcula metricas de rendimiento (retorno total, drawdown maximo, win rate,
ratio de Sharpe). Tambien grafica el precio con las senales marcadas.

Como usarlo:
1. Cambia los parametros en la seccion CONFIG mas abajo.
2. Corre el script: python3 backtester.py
3. Revisa los resultados impresos y el grafico generado (chart.png).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ CONFIG ============
TICKER = "AAPL"          # simbolo del activo (accion o par de forex, ej: "EURUSD=X")
START = "2018-01-01"
END = "2026-07-01"
MA_CORTA = 50             # media movil corta (dias)
MA_LARGA = 200            # media movil larga (dias)
CAPITAL_INICIAL = 10000   # capital de simulacion
USE_YFINANCE = True       # poner False para usar datos simulados de prueba
COMISION_PCT = 0.001      # 0.1% por operacion (entrada o salida), tipico de un broker retail
SLIPPAGE_PCT = 0.0005     # 0.05% de deslizamiento estimado por operacion
# =================================


def descargar_datos(ticker, start, end):
    """Descarga datos historicos con yfinance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}")
    # yfinance a veces devuelve columnas multi-nivel; las aplanamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"})
    return df


def generar_datos_simulados(start, end, seed=42):
    """Genera una serie de precios simulada tipo random walk, solo para
    probar la logica del backtester cuando no hay acceso a internet."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(start=start, end=end, freq="B")  # dias habiles
    retornos = rng.normal(loc=0.0004, scale=0.012, size=len(fechas))
    precio = 100 * (1 + retornos).cumprod()
    df = pd.DataFrame({"close": precio}, index=fechas)
    return df


def calcular_senales(df, ma_corta, ma_larga):
    """Calcula las medias moviles y genera senales de compra/venta."""
    df = df.copy()
    df["ma_corta"] = df["close"].rolling(ma_corta).mean()
    df["ma_larga"] = df["close"].rolling(ma_larga).mean()

    # posicion: 1 = comprado, 0 = fuera del mercado
    df["posicion"] = np.where(df["ma_corta"] > df["ma_larga"], 1, 0)

    # senal: momento exacto del cruce (para marcar en el grafico)
    df["senal"] = df["posicion"].diff()  # 1 = entra, -1 = sale
    return df


def simular_operaciones(df, capital_inicial, comision_pct=0.0, slippage_pct=0.0):
    """Simula el rendimiento de seguir las senales generadas, descontando
    comision y slippage cada vez que se entra o se sale de una posicion."""
    df = df.copy()
    df["retorno_diario"] = df["close"].pct_change()
    # el retorno de la estrategia es el retorno del activo SOLO cuando
    # estamos posicionados (posicion del dia anterior, para evitar look-ahead)
    df["retorno_estrategia"] = df["retorno_diario"] * df["posicion"].shift(1)

    # costo de transaccion: se aplica el dia que hay una senal (entra o sale)
    costo_por_operacion = comision_pct + slippage_pct
    df["costo_transaccion"] = np.where(df["senal"].abs() == 1, costo_por_operacion, 0.0)
    df["retorno_estrategia_neto"] = df["retorno_estrategia"] - df["costo_transaccion"]

    df["capital_estrategia"] = capital_inicial * (1 + df["retorno_estrategia_neto"]).cumprod()
    df["capital_buy_hold"] = capital_inicial * (1 + df["retorno_diario"]).cumprod()
    return df


def calcular_metricas(df, comision_pct=0.0, slippage_pct=0.0):
    """Calcula metricas clave de rendimiento de la estrategia."""
    retorno_total = df["capital_estrategia"].iloc[-1] / df["capital_estrategia"].dropna().iloc[0] - 1

    # drawdown maximo
    cum_max = df["capital_estrategia"].cummax()
    drawdown = (df["capital_estrategia"] - cum_max) / cum_max
    drawdown_maximo = drawdown.min()

    # ratio de Sharpe simplificado (asumiendo tasa libre de riesgo = 0)
    retornos_diarios = df["retorno_estrategia_neto"].dropna()
    sharpe = (retornos_diarios.mean() / retornos_diarios.std()) * np.sqrt(252) if retornos_diarios.std() > 0 else np.nan

    # win rate: de las veces que hubo una operacion cerrada (entra y sale),
    # cuantas fueron positivas
    entradas = df.index[df["senal"] == 1]
    salidas = df.index[df["senal"] == -1]
    operaciones = []
    for entrada in entradas:
        salidas_posteriores = salidas[salidas > entrada]
        if len(salidas_posteriores) > 0:
            salida = salidas_posteriores[0]
            precio_entrada = df.loc[entrada, "close"]
            precio_salida = df.loc[salida, "close"]
            costo_total = 2 * (comision_pct + slippage_pct)  # entrada + salida
            operaciones.append(precio_salida / precio_entrada - 1 - costo_total)

    win_rate = (np.array(operaciones) > 0).mean() if operaciones else np.nan
    num_operaciones = len(operaciones)

    return {
        "retorno_total_%": round(retorno_total * 100, 2),
        "drawdown_maximo_%": round(drawdown_maximo * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "win_rate_%": round(win_rate * 100, 2) if not np.isnan(win_rate) else None,
        "numero_operaciones": num_operaciones,
        "retorno_buy_hold_%": round((df["capital_buy_hold"].iloc[-1] / df["capital_buy_hold"].dropna().iloc[0] - 1) * 100, 2),
    }


def graficar(df, ticker, ma_corta, ma_larga, output_path="chart.png"):
    """Genera el grafico de precio + medias + senales + curva de capital."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["close"], label="Precio", color="#333333", linewidth=1)
    ax1.plot(df.index, df["ma_corta"], label=f"MA {ma_corta}", color="#2563eb", linewidth=1.2)
    ax1.plot(df.index, df["ma_larga"], label=f"MA {ma_larga}", color="#dc2626", linewidth=1.2)

    compras = df[df["senal"] == 1]
    ventas = df[df["senal"] == -1]
    ax1.scatter(compras.index, compras["close"], marker="^", color="green", s=80, label="Compra", zorder=5)
    ax1.scatter(ventas.index, ventas["close"], marker="v", color="red", s=80, label="Venta", zorder=5)

    ax1.set_title(f"{ticker} - Cruce de medias moviles ({ma_corta}/{ma_larga})")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["capital_estrategia"], label="Estrategia", color="#2563eb")
    ax2.plot(df.index, df["capital_buy_hold"], label="Buy & Hold", color="#999999", linestyle="--")
    ax2.set_title("Curva de capital")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Grafico guardado en: {output_path}")


def main():
    if USE_YFINANCE:
        try:
            df = descargar_datos(TICKER, START, END)
        except Exception as e:
            print(f"No se pudo descargar con yfinance ({e}). Usando datos simulados.")
            df = generar_datos_simulados(START, END)
    else:
        df = generar_datos_simulados(START, END)

    df = calcular_senales(df, MA_CORTA, MA_LARGA)

    # simulacion SIN costos (para comparar el impacto)
    df_sin_costos = simular_operaciones(df, CAPITAL_INICIAL, comision_pct=0.0, slippage_pct=0.0)
    metricas_sin_costos = calcular_metricas(df_sin_costos, comision_pct=0.0, slippage_pct=0.0)

    # simulacion CON costos reales
    df_con_costos = simular_operaciones(df, CAPITAL_INICIAL, comision_pct=COMISION_PCT, slippage_pct=SLIPPAGE_PCT)
    metricas_con_costos = calcular_metricas(df_con_costos, comision_pct=COMISION_PCT, slippage_pct=SLIPPAGE_PCT)

    print("\n=== RESULTADOS DEL BACKTEST (sin comisiones ni slippage) ===")
    for k, v in metricas_sin_costos.items():
        print(f"{k}: {v}")

    print("\n=== RESULTADOS DEL BACKTEST (con comisiones y slippage) ===")
    for k, v in metricas_con_costos.items():
        print(f"{k}: {v}")

    impacto = metricas_sin_costos["retorno_total_%"] - metricas_con_costos["retorno_total_%"]
    print(f"\nImpacto de costos de transaccion en el retorno total: -{impacto:.2f} puntos porcentuales")

    graficar(df_con_costos, TICKER, MA_CORTA, MA_LARGA)


if __name__ == "__main__":
    main()
