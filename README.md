import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from pandas.io.excel import ExcelWriter
# Usamos ProcessPoolExecutor para verdadera paralelización de CPU
import concurrent.futures

# Opciones de Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#
#=================================================================
# 1. CONFIGURACIÓN DE PARÁMETROS Y CONSTANTES GLOBALES
#=================================================================

# 📌 CONFIGURACIÓN DE PARALELIZACIÓN
MAX_WORKERS = os.cpu_count() or 8

# 📌 LISTA DE ACTIVOS A EVALUAR
SYMBOLS_TO_EVALUATE = ["USDCLP"]
# 📌 TIME FRAME CLAVE DE EVALUACIÓN
TIMEFRAMES_TO_EVALUATE = (
    mt5.TIMEFRAME_M15,
  
)

# --- CONSTANTES DE ESTRATEGIAS ORIGINALES ---
STRATEGY_EMA_REVERSION_NAME = "EMA_34_55_Reversion"
EMA_SUPPORT_RESISTANCE_FAST = 34
EMA_SUPPORT_RESISTANCE_SLOW = 55
STRATEGY_TRIPLE_FILTER_NAME = "RSI_BB_PinBar_Reversion"
BB_PERIOD = 20
BB_DEV = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STRATEGY_SR_REVERSION_NAME = "SR_EMA55_Reversion"
EMA_TREND_PERIOD = 55
FRACTAL_PERIOD_SR = 8
SR_MAX_DISTANCE_ATR = 0.5
SR_LOOKBACK_BARS = 120
SR_MERGE_DISTANCE_ATR = 1.0
STRATEGY_ATR_REVERSION_NAME = "ATR_Deviation_Reversion"
ATR_DEVIATION_MULTIPLIER = 1.5
PINBAR_WICK_RATIO_ULTRA_SUAVE = 0.45
PINBAR_BODY_MAX_RATIO_SUAVE = 0.35
FRACTAL_PERIOD = 5
EMA_TREND_LONG = 200
STRATEGY_MACD_TREND_NAME = "MACD_EMA200_Trend"
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BODY_MAX_ATR_MULT = 1.5
STRATEGY_ATR_BREAKOUT_NAME = "ATR_Channel_Breakout"
ATR_CHANNEL_PERIOD = 20    
ATR_CHANNEL_MULT = 2.0     
ATR_CONSOLIDATION_MULT = 1.0

# --- CONSTANTES DE NUEVAS ESTRATEGIAS (AÑADIDAS) ---
STRATEGY_ADX_MOMENTUM_NAME = "ADX_Momentum_Break"
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ATR_BREAKOUT_MULT = 1.5
MOMENTUM_PERIOD = 10

STRATEGY_ICHIMOKU_CLOUD_NAME = "Ichimoku_Cloud_Trend"
ICHI_TENKAN = 9
ICHI_KIJUN = 26
ICHI_SENKOU_SPAN_B = 52 # Senkou Span B Period

STRATEGY_RSI_FRACTAL_REVERSAL_NAME = "RSI_Fractal_Reversal"
RSI_REVERSAL_PERIOD = 5
RSI_REVERSAL_OVERSOLD = 10
RSI_REVERSAL_OVERBOUGHT = 90
FRACTAL_REVERSAL_PERIOD = 3 # Período más corto para capturar giros rápidos

STRATEGY_VOLATILITY_COMPRESSION_NAME = "Volatility_Compression_Entry"
VOL_COMP_PERIOD = 50
COMPRESSION_RATIO_THRESHOLD = 0.5 # Comparar ATR de 10 barras vs ATR de 50 barras


# Parámetros Comunes de Backtesting
ATR_PERIOD = 200
end_date = datetime.now() + timedelta(days=300)
MAX_BARS_TO_DOWNLOAD = 100000

# --- CONFIGURACIÓN DE BARRAS DE SEGUIMIENTO ---
TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_M1: 'M1', mt5.TIMEFRAME_M5: 'M5', mt5.TIMEFRAME_M15: 'M15',
    mt5.TIMEFRAME_M30: 'M30', mt5.TIMEFRAME_H1: 'H1', mt5.TIMEFRAME_H4: 'H4',
    mt5.TIMEFRAME_D1: 'D1',
}

# 📌 OBJETIVOS ATR
ATR_TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# --- MAPEADOR DE MINUTOS DEL TIMEFRAME (para cálculo fraccional de tiempo) ---
TIMEFRAME_MINUTES = {
    mt5.TIMEFRAME_M1: 1, mt5.TIMEFRAME_M5: 5, mt5.TIMEFRAME_M15: 15,
    mt5.TIMEFRAME_M30: 30, mt5.TIMEFRAME_H1: 60, mt5.TIMEFRAME_H4: 240,
    mt5.TIMEFRAME_D1: 1440,
}

# --- Constante para reemplazar el infinito en Pot_SL_Min ---
MAX_POTENTIAL_REPLACEMENT = 99999.0

#
#-----------------------------------------------------------------
# 2. FUNCIONES AUXILIARES
#-----------------------------------------------------------------

def get_pip_multiplier(symbol):
    symbol_upper = symbol.upper()
    if "XAUUSD" in symbol_upper or "XAGUSD" in symbol_upper: return 10
    if "JPY" in symbol_upper: return 100
    if any(s in symbol_upper for s in ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "CLP"]): return 10000
    return 1

def get_forward_bars_count(timeframe):
    if timeframe in [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30]: return 16
    elif timeframe in [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]: return 16
    elif timeframe == mt5.TIMEFRAME_D1: return 12
    return 16

def is_fractal(df, index, direction, period):
    center = period
    if index < center or index >= len(df) - center: return False
    window = df.iloc[index - center : index + center + 1]
    if direction == 1:
        return window['High'].iloc[center] == window['High'].max()
    elif direction == -1:
        return window['Low'].iloc[center] == window['Low'].min()
    return False

def merge_sr_levels(levels, atr_value, direction):
    if not levels: return []
    sorted_levels = sorted(levels, reverse=(direction == 1))
    merged_levels = []
    if not sorted_levels: return []
    current_zone_start = sorted_levels[0]
    for i in range(1, len(sorted_levels)):
        level = sorted_levels[i]
        distance = abs(level - current_zone_start)
        if distance <= SR_MERGE_DISTANCE_ATR * atr_value:
            if direction == 1: current_zone_start = max(current_zone_start, level)
            else: current_zone_start = min(current_zone_start, level)
        else:
            merged_levels.append(current_zone_start)
            current_zone_start = level
    merged_levels.append(current_zone_start)
    return merged_levels


#
#-----------------------------------------------------------------
# 3. FUNCIONES DE PATRONES DE VELAS
#-----------------------------------------------------------------

def is_pin_bar_ultrasuave(open_price, high, low, close_price, atr_value, direction=None):
    price_range = high - low
    body = abs(close_price - open_price)
    if price_range == 0: return False
    if body / price_range > PINBAR_BODY_MAX_RATIO_SUAVE: return False
    if direction == 1:
        lower_wick = min(open_price, close_price) - low
        return (lower_wick / price_range) >= PINBAR_WICK_RATIO_ULTRA_SUAVE
    elif direction == -1:
        upper_wick = high - max(open_price, close_price)
        return (upper_wick / price_range) >= PINBAR_WICK_RATIO_ULTRA_SUAVE
    return False

def is_engulfing(current_bar, prev_bar, direction):
    if prev_bar is None: return False
    if direction == 1:
        is_bullish = current_bar['Close'] > current_bar['Open']
        prev_is_bearish = prev_bar['Close'] < prev_bar['Open']
        body_engulfs = (current_bar['Close'] > prev_bar['Open']) and \
                       (current_bar['Open'] < prev_bar['Close'])
        return is_bullish and prev_is_bearish and body_engulfs
    elif direction == -1:
        is_bearish = current_bar['Close'] < current_bar['Open']
        prev_is_bullish = prev_bar['Close'] > prev_bar['Open']
        body_engulfs = (current_bar['Close'] < prev_bar['Open']) and \
                       (current_bar['Open'] > prev_bar['Close'])
        return is_bearish and prev_is_bullish and body_engulfs
    return False


#
#-----------------------------------------------------------------
# 4. FUNCIONES DE SEÑALIZACIÓN POR ESTRATEGIAS (ORIGINALES)
#-----------------------------------------------------------------

def generate_signals_ema_reversion(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    max_ema_period = max(EMA_SUPPORT_RESISTANCE_FAST, EMA_SUPPORT_RESISTANCE_SLOW)
    start_index = max(max_ema_period + 1, FRACTAL_PERIOD // 2 + 1)
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_bar_index = i - 1
        if f'EMA_{EMA_SUPPORT_RESISTANCE_FAST}' not in prev_bar or 'ATR_200' not in prev_bar: continue
        ema_34 = prev_bar[f'EMA_{EMA_SUPPORT_RESISTANCE_FAST}']
        ema_55 = prev_bar[f'EMA_{EMA_SUPPORT_RESISTANCE_SLOW}']
        atr_value = prev_bar['ATR_200']
        zone_upper = max(ema_34, ema_55)
        zone_lower = min(ema_34, ema_55)
        if prev_bar['Close'] > ema_34 and ema_34 > ema_55:
            if is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=1):
                criterio_prueba_ema_34 = (prev_bar['Low'] <= ema_34)
                criterio_dentro_zona = (prev_bar['Low'] < zone_upper) and (prev_bar['High'] > zone_lower)
                EMA_Reversion_Condition = criterio_prueba_ema_34 or criterio_dentro_zona
                fractal_condition = is_fractal(df, prev_bar_index, direction=-1, period=FRACTAL_PERIOD)
                if EMA_Reversion_Condition and fractal_condition:
                    df.loc[current_bar.name, 'Signal'] = 1.0
                    df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                    df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['Low']
        elif prev_bar['Close'] < ema_34 and ema_34 < ema_55:
            if is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=-1):
                criterio_prueba_ema_34 = (prev_bar['High'] >= ema_34)
                criterio_dentro_zona = (prev_bar['High'] > zone_lower) and (prev_bar['Low'] < zone_upper)
                EMA_Reversion_Condition = criterio_prueba_ema_34 or criterio_dentro_zona
                fractal_condition = is_fractal(df, prev_bar_index, direction=1, period=FRACTAL_PERIOD)
                if EMA_Reversion_Condition and fractal_condition:
                    df.loc[current_bar.name, 'Signal'] = -1.0
                    df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                    df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['High']
    return df.iloc[start_index + 1:].copy()

def generate_signals_triple_filter(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    start_index = max(BB_PERIOD, RSI_PERIOD) + 1
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_bar_index = i - 1
        if 'ATR_200' not in prev_bar or 'BB_Lower' not in prev_bar or 'RSI_14' not in prev_bar: continue
        atr_value = prev_bar['ATR_200']
        if prev_bar['RSI_14'] <= RSI_OVERSOLD:
            is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=1)
            touches_lower_band = (prev_bar['Low'] <= prev_bar['BB_Lower'])
            if is_pin and touches_lower_band:
                df.loc[current_bar.name, 'Signal'] = 1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['Low']
        elif prev_bar['RSI_14'] >= RSI_OVERBOUGHT:
            is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=-1)
            touches_upper_band = (prev_bar['High'] >= prev_bar['BB_Upper'])
            if is_pin and touches_upper_band:
                df.loc[current_bar.name, 'Signal'] = -1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['High']
    return df.iloc[start_index + 1:].copy()

def generate_signals_sr_reversion(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    df['Fractal_R'] = [1 if is_fractal(df, i, 1, FRACTAL_PERIOD_SR) else 0 for i in range(len(df))]
    df['Fractal_S'] = [-1 if is_fractal(df, i, -1, FRACTAL_PERIOD_SR) else 0 for i in range(len(df))]
    start_index_indicators = EMA_TREND_PERIOD + FRACTAL_PERIOD_SR + 1
    for i in range(start_index_indicators, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2] if i >= 2 else None
        prev_bar_index = i - 1
        if 'ATR_200' not in prev_bar or f'EMA_{EMA_TREND_PERIOD}' not in prev_bar: continue
        ema_55 = prev_bar[f'EMA_{EMA_TREND_PERIOD}']
        atr_value = prev_bar['ATR_200']
        lookback_start_index = max(0, i - SR_LOOKBACK_BARS)
        df_lookback = df.iloc[lookback_start_index : i].copy()
        resistance_levels_raw = df_lookback[df_lookback['Fractal_R'] == 1]['High'].tolist()
        support_levels_raw = df_lookback[df_lookback['Fractal_S'] == -1]['Low'].tolist()
        resistance_levels = merge_sr_levels(resistance_levels_raw, atr_value, direction=1)
        support_levels = merge_sr_levels(support_levels_raw, atr_value, direction=-1)
        if prev_bar['Close'] > ema_55:
            for s_level in support_levels:
                distance = s_level - prev_bar['Low']
                if abs(distance) < SR_MAX_DISTANCE_ATR * atr_value:
                    is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=1)
                    is_eng = is_engulfing(prev_bar, prev_prev_bar, direction=1)
                    if is_pin or is_eng:
                        df.loc[current_bar.name, 'Signal'] = 1.0
                        df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                        df.loc[current_bar.name, 'Fractal_Plot'] = s_level
                        break
        elif prev_bar['Close'] < ema_55:
            for r_level in resistance_levels:
                distance = prev_bar['High'] - r_level
                if abs(distance) < SR_MAX_DISTANCE_ATR * atr_value:
                    is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=-1)
                    is_eng = is_engulfing(prev_bar, prev_prev_bar, direction=-1)
                    if is_pin or is_eng:
                        df.loc[current_bar.name, 'Signal'] = -1.0
                        df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                        df.loc[current_bar.name, 'Fractal_Plot'] = r_level
                        break
    return df.iloc[start_index_indicators + 1:].copy()

def generate_signals_atr_reversion(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    start_index = EMA_TREND_PERIOD + 2
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2] if i >= 2 else None
        prev_bar_index = i - 1
        if f'EMA_{EMA_TREND_PERIOD}' not in prev_bar or 'ATR_200' not in prev_bar: continue
        ema_55 = prev_bar[f'EMA_{EMA_TREND_PERIOD}']
        atr_value = prev_bar['ATR_200']
        extreme_condition = (ema_55 - prev_bar['Low']) >= (ATR_DEVIATION_MULTIPLIER * atr_value)
        if extreme_condition:
            is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=1)
            is_eng = is_engulfing(prev_bar, prev_prev_bar, direction=1)
            if is_pin or is_eng:
                df.loc[current_bar.name, 'Signal'] = 1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = ema_55
        extreme_condition = (prev_bar['High'] - ema_55) >= (ATR_DEVIATION_MULTIPLIER * atr_value)
        if extreme_condition:
            is_pin = is_pin_bar_ultrasuave(prev_bar['Open'], prev_bar['High'], prev_bar['Low'], prev_bar['Close'], atr_value, direction=-1)
            is_eng = is_engulfing(prev_bar, prev_prev_bar, direction=-1)
            if is_pin or is_eng:
                df.loc[current_bar.name, 'Signal'] = -1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = ema_55
    return df.iloc[start_index + 1:].copy()

def generate_signals_macd_trend(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    start_index = max(EMA_TREND_LONG, MACD_SLOW_PERIOD) + 2
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2]
        prev_bar_index = i - 1
        ema_200_col = f'EMA_{EMA_TREND_LONG}'
        if ema_200_col not in prev_bar or 'ATR_200' not in prev_bar or 'MACD_Hist' not in prev_bar: continue
        ema_200 = prev_bar[ema_200_col]
        ema_200_prev = prev_prev_bar[ema_200_col] if i >= 2 else np.nan
        atr_value = prev_bar['ATR_200']
        prev_body = abs(prev_bar['Close'] - prev_bar['Open'])
        size_condition = (prev_body <= BODY_MAX_ATR_MULT * atr_value)
        cruce_condition = (prev_bar['Close'] > ema_200) and (prev_prev_bar['Close'] <= ema_200_prev)
        macd_condition = (prev_bar['MACD_Hist'] > 0)
        if cruce_condition and macd_condition and size_condition:
            df.loc[current_bar.name, 'Signal'] = 1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = ema_200
        cruce_condition = (prev_bar['Close'] < ema_200) and (prev_prev_bar['Close'] >= ema_200_prev)
        macd_condition = (prev_bar['MACD_Hist'] < 0)
        if cruce_condition and macd_condition and size_condition:
            df.loc[current_bar.name, 'Signal'] = -1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = ema_200
    return df.iloc[start_index + 1:].copy()

def generate_signals_atr_breakout(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    start_index = max(ATR_PERIOD, ATR_CHANNEL_PERIOD) + 2
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_bar_index = i - 1
        if 'ATR_200' not in prev_bar or 'ATR_Upper_Channel' not in prev_bar: continue
        atr_200_value = prev_bar['ATR_200']
        upper_channel = prev_bar['ATR_Upper_Channel']
        lower_channel = prev_bar['ATR_Lower_Channel']
        prev_range = prev_bar['High'] - prev_bar['Low']
        consolidation_condition = (prev_range <= ATR_CONSOLIDATION_MULT * atr_200_value)
        breakout_condition = (prev_bar['Close'] > upper_channel)
        if consolidation_condition and breakout_condition:
            df.loc[current_bar.name, 'Signal'] = 1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = upper_channel
        breakout_condition = (prev_bar['Close'] < lower_channel)
        if consolidation_condition and breakout_condition:
            df.loc[current_bar.name, 'Signal'] = -1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = lower_channel
    return df.iloc[start_index + 1:].copy()

#
#-----------------------------------------------------------------
# 5. FUNCIONES DE SEÑALIZACIÓN POR ESTRATEGIAS (NUEVAS)
#-----------------------------------------------------------------

def generate_signals_adx_momentum_break(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    
    start_index = max(ADX_PERIOD, MOMENTUM_PERIOD, ATR_PERIOD) + 2
    
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2] if i >= 2 else None
        prev_bar_index = i - 1
        
        # Verificar indicadores y barra anterior
        if prev_prev_bar is None or 'ATR_200' not in prev_bar or f'Momentum_{MOMENTUM_PERIOD}' not in prev_bar or 'ADX_14' not in prev_bar: continue
        
        atr_value = prev_bar['ATR_200']
        
        # 🔔 Señal de COMPRA: Tendencia Fuerte (ADX) + Momentum Positivo + Ruptura de High
        if prev_bar['ADX_14'] >= ADX_THRESHOLD and \
           prev_bar[f'Momentum_{MOMENTUM_PERIOD}'] > 0 and \
           (prev_bar['Close'] > prev_prev_bar['High']) and \
           (prev_bar['Close'] - prev_prev_bar['High'] > ATR_BREAKOUT_MULT * atr_value):
            
            df.loc[current_bar.name, 'Signal'] = 1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = prev_prev_bar['High']
            
        # 🔔 Señal de VENTA: Tendencia Fuerte (ADX) + Momentum Negativo + Ruptura de Low
        elif prev_bar['ADX_14'] >= ADX_THRESHOLD and \
             prev_bar[f'Momentum_{MOMENTUM_PERIOD}'] < 0 and \
             (prev_bar['Close'] < prev_prev_bar['Low']) and \
             (prev_prev_bar['Low'] - prev_bar['Close'] > ATR_BREAKOUT_MULT * atr_value):
            
            df.loc[current_bar.name, 'Signal'] = -1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = prev_prev_bar['Low']
            
    return df.iloc[start_index + 1:].copy()

def generate_signals_ichimoku_cloud_trend(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    
    start_index = ICHI_SENKOU_SPAN_B + 2
    
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2] if i >= 2 else None
        prev_bar_index = i - 1
        
        # Verificar indicadores y barra anterior
        if prev_prev_bar is None or 'Tenkan_Sen' not in prev_bar or 'Kijun_Sen' not in prev_bar or 'Senkou_Span_A' not in prev_bar: continue
        
        senkou_max = max(prev_bar['Senkou_Span_A'], prev_bar['Senkou_Span_B'])
        senkou_min = min(prev_bar['Senkou_Span_A'], prev_bar['Senkou_Span_B'])
        
        # 🔔 Señal de COMPRA: Cruce Tenkan > Kijun + Precio por encima de la Nube
        if (prev_bar['Tenkan_Sen'] > prev_bar['Kijun_Sen']) and \
           (prev_prev_bar['Tenkan_Sen'] <= prev_prev_bar['Kijun_Sen']) and \
           (prev_bar['Close'] > senkou_max):
            
            df.loc[current_bar.name, 'Signal'] = 1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = senkou_min # Usamos la nube como referencia de SL
            
        # 🔔 Señal de VENTA: Cruce Tenkan < Kijun + Precio por debajo de la Nube
        elif (prev_bar['Tenkan_Sen'] < prev_bar['Kijun_Sen']) and \
             (prev_prev_bar['Tenkan_Sen'] >= prev_prev_bar['Kijun_Sen']) and \
             (prev_bar['Close'] < senkou_min):
            
            df.loc[current_bar.name, 'Signal'] = -1.0
            df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
            df.loc[current_bar.name, 'Fractal_Plot'] = senkou_max # Usamos la nube como referencia de SL
            
    return df.iloc[start_index + 1:].copy()

def generate_signals_rsi_fractal_reversal(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    
    max_lookback = max(RSI_REVERSAL_PERIOD, FRACTAL_REVERSAL_PERIOD)
    start_index = max_lookback + 2
    
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_bar_index = i - 1
        
        if 'RSI_5' not in prev_bar: continue
        
        # 🔔 Señal de COMPRA: Sobre-venta extrema + Fractal de suelo
        if prev_bar['RSI_5'] <= RSI_REVERSAL_OVERSOLD:
            if is_fractal(df, prev_bar_index, direction=-1, period=FRACTAL_REVERSAL_PERIOD):
                # Criterio adicional: La vela de señal no es un Pin Bar inverso gigante (cierre debe estar por encima del 60% inferior)
                if (prev_bar['Close'] - prev_bar['Low']) > (prev_bar['High'] - prev_bar['Low']) * 0.4:
                    df.loc[current_bar.name, 'Signal'] = 1.0
                    df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                    df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['Low']
                    
        # 🔔 Señal de VENTA: Sobre-compra extrema + Fractal de techo
        elif prev_bar['RSI_5'] >= RSI_REVERSAL_OVERBOUGHT:
            if is_fractal(df, prev_bar_index, direction=1, period=FRACTAL_REVERSAL_PERIOD):
                # Criterio adicional: La vela de señal no es un Pin Bar inverso gigante (cierre debe estar por debajo del 60% superior)
                if (prev_bar['High'] - prev_bar['Close']) > (prev_bar['High'] - prev_bar['Low']) * 0.4:
                    df.loc[current_bar.name, 'Signal'] = -1.0
                    df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                    df.loc[current_bar.name, 'Fractal_Plot'] = prev_bar['High']
            
    return df.iloc[start_index + 1:].copy()

def generate_signals_volatility_compression_entry(df):
    df['Signal'] = 0.0
    df['Signal_Index'] = 0
    df['Fractal_Plot'] = np.nan
    
    start_index = max(VOL_COMP_PERIOD, ATR_PERIOD) + 2
    
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        prev_prev_bar = df.iloc[i-2] if i >= 2 else None
        prev_bar_index = i - 1
        
        if prev_prev_bar is None or 'ATR_50' not in prev_bar or 'ATR_10' not in prev_bar: continue

        # 1. Condición de Compresión: ATR a corto plazo es mucho menor que ATR a largo plazo.
        # Evitar división por cero
        if prev_bar['ATR_50'] == 0: continue 
        compression_ratio = prev_bar['ATR_10'] / prev_bar['ATR_50']
        is_compressed = compression_ratio <= COMPRESSION_RATIO_THRESHOLD

        if is_compressed:
            # 2. Condición de Quiebre (Breakout): La barra anterior rompe claramente el máximo/mínimo anterior
            
            # 🔔 Señal de COMPRA: Quiebre alcista
            if prev_bar['Close'] > prev_prev_bar['High']:
                df.loc[current_bar.name, 'Signal'] = 1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = prev_prev_bar['High']
            
            # 🔔 Señal de VENTA: Quiebre bajista
            elif prev_bar['Close'] < prev_prev_bar['Low']:
                df.loc[current_bar.name, 'Signal'] = -1.0
                df.loc[current_bar.name, 'Signal_Index'] = prev_bar_index
                df.loc[current_bar.name, 'Fractal_Plot'] = prev_prev_bar['Low']
                
    return df.iloc[start_index + 1:].copy()


#
#-----------------------------------------------------------------
# 6. FUNCIÓN DE SEGUIMIENTO Y MÉTRICAS (Lógica de Time_to_SL_Min_Mins)
#-----------------------------------------------------------------

def track_trade_metrics(df_full, index_signal, direction, multiplier, forward_bars, symbol):
    """
    Calcula las métricas de rendimiento, riesgo y el tiempo hasta SL_Min_ATR (Time_to_SL_Min_Mins).
    """
    
    try:
        # Se incluye la barra de señal + las barras de seguimiento. La entrada es en la barra index_signal + 1
        df_track_all = df_full.iloc[index_signal + 1 : index_signal + 1 + forward_bars].copy()
        if df_track_all.empty: return None
            
        entry_bar = df_full.iloc[index_signal]
        # El precio de entrada es el Open de la barra siguiente a la señal
        entry_price = df_track_all.iloc[0]['Open']
        atr_value = entry_bar['ATR_200']
        timeframe_value = df_full.attrs.get('mt5_timeframe')
        bar_duration_minutes = TIMEFRAME_MINUTES.get(timeframe_value, 0)
        
    except IndexError: return None

    metrics = {
        'Max_Gain_Price': entry_price, 'Max_Gain_Pips': 0.0, 'Max_Gain_Time_Bars': 0,
    }
    
    time_to_atr_target = {target: None for target in ATR_TARGETS}
    max_gain_bar_index_in_track = -1
    # SL Estadístico (1 ATR)
    sl_stat_level = entry_price - (direction * atr_value) 
    last_tracked_index = len(df_track_all) - 1
    sl_stat_touched = False
    
    total_time_minutes = 0.0
    
    for i in range(len(df_track_all)):
        bar = df_track_all.iloc[i]
        bar_entry_price = bar['Open'] # Open de la barra actual (para el cálculo fraccional)
        
        # --- B. VERIFICACIÓN DE CIERRE POR SL ESTADÍSTICO (1 ATR) ---
        if direction == 1: # Compra: Low toca o cae por debajo del SL
            if bar['Low'] <= sl_stat_level:
                sl_stat_touched = True
                last_tracked_index = i
                break
        else: # Venta: High toca o sube por encima del SL
            if bar['High'] >= sl_stat_level:
                sl_stat_touched = True
                last_tracked_index = i
                break
                
        # --- C. SEGUIMIENTO DE MÁXIMA GANANCIA Y TIEMPO A OBJETIVOS ---
        
        # Ganancia Absoluta (en el high/low más extremo de la barra)
        current_max_gain_abs = bar['High'] - entry_price if direction == 1 else entry_price - bar['Low']
        
        for target in ATR_TARGETS:
            if time_to_atr_target[target] is None:
                target_level = entry_price + (direction * target * atr_value)
                
                # Nivel de ganancia alcanzado en esta barra
                if (direction == 1 and bar['High'] >= target_level) or (direction == -1 and bar['Low'] <= target_level):
                    
                    if direction == 1:
                        distance_from_bar_open = target_level - bar_entry_price
                        bar_range_in_bar = bar['High'] - bar_entry_price
                    else:
                        distance_from_bar_open = bar_entry_price - target_level
                        bar_range_in_bar = bar_entry_price - bar['Low']

                    if bar_range_in_bar > 0:
                        # Cálculo del tiempo fraccional por interpolación lineal
                        fractional_time = bar_duration_minutes * (distance_from_bar_open / bar_range_in_bar)
                        time_to_atr_target[target] = total_time_minutes + fractional_time
                    else:
                         time_to_atr_target[target] = total_time_minutes


        if current_max_gain_abs * multiplier > metrics['Max_Gain_Pips']:
            metrics['Max_Gain_Pips'] = current_max_gain_abs * multiplier
            metrics['Max_Gain_Price'] = bar['High'] if direction == 1 else bar['Low']
            metrics['Max_Gain_Time_Bars'] = i + 1
            max_gain_bar_index_in_track = i
            
        total_time_minutes += bar_duration_minutes

    # --- POST-PROCESAMIENTO: SL_Min_ATR y Time_to_SL_Min_Mins ---
    atr_pips = atr_value * multiplier
    metrics['Max_Gain_ATR'] = metrics['Max_Gain_Pips'] / atr_pips
    
    time_to_sl_min_mins = 0.0
    
    # Recalcular SL_Min_ATR (El Drawdown MÁXIMO ocurrido ANTES o EN la barra del Max Gain)
    if metrics['Max_Gain_Pips'] > 0.0 and max_gain_bar_index_in_track > -1:
        # Solo miramos hasta la barra donde se alcanzó el Max Gain
        df_range = df_track_all.iloc[0 : max_gain_bar_index_in_track + 1]
        
        if direction == 1:
            max_drawdown_price = df_range['Low'].min()
            max_drawdown_abs_min = entry_price - max_drawdown_price
        else:
            max_drawdown_price = df_range['High'].max()
            max_drawdown_abs_min = max_drawdown_price - entry_price
            
        sl_min_atr_raw = max_drawdown_abs_min / atr_value
        metrics['SL_Min_ATR'] = min(sl_min_atr_raw, 0.99999)
        if metrics['SL_Min_ATR'] < 0.001: metrics['SL_Min_ATR'] = 0.0
        
        # 2. CALCULAR TIME_TO_SL_MIN_MINS
        if max_drawdown_abs_min > 0.0:
            # Encontrar el índice:
            if direction == 1:
                # Filtrar por el precio Low más bajo (Max Drawdown Price)
                dd_bars = df_range[df_range['Low'] == max_drawdown_price]
            else:
                # Filtrar por el precio High más alto
                dd_bars = df_range[df_range['High'] == max_drawdown_price]
            
            # Tomar el PRIMER índice donde se alcanzó el máximo DD
            if not dd_bars.empty:
                idx_dd_timestamp = dd_bars.index[0]
                bar_dd = df_full.loc[idx_dd_timestamp]
                
                # Posición dentro de df_track_all
                # Se utiliza df_track_all.index.get_loc para obtener el índice basado en la marca de tiempo.
                bar_index_in_track = df_track_all.index.get_loc(idx_dd_timestamp)
                
                # Tiempo de barras anteriores completas
                time_to_prev_bar = bar_index_in_track * bar_duration_minutes
                bar_entry_price_dd = bar_dd['Open']
                
                # Tiempo fraccional en la barra de DD (usando la distancia del Open)
                if direction == 1:
                    bar_range_in_bar_dd = bar_entry_price_dd - bar_dd['Low']
                    if bar_range_in_bar_dd > 0:
                        fractional_time = bar_duration_minutes * (bar_entry_price_dd - max_drawdown_price) / bar_range_in_bar_dd
                        time_to_sl_min_mins = time_to_prev_bar + fractional_time
                    else:
                        time_to_sl_min_mins = time_to_prev_bar
                else:
                    bar_range_in_bar_dd = bar_dd['High'] - bar_entry_price_dd
                    if bar_range_in_bar_dd > 0:
                        fractional_time = bar_duration_minutes * (max_drawdown_price - bar_entry_price_dd) / bar_range_in_bar_dd
                        time_to_sl_min_mins = time_to_prev_bar + fractional_time
                    else:
                        time_to_sl_min_mins = time_to_prev_bar
        
    else:
        # Si no hubo ganancia, o el DD fue cero, SL_Min_ATR y tiempo son cero
        max_drawdown_abs_min = 0.0
        metrics['SL_Min_ATR'] = 0.0
        time_to_sl_min_mins = 0.0


    metrics['SL_Min_Pips'] = metrics['SL_Min_ATR'] * atr_pips
    
    # Calcular Potenciales (Usando MAX_POTENTIAL_REPLACEMENT para evitar 'inf')
    pot_sl_min = metrics['Max_Gain_ATR'] / metrics['SL_Min_ATR'] \
                 if metrics['SL_Min_ATR'] > 0 \
                 else MAX_POTENTIAL_REPLACEMENT
    pot_sl_stat = metrics['Max_Gain_ATR'] / 1.0
    
    if metrics['Max_Gain_ATR'] < 0.5:
        # Marcar como inválido si la ganancia máxima no alcanzó 0.5 ATR
        pot_sl_min = -1.0
        pot_sl_stat = -1.0

    
    results = {
        'Nro_Op': np.nan, 'Asset': symbol, 'TimeFrame': TIMEFRAME_NAMES.get(timeframe_value, 'N/A'),
        'Direccion': 'BUY' if direction == 1 else 'SELL', 'Entrada': entry_price,
        'ATR_200_Pips': atr_pips, 'ATR_Half_Pips': atr_pips / 2,
        
        # MÉTRICAS DE GANANCIA/POTENCIAL
        'Max_Gain_Price': metrics['Max_Gain_Price'], 'Max_Gain_Pips': metrics['Max_Gain_Pips'],
        'Max_Gain_ATR': metrics['Max_Gain_ATR'],
        
        # MÉTRICAS DE SL_MIN (Incluyendo el nuevo tiempo)
        'SL_Min_Pips': metrics['SL_Min_Pips'],
        'SL_Min_ATR': metrics['SL_Min_ATR'],
        'Time_to_SL_Min_Mins': time_to_sl_min_mins,
        
        'SL_Stat_Level': sl_stat_level,
        'Pot_SL_Min': pot_sl_min, 'Pot_SL_Stat': pot_sl_stat,
        'Signal_Time': entry_bar.name
    }
    
    for target in ATR_TARGETS:
        col_name = f'Time_to_{target}ATR_Mins'
        results[col_name] = time_to_atr_target[target]
            
    return results


#
#-----------------------------------------------------------------
# 7. FUNCIÓN PRINCIPAL DE BACKTESTING (PARALELIZADA)
#-----------------------------------------------------------------

def calculate_adx(df, period):
    """Calcula el ADX, PDI y NDI."""
    # True Range (TR)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(np.abs(df['High'] - df['Close'].shift(1)), 
                                     np.abs(df['Low'] - df['Close'].shift(1))))
    
    # Directional Movement (DM)
    df['HD'] = df['High'] - df['High'].shift(1)
    df['LD'] = df['Low'].shift(1) - df['Low']
    
    df['PDI'] = np.where((df['HD'] > df['LD']) & (df['HD'] > 0), df['HD'], 0)
    df['NDI'] = np.where((df['LD'] > df['HD']) & (df['LD'] > 0), df['LD'], 0)
    
    # Exponential Moving Average of DM and TR
    df['PDI_EMA'] = df['PDI'].ewm(span=period, adjust=False).mean()
    df['NDI_EMA'] = df['NDI'].ewm(span=period, adjust=False).mean()
    df['TR_EMA'] = df['TR'].ewm(span=period, adjust=False).mean()

    # DI (Directional Indicator)
    # Evitar división por cero
    df['PDI'] = np.where(df['TR_EMA'] != 0, 100 * (df['PDI_EMA'] / df['TR_EMA']), 0)
    df['NDI'] = np.where(df['TR_EMA'] != 0, 100 * (df['NDI_EMA'] / df['TR_EMA']), 0)

    # DX (Directional Index)
    df['DX'] = np.where((df['PDI'] + df['NDI']) != 0, 
                        100 * np.abs(df['PDI'] - df['NDI']) / (df['PDI'] + df['NDI']), 0)

    # ADX
    df[f'ADX_{period}'] = df['DX'].ewm(span=period, adjust=False).mean()

    # Se necesita df[f'ADX_{period}']
    return df

def calculate_ichimoku(df, tenkan_period, kijun_period, senkou_b_period):
    """Calcula las líneas Tenkan, Kijun y las Spans de Ichimoku."""
    
    # Tenkan Sen (Conversión Line)
    low_min_tenkan = df['Low'].rolling(window=tenkan_period).min()
    high_max_tenkan = df['High'].rolling(window=tenkan_period).max()
    df['Tenkan_Sen'] = (high_max_tenkan + low_min_tenkan) / 2
    
    # Kijun Sen (Base Line)
    low_min_kijun = df['Low'].rolling(window=kijun_period).min()
    high_max_kijun = df['High'].rolling(window=kijun_period).max()
    df['Kijun_Sen'] = (high_max_kijun + low_min_kijun) / 2
    
    # Senkou Span A (Leading Span A) - Promedio de Tenkan y Kijun, desplazado 26 períodos
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B) - Máximo/Mínimo de 52 períodos, desplazado 26 períodos
    low_min_senkou_b = df['Low'].rolling(window=senkou_b_period).min()
    high_max_senkou_b = df['High'].rolling(window=senkou_b_period).max()
    df['Senkou_Span_B'] = ((high_max_senkou_b + low_min_senkou_b) / 2).shift(kijun_period)
    
    # Se necesitan df['Tenkan_Sen'], df['Kijun_Sen'], df['Senkou_Span_A'], df['Senkou_Span_B']
    return df


def run_backtest_task(symbol, timeframe, end_date, ATR_PERIOD, strategy_name):

    timeframe_str = TIMEFRAME_NAMES.get(timeframe, str(timeframe))
    sheet_name = f"{symbol}_{timeframe_str}_{strategy_name}"[:31]

    print(f"\n🔄 INICIANDO TAREA: {symbol} | TF: {timeframe_str} | ESTRATEGIA: {strategy_name}")

    # Inicializar y cerrar MT5 dentro de cada proceso hijo
    # NOTA: En algunos entornos, la inicialización puede fallar si ya está inicializado.
    # Se añade un intento simple de manejo de errores
    try:
        if not mt5.initialize():
             # Intento de re-inicialización o pasar si ya está inicializado por otro proceso
            mt5.shutdown()
            if not mt5.initialize():
                print(f"❌ Fallo al inicializar MT5 en el proceso. Tarea: {sheet_name}. Error: {mt5.last_error()}")
                return None, sheet_name
    except Exception:
        pass # A veces ya está inicializado y solo genera una advertencia

    forward_bars_needed = get_forward_bars_count(timeframe)
    
    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        0,
        MAX_BARS_TO_DOWNLOAD
    )
    
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"❌ No se pudieron obtener datos para {symbol} en {timeframe_str}.")
        return None, sheet_name

    df_base = pd.DataFrame(rates)
    df_base['time'] = pd.to_datetime(df_base['time'], unit='s')
    df_base.set_index('time', inplace=True)
    df_base.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'Real_Volume']
    df_base = df_base[['Open', 'High', 'Low', 'Close', 'Volume']]

    df_base.attrs['mt5_timeframe'] = timeframe
    df_base.attrs['symbol'] = symbol

    # --- 1. PRE-CÁLCULO DE INDICADORES BASE (Vectorizado con Pandas) ---
    df = df_base.copy()
    df.attrs['mt5_timeframe'] = timeframe
    df.attrs['symbol'] = symbol

    def calculate_tr(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))

    df['TR'] = calculate_tr(df['High'], df['Low'], df['Close'].shift(1))
    df['ATR_200'] = df['TR'].ewm(span=ATR_PERIOD, adjust=False).mean()
    
    # EMA's ORIGINALES
    df[f'EMA_{EMA_SUPPORT_RESISTANCE_FAST}'] = df['Close'].ewm(span=EMA_SUPPORT_RESISTANCE_FAST, adjust=False).mean()
    df[f'EMA_{EMA_SUPPORT_RESISTANCE_SLOW}'] = df['Close'].ewm(span=EMA_SUPPORT_RESISTANCE_SLOW, adjust=False).mean()
    df[f'EMA_{EMA_TREND_PERIOD}'] = df['Close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    df[f'EMA_{EMA_TREND_LONG}'] = df['Close'].ewm(span=EMA_TREND_LONG, adjust=False).mean()
    
    # BB y RSI 14
    df[f'BB_Mid_{BB_PERIOD}'] = df['Close'].rolling(window=BB_PERIOD).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=BB_PERIOD).std()
    df['BB_Upper'] = df[f'BB_Mid_{BB_PERIOD}'] + (df['BB_StdDev'] * BB_DEV)
    df['BB_Lower'] = df[f'BB_Mid_{BB_PERIOD}'] - (df['BB_StdDev'] * BB_DEV)
    
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA_Fast'] = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD_Line'] = df['EMA_Fast'] - df['EMA_Slow']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

    # Canales ATR
    df[f'EMA_{ATR_CHANNEL_PERIOD}'] = df['Close'].ewm(span=ATR_CHANNEL_PERIOD, adjust=False).mean()
    df['TR_20'] = calculate_tr(df['High'], df['Low'], df['Close'].shift(1))
    df['ATR_20'] = df['TR_20'].ewm(span=ATR_CHANNEL_PERIOD, adjust=False).mean()
    df['ATR_Upper_Channel'] = df[f'EMA_{ATR_CHANNEL_PERIOD}'] + (df['ATR_20'] * ATR_CHANNEL_MULT)
    df['ATR_Lower_Channel'] = df[f'EMA_{ATR_CHANNEL_PERIOD}'] - (df['ATR_20'] * ATR_CHANNEL_MULT)

    # -------------------------------------------------------------
    # CÁLCULOS ADICIONALES PARA NUEVAS ESTRATEGIAS
    # -------------------------------------------------------------

    # ADX (para ADX_Momentum_Break)
    df = calculate_adx(df, ADX_PERIOD)
    df[f'Momentum_{MOMENTUM_PERIOD}'] = df['Close'].diff(MOMENTUM_PERIOD)

    # Ichimoku (para Ichimoku_Cloud_Trend)
    df = calculate_ichimoku(df, ICHI_TENKAN, ICHI_KIJUN, ICHI_SENKOU_SPAN_B)

    # RSI 5 (para RSI_Fractal_Reversal)
    delta_5 = df['Close'].diff(1)
    gain_5 = (delta_5.where(delta_5 > 0, 0)).fillna(0)
    loss_5 = (-delta_5.where(delta_5 < 0, 0)).fillna(0)
    avg_gain_5 = gain_5.ewm(com=RSI_REVERSAL_PERIOD-1, min_periods=RSI_REVERSAL_PERIOD).mean()
    avg_loss_5 = loss_5.ewm(com=RSI_REVERSAL_PERIOD-1, min_periods=RSI_REVERSAL_PERIOD).mean()
    rs_5 = avg_gain_5 / avg_loss_5
    df['RSI_5'] = 100 - (100 / (1 + rs_5))
    
    # ATR 10 y ATR 50 (para Volatility_Compression_Entry)
    df['TR_10'] = calculate_tr(df['High'], df['Low'], df['Close'].shift(1))
    df['ATR_10'] = df['TR_10'].ewm(span=10, adjust=False).mean()
    df['TR_50'] = calculate_tr(df['High'], df['Low'], df['Close'].shift(1))
    df['ATR_50'] = df['TR_50'].ewm(span=VOL_COMP_PERIOD, adjust=False).mean()

    # --- 2. GENERACIÓN DE SEÑALES ---
    if strategy_name == STRATEGY_EMA_REVERSION_NAME:
        df_signals_only = generate_signals_ema_reversion(df.copy())
    elif strategy_name == STRATEGY_TRIPLE_FILTER_NAME:
        df_signals_only = generate_signals_triple_filter(df.copy())
    elif strategy_name == STRATEGY_SR_REVERSION_NAME:
        df_signals_only = generate_signals_sr_reversion(df.copy())
    elif strategy_name == STRATEGY_ATR_REVERSION_NAME:
        df_signals_only = generate_signals_atr_reversion(df.copy())
    elif strategy_name == STRATEGY_MACD_TREND_NAME:
        df_signals_only = generate_signals_macd_trend(df.copy())
    elif strategy_name == STRATEGY_ATR_BREAKOUT_NAME:
        df_signals_only = generate_signals_atr_breakout(df.copy())
    # NUEVAS ESTRATEGIAS
    elif strategy_name == STRATEGY_ADX_MOMENTUM_NAME:
        df_signals_only = generate_signals_adx_momentum_break(df.copy())
    elif strategy_name == STRATEGY_ICHIMOKU_CLOUD_NAME:
        df_signals_only = generate_signals_ichimoku_cloud_trend(df.copy())
    elif strategy_name == STRATEGY_RSI_FRACTAL_REVERSAL_NAME:
        df_signals_only = generate_signals_rsi_fractal_reversal(df.copy())
    elif strategy_name == STRATEGY_VOLATILITY_COMPRESSION_NAME:
        df_signals_only = generate_signals_volatility_compression_entry(df.copy())
    else:
        print(f"❌ Estrategia '{strategy_name}' no reconocida.")
        return None, sheet_name

    entry_signals = df_signals_only[df_signals_only['Signal'] != 0.0].copy()
    entry_signals_count = len(entry_signals)

    # --- 3. SEGUIMIENTO Y CÁLCULO DE MÉTRICAS ---
    all_metrics = []
    pip_multiplier = get_pip_multiplier(symbol)
    
    for i, (index, row) in enumerate(entry_signals.iterrows()):
        # El índice de la señal es el índice real en el DataFrame (número de barra)
        signal_bar_index = int(row['Signal_Index'])
        direction = row['Signal']
        
        metrics = track_trade_metrics(
            df_full=df,
            index_signal=signal_bar_index,
            direction=direction,
            multiplier=pip_multiplier,
            forward_bars=forward_bars_needed,
            symbol=symbol
        )
        
        if metrics is not None:
            metrics['Nro_Op'] = i + 1
            metrics['Asset'] = symbol
            all_metrics.append(metrics)
            
            
    # --- 4. CONSOLIDACIÓN FINAL ---
    if all_metrics:
        df_results = pd.DataFrame(all_metrics)
        
        # Columnas de salida
        cols_to_display = [
            'Nro_Op', 'Asset', 'TimeFrame', 'Direccion', 'Entrada',
            'ATR_200_Pips',
            
            'SL_Min_Pips', 'SL_Min_ATR', 'Time_to_SL_Min_Mins', # MÉTRICAS DE RIESGO REQUERIDO
            
            'Max_Gain_Price', 'Max_Gain_Pips', 'Max_Gain_ATR',
            'SL_Stat_Level',
            'Pot_SL_Min', 'Pot_SL_Stat', 'Signal_Time'
        ] + [f'Time_to_{target}ATR_Mins' for target in ATR_TARGETS]
        
        df_results = df_results[cols_to_display]
        # Reemplazar el infinito generado por MAX_POTENTIAL_REPLACEMENT con el valor constante
        df_results.replace([np.inf, -np.inf], MAX_POTENTIAL_REPLACEMENT, inplace=True)

        print(f"✅ TAREA FINALIZADA: {symbol} | TF: {timeframe_str} | Señales: {entry_signals_count}")
        return df_results, sheet_name
    else:
        print(f"⚠️ TAREA FINALIZADA: {symbol} | TF: {timeframe_str} | No se encontraron señales.")
        return None, sheet_name


#
#-----------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL DEL PROGRAMA (MÁXIMA VELOCIDAD)
#-----------------------------------------------------------------

if __name__ == '__main__':
    
    if not mt5.initialize():
        print(f"initialize() falló, error code: {mt5.last_error()}")
        exit()
    mt5.shutdown()
    print("MT5 (Proceso Principal) preparado.")

    STRATEGIES_TO_EVALUATE = [
        STRATEGY_EMA_REVERSION_NAME,
        STRATEGY_TRIPLE_FILTER_NAME,
        STRATEGY_SR_REVERSION_NAME,
        STRATEGY_ATR_REVERSION_NAME,
        STRATEGY_MACD_TREND_NAME,
        STRATEGY_ATR_BREAKOUT_NAME,
        # NUEVAS ESTRATEGIAS A EVALUAR
        STRATEGY_ADX_MOMENTUM_NAME,
        STRATEGY_ICHIMOKU_CLOUD_NAME,
        STRATEGY_RSI_FRACTAL_REVERSAL_NAME,
        STRATEGY_VOLATILITY_COMPRESSION_NAME
    ]
    all_results = {}
    tasks = []

    for symbol_to_test in SYMBOLS_TO_EVALUATE:
        for tf_to_test in TIMEFRAMES_TO_EVALUATE:
            for strategy_name in STRATEGIES_TO_EVALUATE:
                tasks.append((symbol_to_test, tf_to_test, end_date, ATR_PERIOD, strategy_name))
    
    total_tasks = len(tasks)
    print(f"🚀 Creando {total_tasks} tareas. Ejecutando en paralelo con {MAX_WORKERS} procesos (cores)...")

    start_time = time.time()
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            futures = [executor.submit(run_backtest_task, *task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                df_metrics, sheet_name = future.result()

                if df_metrics is not None and not df_metrics.empty:
                    all_results[sheet_name] = df_metrics
    except Exception as e:
        print(f"❌ Error crítico en el Pool de Procesos: {e}")

    end_time = time.time()
    time_taken = end_time - start_time
    
    output_dir = "Backtest_Results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"REPORTE_METRICAS_PROCESO_PARALELO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    if all_results:
        try:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                for sheet_name, df_data in all_results.items():
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n\n========================================================")
            print(f"✅ EXPORTACIÓN FINALIZADA. TIEMPO TOTAL: {time_taken:.2f} segundos.")
            print(f"Archivo: {filename}")
            print(f"Total de reportes/hojas creadas: {len(all_results)}")
            print(f"========================================================")

        except Exception as e:
            print(f"❌ Error al exportar el archivo consolidado de Excel: {e}")
    else:
        print("\n⚠️ No se generaron resultados de métricas válidas para exportar.")


    print(f"\nProceso principal completado. 🎉")
