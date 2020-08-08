"""
Created on 5 Jul 2020

@author: huailin

"""

import numpy as np
import pandas as pd

def typical_price(bars):
    res = (bars['_h'] + bars['_l'] + bars['_c']) / 3.
    return pd.Series(index=bars.index, data=res)

def rolling_mean(series, window, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return series.rolling(window).mean()
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except BaseException:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()

def true_range(bars):
    return pd.DataFrame({   "hl": bars['_h'] - bars['_l'],
                            "hc": abs(bars['_h'] - bars['_c'].shift(1)),
                            "lc": abs(bars['_l'] - bars['_c'].shift(1))     }).max(axis=1)

def atr(bars, window=14, exp=False):
    tr = true_range(bars)
    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)
    return pd.Series(res)

def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)

def keltner_channel(bars, window, atrs): # exponential
    typical_mean = rolling_weighted_mean(typical_price(bars), window)
    atrval = atr(bars, window, exp=True) * atrs

    upper = typical_mean + atrval
    lower = typical_mean - atrval

    return pd.DataFrame(index=bars.index, data={    'upper': upper.values,
                                                    'mid': typical_mean.values,
                                                    'lower': lower.values   })

def Bbands(df, window=None, width=None, numsd=None):
    '''
    Returns average, upper band, and lower band
    '''
    ave = df.rolling(window).mean()
    sd = df.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return ave, upband, dnband
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return pd.DataFrame(index=df.index, data={    'upper': upband.values,
                                                        'mid': ave.values,
                                                        'lower': dnband.values   })

#def RSI(close, window):
    #close= closeprice dataframe
    #delta = close.diff()
    #delta = delta[1:]
    #up, down = delta.copy(), delta.copy()
    #up[up < 0] = 0
    #down[down > 0] = 0

    # Calculate the EWMA
    #roll_up1 = up.ewm(span=window).mean()
    #roll_down1 = down.abs().ewm(span=window).mean()

    # Calculate the RSI based on EWMA
    #RS1 = roll_up1 / roll_down1
    #RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    #return RSI1

def RSI(df, period=14):  #  df= close df
    '''
    Returns RSI values
    '''
    diff = df.diff().dropna()
    first = diff.iloc[:period]
    _rsi = np.zeros(df.shape[0])
    _rsi[:period] = np.nan
    gain = np.zeros(df.shape[0] - period)
    loss = np.zeros(df.shape[0] - period)
    gain[0] = abs(first[first>0].sum())
    loss[0] = abs(first[first<0].sum())
    for i in range(1, len(gain)):
        change = diff.iloc[period+i-1]
        gain[i] = gain[i-1]*(period-1)/period + abs(change*int(change>0))
        loss[i] = loss[i-1]*(period-1)/period + abs(change*int(change<0))
    for i in range(gain.shape[0]):
        if loss[i] == 0:
            _rsi[i+period] = 100
        else:
            _rsi[i+period] = 100 - 100/(1+gain[i]/loss[i])
    return _rsi

def OBV (df_close, df_volume): #dataframe including close price and volume
    OBVarray = np.where(df_close > df_close.shift(1), df_volume,
    np.where(df_close < df_close.shift(1), -df_volume, 0)).cumsum()

    return OBVarray
