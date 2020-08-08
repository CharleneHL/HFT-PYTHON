"""
Created on 5 Jul 2020

@author: huailin

"""

import warnings
import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client

##import sys
##sys.path.insert(0,'Desktop/')

from dateutil import parser
from datetime import datetime, timedelta
from ultilities import timestr
from indicators import typical_price,atr, keltner_channel,RSI, OBV, Bbands
from tradingpy import Backtester, Signal, SIDE, QUANTPRE, PRICEPRE
PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2
if PY3: _pip = 'pip3'
else: _pip = 'pip'


plt.rcParams.update({'font.size': 20})
plt.rcParams.update({"hist.bins": 200})


apiKey = ''
scrKey = ''

client = Client(apiKey,scrKey)
##
def moreBackdata(symbol,lookback): #eg. symbol = 'BTCUSDT', lookback = '2 days UTC ago'  ####
    #retrieve klines

    run_klines = np.array(client.get_historical_klines(symbol, Client.KLINE_INTERVAL_3MINUTE, lookback))   ####

    #Convert output from strings to float
    kline_matrix=np.asfarray(run_klines,float)
    #Store data
    data = {'_t': kline_matrix[:, 0], '_o': kline_matrix[:, 1], '_h': list(kline_matrix[:, 2]), '_l': kline_matrix[:, 3], '_c': kline_matrix[:, 4], '_v': kline_matrix[:, 5], 'CloseTime': kline_matrix[:, 6], 'QuoteVolume': kline_matrix[:, 7], 'BaseVolume': kline_matrix[:, 9]}

    df = pd.DataFrame(data)
    return df

summary = dict()
summary['Symbol'] = []
summary['Trading Time'] = []
summary['Gross Profit'] = []
summary['Gross Loss'] = []
summary['Commission'] = []
summary['Net Profit'] = []
summary['Profit Factor'] = []
summary['Total Numer of Trades'] = []
summary['Numer of Wins'] = []
summary['Number of Losses'] = []
summary['Average Time In Position'] = []


#symList = [x for x in QUANTPRE.keys()]

symList = ['BCHUSDT']
kl_size = '3m'
n = 7
wd = 9
atrnum = 1.5

num = 0

for symbol in symList:

    data = moreBackdata(symbol, '{} days UTC ago'.format(n) )
    data = data.set_index('_t')
    #print(data)

    # Setup the parameters for signals

    kc_df = keltner_channel(data, window=wd, atrs=atrnum)
    bb_df = Bbands(data['_c'], window=wd, numsd=2.5)
    # -----------------------------------------------------
    # incorporate other indicators into the dataframe   ###

    data['obv'] = OBV(data['_c'], data['_v'])
    data['rsi'] = RSI(data['_c'], period=wd)
    data['kc_upper'] = kc_df['upper']
    data['kc_lower'] = kc_df['lower']
    data['kc_mid'] = kc_df['mid']

    rsi = pd.Series(data['rsi'])
    rsi_crit = dict()
    rsi_crit['lower'] = round(rsi[rsi<30].mean(),2)
    rsi_crit['upper'] = round(rsi[rsi>70].mean(),2)


    data['kc_upper'] = kc_df['upper']
    data['kc_lower'] = kc_df['lower']
    data['kc_mid'] = kc_df['mid']

    data['bb_upper'] = bb_df['upper']
    data['bb_lower'] = bb_df['lower']
    data['bb_mid'] = bb_df['mid']


    # short signals
    crit1 = data['_c'] >= data['kc_upper']
    crit2 = (data['bb_upper']-data['kc_upper']) > (data['kc_upper']-data['kc_mid'])*1.5
    crit3 = data['rsi'].shift(1) > rsi_crit['upper']
    up_cross = data[ crit1 & crit2 & crit3]

    # long signals
    crit1 = data['_c'] <= data['kc_lower']
    crit2 = (data['kc_lower']-data['bb_lower']) > (data['kc_mid']-data['kc_lower'])*1.5
    crit3 = data['rsi'].shift(1) < rsi_crit['lower']
    dn_cross = data[ crit1 & crit2 & crit3]

    # -----------------------------------------------------

    data['side'] = np.zeros(data.shape[0])
    data.loc[up_cross.index, 'side'] = -1.
    data.loc[dn_cross.index, 'side'] = 1.


    ## set parameters
    order_size = 7
    model_signals = []
    data = data.dropna()

    equity = 50
    max_loss_per_trade = 0.05 * equity


    # declare a Backtester object
    tradeData = data[['rsi', 'obv', 'kc_lower', 'kc_mid', 'kc_upper']].copy()
    tradeData['_p'] = data['_o'].copy()             # trade with open price not close price
    tradeData['_t'] = tradeData.index + 60*1000*5     # remember to change the min
    tradeData = tradeData.dropna(axis=0)

    #print(tradeData)

    ###
    for i in range(data.shape[0]):
        kc_mid = data['kc_mid'].iloc[i]
        side = data['side'].iloc[i]
        new_sig, startTime, price = None, tradeData['_t'].iloc[i], data['_c'].iloc[i]
        expTime = startTime + 5*60*1000

        if side==1.0:
            new_sig = Signal(symbol=symbol, side='BUY', size=order_size, orderType='MARKET', positionSide='LONG', price=price, \
                             startTime=startTime,expTime=expTime)
        elif side==-1.0:
            new_sig = Signal(symbol=symbol, side='SELL', size=order_size, orderType='MARKET', positionSide='SHORT', price=price, \
                                 startTime=startTime, expTime=expTime)

        if new_sig is not None:
            model_signals.append(new_sig)



    backtest = Backtester(symbol=symbol, tradeData=tradeData, initBalance=50, orderSize=order_size, signalList=[])



    # iteration through all the KC signals
    for sig in model_signals.copy():
        # check if there is an open position
        ready = False
        if len(backtest.signalList)==0: ready = True
        else:
            last_trade = backtest.signalList[-1]
            if sig.startTime > last_trade.clsTime: ready = True

        trades = tradeData[tradeData['_t']>=sig.startTime]
        # only enter trade when ready==True
        if ready and (trades.shape[0] > 0):
            print('\n----------------------')
            print( '\nPlace NEW order: \n' + str(sig) )
            sig.set_active(excTime=trades['_t'].iloc[0], excPrice=trades['_p'].iloc[0], excQty=sig.get_quantity())
            print( '\nSet BOOKED order ACTIVE: \n' + str(sig) )
            _exit = False
            while not _exit:
                for i in range(1, trades.shape[0]):
                    _t, _p = trades['_t'].iloc[i], trades['_p'].iloc[i]
                    sig.path_update(lastPrice=_p, lastTime=_t)
                    if sig.side == 'SELL':
                        if (_p <= trades['kc_lower'].iloc[i]) or ((_p - sig.excPrice) * sig.get_quantity() >= max_loss_per_trade) or (_t==trades['_t'].iloc[-1]):
                            _exit = True    ###
                            sig.set_cnt_ordered(cntorderId=0, cntTime=_t, cntType='MARKET')
                            print( '\nPlace COUNTER order: \n' + str(sig) )
                            sig.set_closed(clsTime=_t, clsPrice=_p)  ##
                            print( '\nSet order CLOSED: \n' + str(sig) )
                            break
                    if sig.side == 'BUY':
                        if (_p >= trades['kc_upper'].iloc[i])  or ((sig.excPrice - _p) * sig.get_quantity() >= max_loss_per_trade) or (_t==trades['_t'].iloc[-1]):
                            _exit = True    ###
                            sig.set_cnt_ordered(cntorderId=0, cntTime=_t, cntType='MARKET')
                            print( '\nPlace COUNTER order: \n' + str(sig) )
                            sig.set_closed(clsTime=_t, clsPrice=_p)  ##
                            print( '\nSet order CLOSED: \n' + str(sig) )
                            break
            backtest.add_signal(sig)
    balance = backtest.balance_update()

    #### Buy and Sell Trades from Backtester ###
    #buy = pd.DataFrame(columns=['symbol', 'excTime', 'excPrice', 'clsTime', 'clsPrice'])
    #sell = pd.DataFrame(columns=['symbol', 'excTime', 'excPrice', 'clsTime', 'clsPrice'])
    tradeLog = pd.DataFrame(columns=['symbol', 'commRate', 'excTime','clsTime', 'legSize', 'excPrice', 'clsPrice'])
    for sig in backtest.signalList:
        #if sig.side == 'BUY':
            #buy = buy.append({'symbol':sig.symbol,'excTime': sig.excTime, 'excPrice': sig.excPrice, 'clsTime': sig.clsTime, 'clsPrice': sig.clsPrice}, ignore_index=True)
        #else:
            #sell = sell.append({'symbol':sig.symbol,'excTime': sig.excTime, 'excPrice': sig.excPrice, 'clsTime': sig.clsTime, 'clsPrice': sig.clsPrice}, ignore_index=True)
        tradeLog = tradeLog.append({'symbol':sig.symbol, 'commRate':backtest.commRate['MARKET'], 'excTime': pd.to_datetime(sig.excTime).strftime("%d/%m/%y %H:%M:%S"), 'clsTime': pd.to_datetime(sig.clsTime).strftime("%d/%m/%y %H:%M:%S"), 'legSize': sig.get_quantity(), 'excPrice': sig.excPrice, 'clsPrice': sig.clsPrice}, ignore_index=True)

    trading_time, comm, gross_profit, gross_loss,  profit_factor,  net_profit, total_trades, win, loss, time_av = backtest.summary()


    ##------------------------export summary data--------------------------------------


    if symbol not in summary['Symbol']:
        summary['Symbol'].append(symbol)
        summary['Trading Time'].append(round(trading_time,4))
        #print(summary['Trading Time'])
        summary['Gross Profit'].append(round(gross_profit,4))
        summary['Gross Loss'].append(round(gross_loss,4))
        summary['Commission'].append(round(comm,4))
        summary['Net Profit'].append(round(net_profit,4))
        summary['Profit Factor'].append(round(profit_factor,3))
        summary['Total Numer of Trades'].append(round(total_trades,4))
        summary['Numer of Wins'].append(win)
        summary['Number of Losses'].append(loss)
        summary['Average Time In Position'].append(round(time_av,3))

    num = num +1
    print(num)
summary_df = pd.DataFrame(summary)
#print(summary_df)
filepath = '/Users/kiki/Desktop/KC_py/KC_BACKTESTER/'
#summary_file = filepath + '-%sdays-%s-%swd-summary.csv'% (n, kl_size,wd)
#summary_df.to_csv(summary_file)
tradeLog_file = filepath + '-%sdays-%s-%swd-summary.csv'% (n, kl_size,wd)
tradeLog.to_csv(tradeLog_file)
