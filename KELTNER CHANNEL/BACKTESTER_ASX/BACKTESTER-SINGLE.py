"""
Created on 5 Jul 2020
@author: huailin
"""

import warnings
import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dateutil import parser
from datetime import datetime, timedelta
from ultilities import timestr
from indicators import typical_price,atr, keltner_channel,RSI, OBV, Bbands
from tradingpy import Backtester, Signal, SIDE, ASX300_List
PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2
if PY3: _pip = 'pip3'
else: _pip = 'pip'


plt.rcParams.update({'font.size': 20})
plt.rcParams.update({"hist.bins": 200})



# Read candles
start = 273
end = 168
symbol_list = ASX300_List[start:]  # 36:'AVH.AX',50:'BVS.AX'(no data downloaded)  #143: ISX; 237:delisted  (no data found) # 176: after 'MYX.AX'(rsi error) #177  #188 after NIC.AX #196: after OBL.AX #196 #199 #202 #205 #221 #224 #236 #237 #246 #268 #271,272 (共296)
filepath = '/Users/kiki/Snap-Notebook/Keltner Channel/Backtester_Stock/ASX/'
#symbol = 'PMV.AX'
## finished: "ALX.AX",'ANZ.AX','BAP.AX','GPT.AX','CBA.AX','SDF.AX', 'SUN.AX','PMV.AX','QUB.AX'; 'COH.AX','SHL.AX','GPT.AX','CQE.AX','HVN.AX','SUL.AX','ANN.AX'
# "ALX.AX",'ANZ.AX','BAP.AX','GPT.AX','CBA.AX','SDF.AX', 'SUN.AX','PMV.AX','QUB.AX'

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

###########################
num = start-1
for symbol in symbol_list:

    kl_size = '5m'
    n = 30  # number of days
    period = int(n*60*24/5)
    data = pd.read_csv(filepath + "-%s-%s.csv" % (symbol, kl_size), index_col=0)[:period]
    #data = data.rename(columns={'_t': 'timestamp', '_o': 'open', '_h': 'high', '_l': 'low', '_c': 'close', '_v': 'volume'})
    data = data.set_index('_t')

    #print(data)

    # Setup the parameters for signals
    wd = 14
    atrnum = 1.5
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

    # long signals
    crit1 = data['_c'] <= data['kc_lower']
    crit2 = (data['kc_lower']-data['bb_lower']) > (data['kc_mid']-data['kc_lower'])*1.5
    crit3 = data['rsi'].shift(1) < rsi_crit['lower']
    dn_cross = data[ crit1 & crit2 & crit3]
    # -----------------------------------------------------

    data['side'] = np.zeros(data.shape[0])
    ##data.loc[up_cross.index, 'side'] = -1.
    data.loc[dn_cross.index, 'side'] = 1.

    # Plot Strat Signals
##    fig, ax = plt.subplots(1, figsize=(30, 15))
##    ax.set_title('Keltner Channel Signals', fontsize=30)
##    ax.plot(data.index, data['_c'], label='close price', color='black')
##    ax.plot(data.index, data['kc_upper'], label='upper', color='green')
##    ax.plot(data.index, data['kc_lower'], label='lower', color='red')
##    ax.plot(data.index, data['kc_mid'], label='ma-%d' % wd, color='grey')
##    ##ax.scatter(up_cross.index, up_cross['_c'], marker='v', color='r', linewidths=7)
##    ax.scatter(dn_cross.index, dn_cross['_c'], marker='^', color='g', linewidths=7)
##    for i in range(0, data.shape[0], 30):
##        ax.axvline(x=data.index[i], linewidth=1, color='grey')
##    ax.legend(loc=2, prop={'size': 20})

    #plt.show()
    #print(data)


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
        ##    elif side==-1.0:
        ##        new_sig = Signal(symbol=symbol, side='SELL', size=order_size, orderType='MARKET', positionSide='SHORT', price=price, \
        ##                         startTime=startTime, expTime=expTime)

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

    # Buy and Sell Trades from Backtester
    buy = pd.DataFrame(columns=['excTime', 'excPrice', 'clsTime', 'clsPrice'])
    sell = pd.DataFrame(columns=['excTime', 'excPrice', 'clsTime', 'clsPrice'])
    for sig in backtest.signalList:
        if sig.side == 'BUY':
            buy = buy.append({'excTime': sig.excTime, 'excPrice': sig.excPrice, 'clsTime': sig.clsTime, 'clsPrice': sig.clsPrice}, ignore_index=True)
    ##    else:
    ##        sell = sell.append({'excTime': sig.excTime, 'excPrice': sig.excPrice, 'clsTime': sig.clsTime, 'clsPrice': sig.clsPrice}, ignore_index=True)
    ##


    # Plot Trades entry/exit and Equity Curve
##    fig, axs = plt.subplots(2, figsize=(30, 15))
##    axs[0].set_title('Singals', fontsize=30)
##    axs[0].plot(tradeData['_t'], tradeData['_p'], label='trade price', color='grey')
##    axs[0].scatter(buy['excTime'], buy['excPrice'], label='BUY entry', color='green', linewidths=5)
##    axs[0].scatter(buy['clsTime'], buy['clsPrice'], label='BUY exit', color='black', linewidths=5)
##    ##axs[0].scatter(sell['excTime'], sell['excPrice'], label='SELL entry', color='red', linewidths=5)
##    ##axs[0].scatter(sell['clsTime'], sell['clsPrice'], label='SELL exit', color='black', linewidths=5)
##    for i in range(0, tradeData.shape[0], 30):
##        axs[0].axvline(x=tradeData['_t'].iloc[i], linewidth=1, color='grey')
##        axs[0].legend()
##
##    axs[1].set_title('Equity Curve', fontsize=30)
##    axs[1].plot(balance['_t'], balance['_b'], label='equity curve', color='black')
##    for i in range(0, tradeData.shape[0], 30):
##        axs[1].axvline(x=tradeData['_t'].iloc[i], linewidth=1, color='grey')
##    axs[1].legend()

    #plt.show()

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
print(summary_df)


summary_file = filepath + '-%sdays-%s-summary.csv' % (n, kl_size)
summary_df.to_csv(summary_file)
