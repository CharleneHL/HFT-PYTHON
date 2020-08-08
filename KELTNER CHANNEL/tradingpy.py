# -*- coding: utf-8 -*-
"""
Created on 5 Jul 2020

@author: huailin

CREDIT: This is a modified version from tranl.
        For more information, please visit: https://github.com/lambdamirror/Algo-Trading-In-Python
"""

import time, sys, math
import numpy as np
import pandas as pd

from tqdm import tqdm
from binancepy import MarketData
from indicators import keltner_channel, OBV, RSI, Bbands
from ultilities import timestr, print_

###TRADING RULES
QUANTPRE = {  'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 2, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3, \
                'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3, \
                'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1, \
                'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0, \
                'COMPUSDT':3, 'ALGOUSDT':1, 'ZILUSDT':0, 'KNCUSDT':0, 'ZRXUSDT':1, 'OMGUSDT':1,'DOGEUSDT':0,\
                'SXPUSDT':1, 'LENDUSDT':1, 'KAVAUSDT': 1,'RLCUSDT':1}
PRICEPRE = {  'BTCUSDT': 2, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2, \
              'TRXUSDT': 5, 'ETCUSDT':3, 'LINKUSDT': 3  , 'XLMUSDT': 5, 'ADAUSDT': 5, 'XMRUSDT': 2, \
              'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 3, 'ATOMUSDT': 3, 'ONTUSDT': 4, \
              'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6,\
              'COMPUSDT':2, 'ALGOUSDT':4,  'ZILUSDT':5, 'KNCUSDT':5, 'ZRXUSDT':4, 'OMGUSDT':4, 'DOGEUSDT':6,\
              'SXPUSDT':4, 'LENDUSDT':4, 'KAVAUSDT': 4,'RLCUSDT':4}

SIDE = {'BUY': 1.0, 'SELL': -1.0}

min_in_ms = int(60*1000)
sec_in_ms = 1000

###%%%

class Portfolio:
    def __init__( self,
                  client,
                  tradeIns = []):
        '''
        Portfolio class
        '''
        self.client = client
        self.tradeIns = tradeIns.copy()
        self.equity = 0
        self.orderSize = 0
        self.equityDist = {'BUY': 0, 'SELL': 0}
        self.locks = { 'BUY': [], 'SELL': []}

    def equity_distribution(self, longPct=0.5, shortPct=0.5, currency='USDT', orderPct=0.02):
        '''
        Retrun number of buy/sell orders with currenty equity

            longPct : percentage of equity assigned for buying

            shortPct : percentage of equity assigned for selling

            orderPct : percentage of equity for a single order
        '''
        balance = self.client.balance()
        equity, available = 0, 0
        for b in balance:
            if b['asset']==currency:
                equity, available = float(b['balance']), float(b['withdrawAvailable'])
                break
        long_equity = longPct*equity
        short_equity = shortPct*equity

        info = pd.DataFrame(self.client.position_info())
        short_info = info[info['positionAmt'].astype(float) < 0]
        long_info = info[info['positionAmt'].astype(float) > 0]
        short_position = abs(short_info['positionAmt'].astype(float) @ short_info['entryPrice'].astype(float))
        long_position = abs(long_info['positionAmt'].astype(float) @ long_info['entryPrice'].astype(float))

        self.orderSize = round(orderPct*equity, 2)
        long_order = int((long_equity - long_position)/self.orderSize)
        short_order = int((short_equity - short_position)/self.orderSize)
        self.equityDist = {'BUY': long_order, 'SELL': short_order}
        self.equity = equity
        return long_order, short_order

    def position_locks(self, prelocks={ 'BUY': [], 'SELL': []}):
        '''
        Check for open positions and return a tradable instruments
        '''
        info = self.client.position_info()
        self.locks = prelocks
        for pos in info:
            amt = float(pos['positionAmt'])
            if amt < 0 and not pos['symbol'] in self.locks['SELL']: self.locks['SELL'].append(pos['symbol'])
            elif amt > 0 and not pos['symbol'] in self.locks['BUY']: self.locks['BUY'].append(pos['symbol'])
        drop_out = set(self.locks['SELL']).intersection(self.locks['BUY'])
        for s in drop_out: self.tradeIns.remove(s)
        return self.tradeIns

###%%%

class TradingModel:
    def __init__( self,
                  symbol: str,
                  testnet: bool,
                  modelType: str,
                  marketData,
                  pdObserve: int,
                  features: dict = None,
                  inputData = None,
                  orderSize = 1.0, #USDT
                  maxLoss = 1.0,
                  breath: float = 0.01/100):
        '''
        Trading Model class
        '''
        self.symbol = symbol
        self.testnet = testnet
        self.modelType = modelType
        self.marketData = marketData
        self.pdObserve = pdObserve
        self.inputData = inputData
        self.timeLimit = int(self.pdObserve*10)
        self.orderSize = orderSize
        self.maxLoss = maxLoss
        self.breath = breath
        self.rsi_crit = {'lower': 30, 'upper': 70}
        self.signalLock = []

    def add_signal_lock(self, slock=None):
        '''
        Add a signal to lock positions i.e. abandon BUY/SELL the instrument
        '''
        if (slock is not None) and (not slock in self.signalLock):
            self.signalLock.append(slock)

    def remove_signal_lock(self, slock=None):
        '''
        Remove a signal from lock positions i.e. allows BUY/SELL the instrument
        '''
        if (slock is not None) and (slock in self.signalLock):
            self.signalLock.remove(slock)

    def build_initial_input(self, wd, min_in_candle, period=180):
        '''
        Download and store historical data
        '''
        if self.modelType=='keltner-channel':
            #min_in_candle = 3
            num_klns = period
            t_server = self.marketData.server_time()['serverTime']
            t_start = t_server - num_klns*min_in_candle*60*1000
            df = klns_to_df(self.marketData.candles_data(interval=f'{str(min_in_candle)}m', startTime=t_start, limit=num_klns), ['_t', '_o', '_h', '_l', '_c', '_v'])
            if self.inputData is None:
                self.inputData = df
            else:
                df = df[df['_t'] > self.inputData['_t'].iloc[-1]]
                self.inputData = self.inputData.append(df, ignore_index=True)
            # calculate upper and lower RSI threshholds
            rsi = pd.Series(RSI(self.inputData['_c'], period=wd))
            self.rsi_crit['lower'] = round(rsi[rsi<30].mean(), 2)
            self.rsi_crit['upper'] = round(rsi[rsi>70].mean(), 2)
            if np.isnan(self.rsi_crit['lower']): self.rsi_crit['lower'] = 30
            if np.isnan(self.rsi_crit['upper']): self.rsi_crit['upper'] = 70
        return self.inputData

    def get_last_signal(self, wd = None, atrnum = None, dataObserve=None, lastObserve=None):
        '''
        Process the lastest data for a potential singal
        '''
        if self.modelType=='keltner-channel':
            feats = ['_t', '_o', '_h', '_l', '_c', '_v']
            data = dataObserve[dataObserve['_t'] > self.inputData['_t'].iloc[-1]][feats].copy()
            data = self.inputData.append(data, ignore_index=True).iloc[-30:]  ###
            if lastObserve['_L']!=dataObserve['_L'].iloc[-1]: # "L":last trade ID
                data = data.append({k:v for k,v in lastObserve.items() if k in feats}, ignore_index=True)

            #wd = 14
            #atrnum = 1.5
            kc_df = keltner_channel(data, window=wd, atrs=atrnum)
            bb_df = Bbands(data['_c'], window=wd, numsd=2.5)
            # -----------------------------------------------------
            # incorporate other indicators into the dataframe   ###

            #data['obv'] = OBV(data['_c'], data['_v'])
            #data['rsi'] = RSI(data['_c'], period=wd)
            data['kc_upper'] = kc_df['upper']
            data['kc_lower'] = kc_df['lower']
            data['kc_mid'] = kc_df['mid']

            data['bb_upper'] = bb_df['upper']
            data['bb_lower'] = bb_df['lower']
            data['bb_mid'] = bb_df['mid']


            # short signals
            crit1 = data['_c'] >= data['kc_upper']
            crit2 = (data['bb_upper']-data['kc_upper']) > (data['kc_upper']-data['kc_mid'])*1.5
            crit3 = data['rsi'].shift(1) > self.rsi_crit['upper']
            up_cross = data[ crit1 & crit2 & crit3]

            # long signals
            crit1 = data['_c'] <= data['kc_lower']
            crit2 = (data['kc_lower']-data['bb_lower']) > (data['kc_mid']-data['kc_lower'])*1.5
            crit3 = data['rsi'].shift(1) < self.rsi_crit['lower']
            dn_cross = data[ crit1 & crit2 & crit3]

            data['side'] = np.zeros(data.shape[0])
            data.loc[up_cross.index, 'side'] = -1.
            data.loc[dn_cross.index, 'side'] = 1.
            _side = data['side'].iloc[-1]

            if _side == 1.: # and not 'BUY' in self.signalLock:
                return {'side': 'BUY', 'positionSide': 'LONG', '_t': data['_t'].iloc[-1], '_p': data['_c'].iloc[-1]}
            elif _side == -1.: # and not 'SELL' in self.signalLock:
                return {'side': 'SELL', 'positionSide': 'SHORT', '_t': data['_t'].iloc[-1], '_p': data['_c'].iloc[-1]}
        return None

#%%%%

class Signal:
    def __init__(self,
                 symbol: str,
                 side: str,
                 size: float,
                 orderType: str,
                 positionSide: str = 'BOTH',
                 price: float = None,
                 startTime: int = time.time()*1000,
                 expTime: float = (time.time()+60)*1000,
                 stopLoss: float = None,
                 takeProfit: float = None,
                 timeLimit: int = None, #minutes
                 timeInForce: float = None):
        '''

        Signal class to monitor price movements

        To change currency pair     -> symbol = 'ethusdt'

        To change side              -> side = 'BUY'/'SELL'

        To change order size        -> size = float (dollar amount)

        To change order type        -> orderType = 'MARKET'/'LIMIT'

        To change price             -> price = float (required for 'LIMIT' order type)

        stopLoss, takeProfit -- dollar amount

        To change time in force     -> timeInForce =  'GTC'/'IOC'/'FOK' (reuired for 'LIMIT' order type)

        '''
        self.symbol = symbol
        self.side = side #BUY, SELL
        self.positionSide = positionSide #LONG, SHORT
        self.orderType = orderType #LIMIT, MARKET, STOP, TAKE_PROFIT
        # predefined vars
        self.price = float(price)
        if size < self.price*10**(-QUANTPRE[symbol]):
            size = self.price*10**(-QUANTPRE[symbol])*1.01
        self.size = float(size) #USDT
        self.quantity = round(self.size/self.price, QUANTPRE[self.symbol])
        self.startTime = int(startTime)
        self.expTime = expTime
        # 3 exit barriers
        if stopLoss is not None: self.stopLoss = round(float(stopLoss), 4)
        else: self.stopLoss = None
        if takeProfit is not None: self.takeProfit = round(float(takeProfit), 4)
        else: self.takeProfit = None
        if timeLimit is not None: self.timeLimit = int(timeLimit*sec_in_ms) # miliseconds
        else: self.timeLimit = None

        self.timeInForce = timeInForce
        self.status = 'WAITING' #'ORDERED' #'ACTIVE' #'CNT_ORDERED' #'CLOSED' # 'EXPIRED'
        self.limitPrice, self.orderTime = None, None
        self.excPrice, self.excTime = None, None
        self.cntlimitPrice, self.cntTime, self.cntType = None, None, None
        self.clsPrice, self.clsTime = None, None
        self.orderId = None
        self.cntorderId = None
        self.pricePath = []
        self.exitSign = None
        self.secPos = False   ###

    '''
    Function to check and set STATUS of the signals :
        - WAITING
        - ORDERED
        - ACTIVE
        - CNT_ORDERED
        - CLOSED
        - EXPIRED
    '''
    def is_waiting(self):
        return bool(self.status == 'WAITING')

    def set_waiting(self):
        self.status = 'WAITING'

    def is_ordered(self):
        return bool(self.status == 'ORDERED')

    def set_ordered(self, orderId, orderTime=None, limitPrice=None):
        self.status = 'ORDERED'
        self.orderId = int(orderId)
        self.orderTime, self.limitPrice = orderTime, limitPrice

    def is_active(self):
        return bool(self.status == 'ACTIVE')

    def set_active(self, excTime=time.time()*1000, excPrice=None, excQty: float = None):
        self.excPrice = float(excPrice)
        self.excTime = int(excTime)
        if bool(excQty): self.quantity = round(float(excQty), QUANTPRE[self.symbol])
        self.status = 'ACTIVE'

    def is_cnt_ordered(self):
        return bool(self.status == 'CNT_ORDERED')

    def set_cnt_ordered(self, cntorderId, cntType=None, cntTime=None,  cntlimitPrice=None):
        self.status = 'CNT_ORDERED'
        self.cntorderId = int(cntorderId)
        self.cntType, self.cntTime, self.cntlimitPrice = cntType, cntTime, cntlimitPrice

    def is_closed(self):
        return bool(self.status == 'CLOSED')

    def set_closed(self, clsTime=time.time()*1000, clsPrice=None):
        self.clsTime = int(clsTime)
        if clsPrice is not None: self.clsPrice = float(clsPrice)
        else: self.clsPrice = None
        self.status = 'CLOSED'

    def is_expired(self):
        return bool(self.status == 'EXPIRED')

    def set_expired(self):
        self.status = 'EXPIRED'

    def get_quantity(self):
        '''
        Return quantity
        '''
        return self.quantity

    def counter_order(self):
        '''
        Return counter (close) order with same size but opposite side
        '''
        if self.side=='BUY': side = 'SELL'
        else: side = 'BUY'
        if self.positionSide == 'LONG': posSide = 'SHORT'
        elif self.positionSide =='SHORT': posSide = 'LONG'
        else: posSide = 'BOTH'
        counter = {'side': side, 'positionSide': posSide, 'type': self.orderType, \
                    'amt': self.get_quantity(),'TIF': self.timeInForce}
        return counter

    def path_update(self, lastPrice, lastTime):
        '''
        Update last traded prices to pricePath
        '''
        self.pricePath.append({'timestamp': int(lastTime), 'price': float(lastPrice)})

    def get_price_path(self):
        '''
        Return price movements since the entry
        '''
        return pd.DataFrame(self.pricePath)

    def exit_triggers(self, wd=None, atrnum=None, tradingModel=None, dataObserve=None, lastObserve=None, file=None):
        '''
        Return an exit signal
        '''
        if not self.is_active():
            return None
        else:
            exit_sign = None
            mk_depth = tradingModel.marketData.order_book(limit=5)
            bids = list(float(x[0]) for x in mk_depth['bids'])
            asks = list(float(x[0]) for x in mk_depth['asks'])
            _p = (self.side=='SELL')*bids[0] + (self.side=='BUY')*asks[0]
            pos = SIDE[self.side]*(_p - self.excPrice)*self.get_quantity()

            #data = dataObserve[dataObserve['_t'] > tradingModel.inputData['_t'].iloc[-1]]
            #data = tradingModel.inputData.append(data, ignore_index=True)

            feats = ['_t', '_o', '_h', '_l', '_c', '_v']
            data = dataObserve[dataObserve['_t'] > self.inputData['_t'].iloc[-1]][feats].copy()
            data = self.inputData.append(data, ignore_index=True).iloc[-30:]  ###
            if lastObserve['_L']!=dataObserve['_L'].iloc[-1]: # "L":last trade ID
                data = data.append({k:v for k,v in lastObserve.items() if k in feats}, ignore_index=True)

            data['rsi'] = RSI(data['_c'], period=wd)
            kc_df = keltner_channel(data, window=wd, atrs=atrnum)
            kc_mid = kc_df['mid'].iloc[-1]
            kc_lower = kc_df['lower'].iloc[-1]
            kc_upper = kc_df['upper'].iloc[-1]
            tp_sell = kc_mid*1.005
            tp_buy = kc_mid*0.995


            if SIDE[self.side] < 0:
                crit_1 = data['rsi'].iloc[-1] <= 50
                #crit_1 = ( SIDE[self.side]*(_p - kc_lower) >= 0 )
                #crit_2 = ((_p - self.excPrice) > 0) and ((kc_lower - self.excPrice) >= 0)
            elif SIDE[self.side] > 0:
                crit_1 = data['rsi'].iloc[-1] >= 50
                #crit_1 = ( SIDE[self.side]*(_p - kc_upper) >= 0 )
                #crit_2 = ((_p - self.excPrice) < 0) and ((kc_upper - self.excPrice) <= 0)
            #crit_2 = ( SIDE[self.side]*(_p - self.excPrice) < 0 ) and ( SIDE[self.side]*(kc_mid - self.excPrice) < 0 )
            crit_3 = pos < -tradingModel.maxLoss

            if crit_1 or crit_3:
                if pos > 0 : exit_sign = 'takeProfit'
                else: exit_sign = 'stopLoss'
                print_('\n\t' + self.symbol + '\tcrit1=' + str(crit_1) + '\tcrit3=' + str(crit_3), file)
            self.exitSign = exit_sign
            return exit_sign, pos

    def __str__(self):
        '''
        Print out infomation of the signal

        '''
        s = 'Singal info: ' + self.symbol
        gen_ =  ' status:' + str(self.status) + ' side:' + str(self.side) + ' type:' + str(self.orderType) + ' quantity:' + str(self.get_quantity())
        if self.is_waiting() or self.is_expired():
            id_ = ' Id:None '
            price_ = ' price:' + str(self.price) + ' time:' + timestr(self.startTime, end='s')
        elif self.is_ordered():
            id_ = ' Id:'+ str(self.orderId)
            if self.orderType=='LIMIT':
                price_ = ' price:' + str(self.limitPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(self.startTime, end='s')
            else: price_ = ' type:' + str(self.orderType) + ' time:' + timestr(self.orderTime, end='s')
        elif self.is_active():
            id_ = ' Id:'+ str(self.orderId)
            if self.orderType=='LIMIT':
                price_ = ' price:' + str(self.excPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(self.excTime, end='s')
            else: price_ = ' price:' + str(self.excPrice) + ' time:' + timestr(self.excTime, end='s')
        elif self.is_cnt_ordered():
            gen_ = ' status:' + str(self.status) + ' side:' + str(self.counter_order()['side']) + ' type:' + str(self.cntType) + ' quantity:' + str(self.get_quantity())
            id_ = ' Id:'+ str(self.cntorderId)
            if self.cntType=='LIMIT':
                price_ = ' price:' + str(self.cntlimitPrice) + ' TIF:' + str(self.timeInForce) + ' time:' + timestr(self.cntTime, end='s')
            else: price_ = ' type:' + str(self.cntType) + ' time:' + timestr(self.cntTime, end='s')
        elif self.is_closed():
            gen_ = ' status:' + str(self.status) + ' side:' + str(self.counter_order()['side']) + ' type:' + str(self.cntType) + ' quantity:' + str(self.get_quantity())
            id_ = ' Id: ' + str(self.cntorderId)
            price_ = ' price:' + str(self.clsPrice) + ' time:' + timestr(self.clsTime, end='s')
        if self.stopLoss is None: sl_ = 'None'
        else: sl_ = str(self.stopLoss)
        if self.takeProfit is None: tp_ = 'None'
        else: tp_ = str(self.takeProfit)
        if self.timeLimit is None: tl_ = 'None'
        else: tl_ = str(int(self.timeLimit/sec_in_ms))
        exits_ = ' exits:[' + sl_ + ', ' + tp_ + ', ' + tl_ + ']'
        s += id_ + gen_ + price_ + exits_
        return s

###%%%

def klns_to_df(market_data, feats):
    '''
    Return a pd.DataFrame from candles data received from the exchange
    '''
    fts = list(str(f) for f in feats)
    df_ = pd.DataFrame(market_data, columns = ['_t', '_o', '_h', '_l', '_c', '_v', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df_[['_o', '_h', '_l', '_c', '_v']] = df_[['_o', '_h', '_l', '_c', '_v']].astype(float)
    return df_[fts]



def indcator_df(hist_ohlvc_df, wd, atrnum):  ####
    '''
    incorporate other indicators into the dataframe
    '''

    ind = pd.DataFrame()
    ind['_t'] = hist_ohlvc_df['_t']
    ind['_c'] = hist_ohlvc_df['_c']
    ind['_v'] = hist_ohlvc_df['_v']
    ind['rsi'] = RSI(ind['_c'], period=wd)
    kc_df = keltner_channel(hist_ohlvc_df, window=wd, atrs=atrnum)
    ind['kc_upper'] = kc_df['upper']
    ind['kc_lower'] = kc_df['lower']
    ind['kc_mid'] = kc_df['mid']
    ind['obv'] = OBV(ind['_c'], ind['_v'])
    return ind

def partialcandel_df(exchange, hist_df):  ####

    try:

        curr_ohlvc = dict()
        hist_t = hist_df['_t']
        hist_o = hist_df['_o']
        hist_c = hist_df['_c']
        hist_h = hist_df['_h']
        hist_l = hist_df['_l']
        hist_v = hist_df['_v']

        new = exchange.fetch_ticker(symbol=instrument['symbol'])
        #print(new)
        new_t = new['info']["closeTime"]
        new_o = new['info']["openPrice"]
        new_h = new['info']["highPrice"]
        new_l = new['info']["lowPrice"]
        new_v = new['info']["volume"]
        new_c = new['info']["lastPrice"]

        # update ohlvc dict
        curr_ohlvc = dict()
        curr_ohlvc['_t'] = hist_t.append(pd.Series(new_t))
        curr_ohlvc['_o'] = hist_o.append(pd.Series(new_o))
        curr_ohlvc['_h'] = hist_c.append(pd.Series(new_h))
        curr_ohlvc['_l'] = hist_h.append(pd.Series(new_l))
        curr_ohlvc['_v'] = hist_l.append(pd.Series(new_v))
        curr_ohlvc['_c'] = hist_v.append(pd.Series(new_c))

        curr_df = pd.DataFrame(curr_ohlvc)

        return curr_df

    except ccxt.RequestTimeout:
        print("TRY AGAIN, REQUEST TIME OUT")
