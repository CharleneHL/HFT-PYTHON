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
import websocket
import threading
import json

from tradingpy import PRICEPRE, QUANTPRE, SIDE, Signal
from ultilities import print_, orderstr, timestr, barstr

def wss_run(*args):
    ### threading functions
    def data_stream(*args):
        '''
        First thread to send subscription to the exchange
        '''
        params = [str.lower(ins) + str(s) for ins in insIds for s in stream]
        print_(params, fileout)
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": params, "id": 1 }))
        t1_idx = 0
        while len(endFlag)==0:
            try:
                if len(SymKlns[insIds[0]]) % min_in_candle == 0 and len(SymKlns[insIds[0]]) > t1_idx and len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:  ####
                    client.keepalive_stream()
                    t1_idx = len(SymKlns[insIds[0]])
            except Exception:
                print_('\n\tClose on data_stream()', fileout)
                ws.close()
        return

    def strategy(*args):
        '''
        Second thread to generate signals upon the message from the exchange
        '''
        while len(endFlag)==0 and len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:
            try:
                for symbol in insIds:
                    sym_ = SymKlns[symbol].copy()

                    if len(sym_) > 0:
                        if models[symbol].modelType == 'keltner-channel':
                            data_ob = pd.DataFrame(sym_)
                            model_sig = models[symbol].get_last_signal(wd=wd, atrnum=atrnum, dataObserve=data_ob, lastObserve=LastObs[symbol])
                        else: model_sig = None
                        if model_sig is not None:
                            side, positionSide, startTime = model_sig['side'], model_sig['positionSide'], client.timestamp()
                            expTime, price = startTime + min_in_candle*60*1000, round(model_sig['_p'], PRICEPRE[symbol]) #
                            new_sig = Signal(symbol=symbol, side=side, size=models[symbol].orderSize, orderType='LIMIT', positionSide=positionSide, price=price, startTime=startTime, expTime=expTime, timeInForce='GTC')
                            if in_possition_(Signals[symbol], side='BOTH') or position_count(insIds, Signals, side=side) >= portfolio.equityDist[side]:
                                new_sig.set_expired()
                            else:
                                Signals[symbol].append(new_sig)
                                print_('\n\tFOUND ' + str(new_sig), fileout)

            except Exception:
                print_('\n\tClose on strategy()', fileout)
                ws.close()
        return

    def book_manager(*args):
        '''
        Third thread to excecute/cancel/track the signals generated in strategy()
        '''
        while len(endFlag)==0 and len(SymKlns[insIds[0]]) < models[insIds[0]].pdObserve:
            try:
                time.sleep(1)
                for symbol in insIds:
                    in_position = False
                    last_signal = None
                    for sig in Signals[symbol]:
                        model = models[symbol]
                        sv_time = client.timestamp()
                        if sig.is_waiting():
                            ### Check for EXPIRED order here ###
                            if sv_time > sig.expTime:
                                sig.set_expired()
                                print_('\n\tSet WAITING signal EXPIRED: \n\t' + str(sig), fileout)
                            else:
                                last_signal = sig

                        elif sig.is_ordered():
                            ### Set ACTIVE order here ###
                            in_position = True
                            order_update = client.query_order(symbol, sig.orderId)
                            if order_update['status'] == 'FILLED':
                                sig.set_active(excTime=order_update['updateTime'], excPrice=order_update['avgPrice'])
                                sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)

                            ### to handle EXPIRED and PARTIALLY_FILLED order here ###
                            elif sv_time > order_update['updateTime'] + 2*60*1000:
                                if order_update['status'] == 'PARTIALLY_FILLED':
                                    client.cancel_order(symbol, sig.orderId)    ###
                                    sig.set_active(excTime=order_update['updateTime'], excPrice=order_update['avgPrice'], excQty=order_update['executedQty'])
                                    sig.path_update(lastTime=sig.excTime, lastPrice=sig.excPrice)
                                    print_('\n\tSet BOOKED order ACTIVE: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)
                                else:
                                    client.cancel_order(symbol, sig.orderId)
                                    sig.set_expired()
                                    order_update = client.query_order(symbol, sig.orderId)
                                    print_('\n\tSet BOOKED order EXPIRED: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)

                        elif sig.is_active():
                            ### Control ACTIVE position here ###
                            #### for second position ####
                            in_position = True
                            data_ob = pd.DataFrame(SymKlns[symbol].copy())
                            exit_sign, pos = sig.exit_triggers( wd=wd, atrnum=atrnum, tradingModel=model, dataObserve=data_ob, lastObserve=LastObs[symbol], file=fileout )
                            if exit_sign is not None:
                                print_('\n\tFound ' + str(exit_sign) + '{}\n'.format(round(pos,2)), fileout)
                                cnt_order = sig.counter_order()
                                order = client.new_order(symbol=symbol, side=cnt_order['side'], orderType='MARKET', quantity=cnt_order['amt'], positionSide=sig.positionSide)
                                sig.set_cnt_ordered(cntorderId=order['orderId'], cntType='MARKET', cntTime=order['updateTime'])
                                print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)

                        elif sig.is_cnt_ordered():
                            ### Set CLOSED position here ###
                            in_position = True
                            order_update = client.query_order(symbol, sig.cntorderId)
                            if order_update['status'] == 'FILLED':
                                sig.set_closed(clsTime=order_update['updateTime'], clsPrice=order_update['avgPrice'])
                                print_('\n\tClosed order: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)

                    if (not in_position) and (last_signal is not None):
                        ### Check for ENTRY and place NEW order here ###
                        sig = last_signal
                        if sig.orderType == 'MARKET':
                            order  = client.new_order(symbol=symbol, side=sig.side, orderType=sig.orderType, quantity=sig.get_quantity(), positionSide=sig.positionSide)
                            sig.set_ordered(orderId=order['orderId'], orderTime=order['updateTime'], limitPrice=None)
                            print_('\n\tPlaced NEW order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                        elif sig.orderType=='LIMIT':
                            bids, asks, lim = get_possible_price(model.marketData, sig.side)
                            if lim is not None:
                                order = client.new_order(symbol=symbol, side=sig.side, orderType=sig.orderType, quantity=sig.get_quantity(), positionSide=sig.positionSide, timeInForce='GTC', price=lim)
                                sig.set_ordered(orderId=order['orderId'], orderTime=order['updateTime'], limitPrice=lim)
                                print_('\n\tPlaced NEW order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
            except Exception:
                print_('\n\tClose on book_manager()', fileout)
                ws.close()
        ws.close()
        return

    ### wss functions
    def on_message(ws, message):
        '''
        Control the message received from
        '''
        mess = json.loads(message)
        if mess['e'] == 'kline':
            kln = mess['k']     # update speed: 20000ms; see https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md#subscribe-to-a-stream
            symbol = kln['s'].upper()
            LastObs[symbol] = { '_t': int(kln['t']), '_o': float(kln['o']), '_h': float(kln['h']), '_l': float(kln['l']), '_c': float(kln['c']), '_v': float(kln['q']), "_L": int(kln['L']) }
            if kln['x'] is True:
                new_kln = LastObs[symbol].copy()
                SymKlns[symbol].append(new_kln)
                # print_( '%d. %s\t' % (len(SymKlns[symbol]), symbol) + timestr(new_kln['_t']) + '\t' + \
                        # ''.join(['{:>3}:{:<10}'.format(k, v) for k,v in iter(new_kln.items()) if not k=='_t']), fileout)
            else:
                pass

    def on_error(ws, error):
        '''
        Do something when websocket has an error
        '''
        print_(error, fileout)
        ws.close()
        return

    def on_close(ws):
        '''
        Do something when websocket closes
        '''
        endFlag.append(1)
        for t in [t1, t2, t3]: t.join()
        return

    def on_open(ws, *args):
        '''
        Main function to run multi-threading
        '''
        t1.start()
        t2.start()
        t3.start()
        return

    def position_count(insIds, signal_list, side='BOTH'):
        '''
        Returns number of open positions
        '''
        count = 0
        for s in insIds:
            for sig in signal_list[s]:
                if sig.side==side or side=='BOTH':
                    if sig.is_ordered() or sig.is_active() or sig.is_cnt_ordered():
                        count += 1
        return count

    def in_active_position_(signal_list, side='BOTH'):
        '''
        Check if there is any open positions
        '''
        in_pos = False
        for sig in signal_list:
            if sig.side==side or side=='BOTH':
                if sig.is_active():
                    in_pos = True
                    break
        return in_pos

    def in_possition_2(signal_list, side='BOTH'):
        '''
        Check if there is any open positions
        '''
        in_pos = False
        for sig in signal_list:
            if sig.side==side or side=='BOTH':
                if sig.is_waiting() or sig.is_ordered() or sig.is_cnt_ordered():
                    in_pos = True
                    break
        return in_pos

    def in_possition_(signal_list, side='BOTH'):
        '''
        Check if there is any open positions
        '''
        in_pos = False
        for sig in signal_list:
            if sig.side==side or side=='BOTH':
                if sig.is_waiting() or sig.is_ordered() or sig.is_active() or sig.is_cnt_ordered():
                    in_pos = True
                    break
        return in_pos

    def get_possible_price(mk_data, side):
        '''
        Return a safe limit price available on the market
        '''
        mk_depth = mk_data.order_book(limit=5)
        bids = list(float(x[0]) for x in mk_depth['bids'])
        asks = list(float(x[0]) for x in mk_depth['asks'])
        try:
            lim = (side=='BUY')*bids[0] + (side=='SELL')*asks[0]   #
            lim = round(lim, PRICEPRE[mk_data.symbol.upper()])
        except:
            lim = None
        return bids, asks, lim

    start_time = time.time()
    portfolio, client, testnet, stream, models, fileout = args
    insIds = portfolio.tradeIns
    wd = 14
    atrnum = 1.5
    min_in_candle = 3
    SymKlns = {}
    LastObs = {}
    Signals = {}

    for symbol in insIds:
        SymKlns[symbol] = []
        Signals[symbol] = []
        LastObs[symbol] = []

    endFlag = []
    t1 = threading.Thread( target=data_stream )
    t2 = threading.Thread( target=strategy )
    t3 = threading.Thread( target=book_manager )
    listen_key = client.get_listen_key()
    ws = websocket.WebSocketApp(f'{client.wss_way}{listen_key}',
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
    client.close_stream()

    print_('\n' + barstr('Close Opening Positions', length=100, space_size=5) + '\n', fileout)

    ### to close all positions here ###
    in_position = False
    for symbol in insIds:
        if in_possition_(Signals[symbol]):
            in_position = True
    while in_position:
        for symbol in insIds:
            model = models[symbol]
            for sig in Signals[symbol]:
                if sig.is_waiting():
                    sig.set_expired()
                    print_('\n\tSet WAITING signal EXPIRED: \n\t' + str(sig), fileout)
                elif sig.is_ordered():
                    client.cancel_order(symbol, sig.orderId)
                    sig.set_expired()
                    order_update = client.query_order(symbol, sig.orderId)
                    print_('\n\tSet BOOKED order EXPIRED: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)
                elif sig.is_active():
                    cnt_order = sig.counter_order()
                    lim = round(sig.excPrice*(1 + SIDE[sig.side]*0.1/100), PRICEPRE[symbol])
                    order = client.new_order(symbol=symbol, side=cnt_order['side'], orderType='LIMIT', quantity=cnt_order['amt'], positionSide=sig.positionSide, timeInForce=cnt_order['TIF'], price=lim)
                    sig.set_cnt_ordered(cntorderId=order['orderId'], cntType='LIMIT', cntTime=order['updateTime'], cntlimitPrice=lim)
                    print_('\tPlaced COUNTER order: \n\t' + str(sig) + '\n\t' + orderstr(order), fileout)
                elif sig.is_cnt_ordered():
                    order_update = client.query_order(symbol, sig.cntorderId)
                    sig.set_closed(clsTime=order_update['updateTime'], clsPrice=None)
                    print_('\n\tClosed order: \n\t' + str(sig) + '\n\t' + orderstr(order_update), fileout)
        _position = False
        for symbol in insIds:
            if in_possition_(Signals[symbol]):
                _position = True
                break
        in_position = _position

    return Signals
