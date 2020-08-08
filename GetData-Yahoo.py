"""
Created on 5 Jul 2020

@author: huailin

"""

#import requests,json, csv, os
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.simplefilter("ignore")

import time
import numpy as np
from tqdm import tqdm
from binance.client import Client
from ultilities import barstr, timestr



interval = '5m'
filepath = '/Users/kiki/Snap-Notebook/Keltner Channel/Backtester_Stock/ASX/'
#symbol_list = ['COH.AX','SHL.AX','GPT.AX','CQE.AX','HVN.AX','SUL.AX','ANN.AX']#["ALX.AX",'ANZ.AX','BAP.AX','QUB.AX','GPT.AX','CBA.AX','SDF.AX','SUN.AX','PMV.AX']
#symbol_list = ["ALX.AX"]

data = pd.read_csv('/Users/kiki/Snap-Notebook/Keltner Channel/Backtester_Stock/asx300.csv')
df = data.drop(data.index[0]).reset_index(drop=True)
df = df.rename(columns={'S&P/ASX 300 Index (1 May 2020)':'code'})

ASX300_List = []
for i in df['code']:
    sym = '{}.AX'.format(i)
    ASX300_List.append(sym)

start_time = time.time()
num = 237
for symbol in ASX300_List[238:]:
    s = yf.Ticker(symbol)
    kln_df = s.history(period='60d',interval=interval)   # 5min data request must within last 60days; 1min---last 7 days
    kln_df = kln_df.drop(['Dividends','Stock Splits'],axis=1).reset_index()
    kln_df['_t'] = 0
    for i in range(len(kln_df)):
        kln_df.loc[i, '_t'] = int(pd.Timestamp(datetime.strptime(str(kln_df.loc[i, 'Datetime']), '%Y-%m-%d %H:%M:%S%z')).value/10**6)
    #kln_df = kln_df.set_index('_t')
    kln_df = kln_df.rename(columns ={'Open':'_o', 'High':'_h','Low':'_l','Close':'_c','Volume':'_v'}).drop(['Datetime'], axis=1)

    #print(kln_df)
    klnfile = filepath + '-%s-%s.csv' % (symbol, interval)
    kln_df.to_csv(klnfile)
    print(symbol)
    num = num + 1
    print(num)

print('\n' + barstr(text='Elapsed time = {} seconds'.format(round(time.time()-start_time,2))))
print(barstr(text="", space_size=0))
