# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 2019

@author: jamal
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import os
import pickle
import datetime as dt
from eqd import bbg

########UPDATE PATH############
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Commodity gamma/Data'
data_dir2 = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
###############################

'''
!!UPDATES 9.20.19!! 
1) pull data using bbg api 
2) adjust gasoline OI series to HU1 pre 10/31/05 and XB1 afterwards
3) scale betas to interpret as $ flow instead of # of contracts 
'''


#############################################################
#1. Import data: CFTC Futures and F&O positions, 
#                Front Month Commodity Futures Prices,
#                Aggregate Open Interest
#############################################################


start='19900101'
end  =dt.date.today().strftime('%Y%m%d')


    ###create a list of column names, replacing tickers with commodity name
def clean_tickers(dat,tickerdat,label):
    colnames=dat.columns.tolist()
    colnames=pd.DataFrame(colnames)
    colnames.columns=[label]
    colnames=pd.merge(colnames,tickerdat,on=label,how='outer')
    colnames[colnames[label].str.contains('^Unnamed:')]='Date'
    dat.columns=colnames.commodity
    dat.columns.name=""

    ###import CFTC tickers#
fname = os.path.join(data_dir, 'Tickers_Net Total.Disagg Futures Only.xlsx')
FonlyTickers=pd.read_excel(io=fname,sheet_name='tickerToCom',skiprows=range(0))

fname = os.path.join(data_dir, 'Tickers_Net Total.Disagg Combined.xlsx')
FandOTickers=pd.read_excel(io=fname,sheet_name='tickerToCom',skiprows=range(0))

    ###import futures only positions#
fonly = bbg.bdh(list(FonlyTickers.ticker),'px_last',start,end)
clean_tickers(fonly,FonlyTickers,'ticker')

    ###import futures & options positions#
fando = bbg.bdh(list(FandOTickers.ticker),'px_last',start,end)
clean_tickers(fando,FandOTickers,'ticker')

    ###import Futures tickers
fname = os.path.join(data_dir2, 'commodity_tickers.csv')
com_ticker_map = pd.read_csv(fname)
com_fut_tickers = com_ticker_map.iloc[:,[0,2,3]]

fname=os.path.join(data_dir2, 'com_fut_tickers.pickle')
com_fut_tickers = pickle.load( open( fname, "rb" ) )

    ###import commodity futures price daily data
com_fut_daily = bbg.bdh(list(com_fut_tickers.future),'PX_LAST',start,end)
clean_tickers(com_fut_daily,com_fut_tickers,'future')
    ###forward fill daily data to get non-missing end of week obs
com_fut_daily = com_fut_daily.fillna(method='ffill')    

    ###import futures open interest daily data
com_fut_oi_daily=bbg.bdh(list(com_fut_tickers.future),'FUT_AGGTE_OPEN_INT',start,end)
clean_tickers(com_fut_oi_daily,com_fut_tickers,'future')
    ###forward fill daily data to get non-missing end of week obs
com_fut_oi_daily=com_fut_oi_daily.fillna(method='ffill')    

    ### adjust gasoline series to HU1 pre 10/31/05 and XB1 afterwards
    
gasoline_old_px = bbg.bdh(['HU1 Comdty'],'px_last',start,end)
gasoline_old_oi = bbg.bdh(['HU1 Comdty'],'FUT_AGGTE_OPEN_INT',start,end)
    
com_fut_daily['Gasoline'].loc[com_fut_daily.index<dt.datetime(2005,11,1)]      = gasoline_old_px['HU1 Comdty'].loc[gasoline_old_px.index<dt.datetime(2005,11,1)]
com_fut_oi_daily['Gasoline'].loc[com_fut_oi_daily.index<dt.datetime(2005,11,1)]= gasoline_old_oi['HU1 Comdty'].loc[gasoline_old_oi.index<dt.datetime(2005,11,1)]

###create date column
def create_date(dat):
    dat=dat.reset_index()
    dat['Date']=pd.to_datetime(dat['Date'])
    return dat

com_fut_daily=create_date(com_fut_daily)
com_fut_oi_daily=create_date(com_fut_oi_daily)
fonly=create_date(fonly)
fando=create_date(fando)
    
#############################################################
#2. Save datasets 
#############################################################


##time series
with open(os.path.join(data_dir2, 'cftc_futures.pickle'), 'wb') as output:
    pickle.dump(fonly, output)
with open(os.path.join(data_dir2, 'cftc_FandO.pickle'), 'wb') as output:
    pickle.dump(fando, output)
with open(os.path.join(data_dir2, 'com_fut.pickle'), 'wb') as output:
    pickle.dump(com_fut_daily, output)
with open(os.path.join(data_dir2, 'com_fut_oi.pickle'), 'wb') as output:
    pickle.dump(com_fut_oi_daily, output)

##tickers
with open(os.path.join(data_dir2, 'cftc_tickers_f.pickle'), 'wb') as output:
    pickle.dump(FonlyTickers, output)
with open(os.path.join(data_dir2, 'cftc_tickers_fando.pickle'), 'wb') as output:
    pickle.dump(FandOTickers, output)
with open(os.path.join(data_dir2, 'com_ticker_map.pickle'), 'wb') as output:
    pickle.dump(com_ticker_map, output)
with open(os.path.join(data_dir2, 'com_fut_tickers.pickle'), 'wb') as output:
    pickle.dump(com_fut_tickers, output)

