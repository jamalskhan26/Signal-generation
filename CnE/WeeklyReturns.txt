# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:52:46 2019

@author: jamal
"""
'''
Prepare weekly returns series for chicken and egg model
Generate signals on Wednesday each week  and perform rebalancing on Thursday
=> model inputs based on Wed-Wed returns series & 
    realized returns based on Thu-Thu returns series
'''

from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import os
import pickle
from datetime import timedelta
###import helper functions
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import HelperFunctions as fns

########UPDATE PATH############
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Data'
###############################


#############################################################
#0. Load Data
#############################################################

    ### fx_monopairs price series
fname=os.path.join(out_dir, 'fx_monopairs.pickle')
fx_monopairs = pickle.load( open( fname, "rb" ) )

    ### bcom price series
fname=os.path.join(out_dir, 'bcom_daily.pickle')
bcom_daily = pickle.load( open( fname, "rb" ) )

#############################################################
#1. Generate Daily Returns Series for Signal Generation and Rebalancing Trades 
#############################################################

#FX
signal_returns_daily_fx = fns.ret(fx_monopairs)
trade_returns_daily_fx =  signal_returns_daily_fx.shift(-2)

signal_returns_daily_fx.rename(columns = {'Date':'signalDate'}, inplace = True)
signal_returns_daily_fx = pd.concat([fx_monopairs[['Date']], signal_returns_daily_fx], axis = 1)
signal_returns_daily_fx.rename(columns = {'Date':'date'}, inplace = True)

trade_returns_daily_fx.rename(columns = {'Date':'tradeDate'}, inplace = True)
trade_returns_daily_fx = pd.concat([fx_monopairs[['Date']], trade_returns_daily_fx], axis = 1)
trade_returns_daily_fx.rename(columns = {'Date':'date'}, inplace = True)

signal_returns_daily_fx = signal_returns_daily_fx.dropna(thresh = 2)

#Commodities
signal_returns_daily_co = fns.ret(bcom_daily)
trade_returns_daily_co =  signal_returns_daily_co.shift(-2)

signal_returns_daily_co.rename(columns = {'Date':'signalDate'}, inplace = True)
signal_returns_daily_co = pd.concat([bcom_daily[['Date']], signal_returns_daily_co], axis = 1)
signal_returns_daily_co.rename(columns = {'Date':'date'}, inplace = True)

trade_returns_daily_co.rename(columns = {'Date':'tradeDate'}, inplace = True)
trade_returns_daily_co = pd.concat([bcom_daily[['Date']], trade_returns_daily_co], axis = 1)
trade_returns_daily_co.rename(columns = {'Date':'date'}, inplace = True)

with open(os.path.join(out_dir, 'signal_returns_daily_fx.pickle'), 'wb') as output:
    pickle.dump(signal_returns_daily_fx.fillna(0), output)
with open(os.path.join(out_dir, 'signal_returns_daily_co.pickle'), 'wb') as output:
    pickle.dump(signal_returns_daily_co.fillna(0), output)
with open(os.path.join(out_dir, 'trade_returns_daily_fx.pickle'), 'wb') as output:
    pickle.dump(trade_returns_daily_fx.fillna(0), output)
with open(os.path.join(out_dir, 'trade_returns_daily_co.pickle'), 'wb') as output:
    pickle.dump(trade_returns_daily_co.fillna(0), output)


#############################################################
#2. Generate Monthly Returns Series for Signal Generation and Rebalancing Trades 
#############################################################
#FX
fx_monopairs_monthly = fx_monopairs.copy()
fx_monopairs_monthly = fx_monopairs_monthly.fillna(method='ffill')
fx_monopairs_monthly['tradeday'] = fx_monopairs_monthly ['Date'].dt.month.diff()
fx_monopairs_monthly['tradeday'][fx_monopairs_monthly ['tradeday'] != 0] = 1 
fx_monopairs_monthly['signalday'] = fx_monopairs_monthly ['tradeday'].shift(-1)

signal_returns_monthly_fx = fx_monopairs_monthly[fx_monopairs_monthly['signalday'] == 1][fx_monopairs.columns].reset_index(drop = True)
signal_returns_monthly_fx = fns.ret(signal_returns_monthly_fx)
signal_returns_monthly_fx.rename(columns = {'Date':'signalDate'}, inplace = True)
signal_returns_monthly_fx = pd.concat([fx_monopairs_monthly[fx_monopairs_monthly['signalday'] == 1]['Date'].reset_index(drop = True), signal_returns_monthly_fx], axis = 1)
signal_returns_monthly_fx.rename(columns = {'Date':'date'}, inplace = True)

trade_returns_monthly_fx = fx_monopairs_monthly[fx_monopairs_monthly['tradeday'] == 1][fx_monopairs.columns].iloc[1:,:].reset_index(drop = True)
trade_returns_monthly_fx = fns.ret_next(trade_returns_monthly_fx)
trade_returns_monthly_fx.rename(columns = {'Date':'tradeDate'}, inplace = True)
trade_returns_monthly_fx = pd.concat([fx_monopairs_monthly[fx_monopairs_monthly['signalday'] == 1]['Date'].reset_index(drop = True), trade_returns_monthly_fx], axis = 1)
trade_returns_monthly_fx.rename(columns = {'Date':'date'}, inplace = True)

#Commodities
bcom_monthly = bcom_daily.copy()
bcom_monthly = bcom_monthly.fillna(method='ffill')
bcom_monthly['tradeday'] = bcom_monthly['Date'].dt.month.diff()
bcom_monthly['tradeday'][bcom_monthly['tradeday'] != 0] = 1 
bcom_monthly['signalday'] = bcom_monthly['tradeday'].shift(-1)

signal_returns_monthly_co = bcom_monthly[bcom_monthly['signalday'] == 1][bcom_daily.columns].reset_index(drop = True)
signal_returns_monthly_co = fns.ret(signal_returns_monthly_co)
signal_returns_monthly_co.rename(columns = {'Date':'signalDate'}, inplace = True)
signal_returns_monthly_co = pd.concat([bcom_monthly[bcom_monthly['signalday'] == 1]['Date'].reset_index(drop = True), signal_returns_monthly_co], axis = 1)
signal_returns_monthly_co.rename(columns = {'Date':'date'}, inplace = True)

trade_returns_monthly_co = bcom_monthly[bcom_monthly['tradeday'] == 1][bcom_daily.columns].iloc[1:,:].reset_index(drop = True)
trade_returns_monthly_co = fns.ret_next(trade_returns_monthly_co)
trade_returns_monthly_co.rename(columns = {'Date':'tradeDate'}, inplace = True)
trade_returns_monthly_co = pd.concat([bcom_monthly[bcom_monthly['signalday'] == 1]['Date'].reset_index(drop = True), trade_returns_monthly_co], axis = 1)
trade_returns_monthly_co.rename(columns = {'Date':'date'}, inplace = True)

with open(os.path.join(out_dir, 'signal_returns_monthly_fx.pickle'), 'wb') as output:
    pickle.dump(signal_returns_monthly_fx.fillna(0), output)
with open(os.path.join(out_dir, 'signal_returns_monthly_co.pickle'), 'wb') as output:
    pickle.dump(signal_returns_monthly_co.fillna(0), output)
with open(os.path.join(out_dir, 'trade_returns_monthly_fx.pickle'), 'wb') as output:
    pickle.dump(trade_returns_monthly_fx.fillna(0), output)
with open(os.path.join(out_dir, 'trade_returns_monthly_co.pickle'), 'wb') as output:
    pickle.dump(trade_returns_monthly_co.fillna(0), output)


#############################################################
#2. Create Date Series for Signal and Trade days
#############################################################

d1 = fx_monopairs['Date'].iloc[0]
d2 = fx_monopairs['Date'].iloc[-1]

def create_datelist(flag,dayofweek):
    '''
    create a list of dates given day of the week as input
    '''
    datelist= pd.DataFrame([d1 + timedelta(days=x) for x in range((d2-d1).days + 1)])
    datelist.columns=[flag]
    datelist['temp']=datelist[flag].dt.dayofweek
    datelist=datelist[datelist['temp']==dayofweek].drop(['temp'],axis=1).reset_index(drop=True) 
    return datelist

    # construct list of signal days -- each Wednesday -- in sample period 
signalDays=create_datelist('signalDate',2)
    # construct list of trade days -- each Thursday -- in sample period 
tradeDays=create_datelist('tradeDate',3)
    # construct identifier weekly date series -- each Monday -- in sample period 
_weekly=create_datelist('date',0)
    # merge date series
dates = pd.concat([pd.concat([_weekly,signalDays],axis=1),tradeDays],axis=1)
    # get identifiers for weeks
dates['id'] = dates.date.apply(lambda x: x.isocalendar()[:2])

#############################################################
#3. Generate Weekly Returns Series for Signal Generation and Rebalancing Trades 
#############################################################

# create price series for all calendar days in sample period
# forward fill missing obs
datelist = pd.DataFrame([d1 + timedelta(days=x) for x in range((d2-d1).days + 1)])
datelist.columns=['Date']

#FX
fx_monopairs=pd.merge(datelist,fx_monopairs,on='Date',how='left')
fx_monopairs=fx_monopairs.fillna(method='ffill')

#Commodities
bcom_daily=pd.merge(datelist,bcom_daily,on='Date',how='left')
bcom_daily=bcom_daily.fillna(method='ffill')

# Calculate weekly returns to generate signals: signal on T based on prices from T-1 and T

def signal_returns(dat,date_series):
    signal_returns=pd.merge(date_series.drop(['tradeDate','id'],axis=1),dat,left_on='signalDate',right_on='Date',how='left')
    signal_returns[signal_returns.isna().any(axis=1)]
    signal_returns=pd.concat([signal_returns[['date','signalDate']],fns.ret(signal_returns.iloc[:,2:])],axis=1).drop(['Date'],axis=1)
    return signal_returns

#FX
signal_returns_fx=signal_returns(fx_monopairs, dates)

#Commodities
signal_returns_co=signal_returns(bcom_daily, dates)


# calculate weekly trade (realized) returns in the subsequent week on long FX position: return on T based on price change from T to T+1
def trade_returns(dat,date_series):
    trade_returns=pd.merge(date_series.drop(['signalDate','id'],axis=1),dat,left_on='tradeDate',right_on='Date',how='left')
    trade_returns[trade_returns.isna().any(axis=1)]
    trade_returns=pd.concat([trade_returns[['date','tradeDate']],fns.ret_next(trade_returns.iloc[:,2:])],axis=1).drop(['Date'],axis=1)
    return trade_returns

#FX
trade_returns_fx=trade_returns(fx_monopairs, dates)

#Commodities
trade_returns_co=trade_returns(bcom_daily, dates)


#############################################################
#Save datasets 
#############################################################

with open(os.path.join(out_dir, 'signal_returns_fx.pickle'), 'wb') as output:
    pickle.dump(signal_returns_fx, output)
with open(os.path.join(out_dir, 'signal_returns_co.pickle'), 'wb') as output:
    pickle.dump(signal_returns_co, output)
with open(os.path.join(out_dir, 'trade_returns_fx.pickle'), 'wb') as output:
    pickle.dump(trade_returns_fx, output)
with open(os.path.join(out_dir, 'trade_returns_co.pickle'), 'wb') as output:
    pickle.dump(trade_returns_co, output)

writer=pd.ExcelWriter(os.path.join(out_dir,'Weekly_Returns_C&E.xlsx'))
signal_returns_fx.to_excel(writer,'signal_returns_fx',index=False)
signal_returns_co.to_excel(writer,'signal_returns_co',index=False)
trade_returns_fx.to_excel(writer,'trade_returns_fx',index=False)
trade_returns_co.to_excel(writer,'trade_returns_co',index=False)
writer.save()





