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

###import helper functions
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import DataPrepFunctions as fns

########OUTPUT PATH############
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
###############################

#############################################################
#1. Import data: FX:BofAML FX Risk Premia Strategies 
#                ASSETS: Equity and Fixed Income Indices
#############################################################

########UPDATE PATH############
data_dir = '//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies\Data'
###############################

start='19900101'
end  =dt.date.today().strftime('%Y%m%d')
fname = os.path.join(data_dir, 'tickers.xlsx')

    ###import FX Factor Strategy tickers#
fx_factors_tickers = pd.read_excel(io=fname,sheet_name='FX_factors',skiprows=range(0))
    ###import Equity / Fixed Income Index tickers#
asset_tickers = pd.read_excel(io=fname,sheet_name='Assets',skiprows=range(0))
    ###Import bloomberg fx_monopairs tickers
fxnames=pd.read_excel(io=fname,sheet_name='FX_mono',skiprows=range(0))
    ###Import currency spot tickers
fx_spot_tickers=pd.read_excel(io=fname,sheet_name='FX_spot',skiprows=range(0))
    ###Import fx futures tickers
fx_fut_tickers=pd.read_excel(io=fname,sheet_name='FX_fut',skiprows=range(0))
   ###Import fx forward rates tickers
fx_fwd_tickers=pd.read_excel(io=fname,sheet_name='FX_fwdrate',skiprows=range(0))


    ###import FX Factor Strategy data#
fx_factors= bbg.bdh(list(fx_factors_tickers.ticker),'px_last',start,end)
fns.clean_tickers(fx_factors,fx_factors_tickers,'ticker','strategy')
fx_factors=fx_factors.dropna(thresh=2)
fx_factors=fns.create_date(fx_factors)

    ###import Equity / Fixed Income Index #
assets = bbg.bdh(list(asset_tickers.ticker),'px_last',start,end)
fns.clean_tickers(assets,asset_tickers,'ticker','asset')
assets=assets.dropna(thresh=2)
assets=fns.create_date(assets)

    ###Import Bloomberg FX_monopairs 
fx_monopairs = bbg.bdh(list(fxnames.ticker),'px_last',start,end)
fns.clean_tickers(fx_monopairs,fxnames,'ticker','name')
fx_monopairs = fx_monopairs.dropna(thresh=2)
fx_monopairs =fns.create_date(fx_monopairs)

    ###Import FX spot rates
fx_spot = bbg.bdh(list(fx_spot_tickers.ticker),'px_last',start,end)
fns.clean_tickers(fx_spot,fx_spot_tickers,'ticker','name')
fx_spot = fx_spot.dropna(thresh=2)
fx_spot = fns.create_date(fx_spot)


    ###Import FX forward rates outright (after changing default settings in BB XDF<GO>)
fx_fwd= bbg.bdh(list(fx_fwd_tickers.ticker),'px_last',start,end)
fns.clean_tickers(fx_fwd,fx_fwd_tickers,'ticker','name')
fx_fwd = fx_fwd.dropna(thresh=2)
fx_fwd = fns.create_date(fx_fwd)


    ###Import FX futures front month active contracts
fx_fut= bbg.bdh(list(fx_fut_tickers.ticker),'px_last',start,end)
fns.clean_tickers(fx_fut,fx_fut_tickers,'ticker','name')
fx_fut = fx_fut.dropna(thresh=2)
fx_fut = fns.create_date(fx_fut)

#########################################
    ### make spot rates consistent
    ### Price of 1 USD in FOR
#########################################
inverse=['Australia','New Zealand','United Kingdom','European Union']
fx_spot[inverse]=1/fx_spot[inverse]
fx_spot.iloc[-2,:]

#########################################
    ### make forward rates consistent
    ### Price of 1 USD in FOR
#########################################
inverse=['Australia','New Zealand','United Kingdom','European Union']
fx_fwd[inverse]=1/fx_fwd[inverse]
fx_fwd.iloc[-2,:]

#########################################
    ### Adjust forward prices 
    ### by contract size and quote currency
    ### Price of 1 USD in FOR
#########################################

for name in list(fx_fut_tickers.name):
    fx_fut[name]=fx_fut[name]**float(fx_fut_tickers[fx_fut_tickers['name']==name].reciprocal)
    fx_fut[name]=fx_fut[name]*float(fx_fut_tickers[fx_fut_tickers['name']==name].scale)
    
    ###import Equity / Fixed Income Index tickers#
dxy = bbg.bdh('DXY Curncy','px_last',start,end)
dxy = fns.create_date(dxy)

#############################################################
#Save datasets 
#############################################################

##time series
with open(os.path.join(out_dir, 'fx_factors_ml.pickle'), 'wb') as output:
    pickle.dump(fx_factors, output)
with open(os.path.join(out_dir, 'equity_fx_indices.pickle'), 'wb') as output:
    pickle.dump(assets, output)
with open(os.path.join(out_dir, 'dxy.pickle'), 'wb') as output:
    pickle.dump(dxy, output)
with open(os.path.join(out_dir, 'fxforward_index_bb.pickle'), 'wb') as output:
    pickle.dump(fx_monopairs, output)
with open(os.path.join(out_dir, 'fx_spot.pickle'), 'wb') as output:
    pickle.dump(fx_spot, output)
with open(os.path.join(out_dir, 'fx_fwd.pickle'), 'wb') as output:
    pickle.dump(fx_fwd, output)
with open(os.path.join(out_dir, 'fx_fut.pickle'), 'wb') as output:
    pickle.dump(fx_fut, output)

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'assets.xlsx'),date_format = 'mm-dd-yyyy',datetime_format = 'mm-dd-yyyy')
assets.to_excel(writer,'fxrphedge_assets',index=False)
writer.save()

writer=pd.ExcelWriter(os.path.join(data_dir,'dxy.xlsx'),date_format = 'mm-dd-yyyy',datetime_format = 'mm-dd-yyyy')
dxy.to_excel(writer,'dxy',index=False)
writer.save()

writer=pd.ExcelWriter(os.path.join(data_dir,'fx_fac.xlsx'),date_format = 'mm-dd-yyyy',datetime_format = 'mm-dd-yyyy')
fx_factors.to_excel(writer,'fx_fac',index=False)
writer.save()
'''