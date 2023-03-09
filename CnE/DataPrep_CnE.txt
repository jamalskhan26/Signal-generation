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
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Data'
###############################

#############################################################
#1. Import data: FX:BofAML FX Risk Premia Strategies 
#                ASSETS: Equity and Fixed Income Indices
#############################################################

########UPDATE PATH############
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
###############################

start='20050715'
end  =dt.date.today().strftime('%Y%m%d')

    ### ML fxmonopair tickers
fname=os.path.join(data_dir, 'fx_monopairs_tickers.pickle')
fx_monopairs_tickers = pickle.load( open( fname, "rb" ) )
fx_monopairs = bbg.bdh(list(fx_monopairs_tickers.ticker),'px_last',start,end)
fns.clean_tickers(fx_monopairs,fx_monopairs_tickers,'ticker','name')
fx_monopairs=fx_monopairs.dropna(thresh=2)
fx_monopairs=fns.create_date(fx_monopairs)

    ### BCOM tickers
fname = os.path.join(data_dir, 'BCOM_tickers.xlsx')
bcom_tickers = pd.read_excel(io=fname,skiprows=range(0))
bcom_daily = bbg.bdh(list(bcom_tickers.ticker),'px_last',start,end)
fns.clean_tickers(bcom_daily,bcom_tickers,'ticker','name')
bcom_daily=bcom_daily.dropna(thresh=2)
bcom_daily=fns.create_date(bcom_daily)


#############################################################
#Save datasets 
#############################################################

##time series
with open(os.path.join(out_dir, 'fx_monopairs.pickle'), 'wb') as output:
    pickle.dump(fx_monopairs, output)
with open(os.path.join(out_dir, 'bcom_daily.pickle'), 'wb') as output:
    pickle.dump(bcom_daily, output)

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'assets.xlsx'),date_format = 'mm-dd-yyyy',datetime_format = 'mm-dd-yyyy')
assets.to_excel(writer,'fxrphedge_assets',index=False)
writer.save()
'''