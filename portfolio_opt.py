# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:27:45 2020

@author: jamal
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
import datetime as dt
from datetime import timedelta as td
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import scipy
from scipy import optimize as sco


########UPDATE PATH############
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Data'
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Output'
###############################

#############################################################
#1. Load Data
#############################################################
from eqd import bbg
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import DataPrepFunctions as fns

    # BCOM LS index
fname=os.path.join(data_dir, 'BCOM_LS.pickle')
bcom_ls = pickle.load( open( fname, "rb" ) )
    # asset tickers
fname = os.path.join(data_dir, 'asset_tickers.xlsx')
asset_tickers = pd.read_excel(io=fname,skiprows=range(0))

start='20060801'
end  =dt.date.today().strftime('%Y%m%d')

dat = bbg.bdh(list(asset_tickers.ticker),'px_last',start,end)
fns.clean_tickers(dat,asset_tickers,'ticker','asset')
dat = fns.create_date(dat)

    # weekly returns
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import HelperFunctions as fns
assets = pd.merge(bcom_ls,dat.fillna(method='ffill'),on='Date',how='left')
asset_ret = fns.ret(assets)
asset_ret = asset_ret.dropna()


#############################################################
#2. Analysis
#############################################################

os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import portfolioOptimizationFunctions as po
# Set PARAMS
num_portfolios = 50000
risk_free_rate = 0.0137
_freq = 52

tickers = ['nasdaq','russell','SPX', 'BCOM']

table = asset_ret[['Date']+tickers]
table.index = table.Date
table.drop(['Date'],axis=1,inplace=True)

mean_returns = table.mean()
cov_matrix = table.cov()

bcom_baseline = po.display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, _freq)

tickers = tickers = ['nasdaq','russell','SPX', 'BCOM_LS']

table = asset_ret[['Date']+tickers]
table.index = table.Date
table.drop(['Date'],axis=1,inplace=True)

mean_returns = table.mean()
cov_matrix = table.cov()

bcom_ls = po.display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, _freq)


################################################################################################################################################################################################

writer=pd.ExcelWriter(os.path.join(data_dir,'portfolio_opt.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'portfolio_opt.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'portfolio_opt.xlsx'),engine='openpyxl')
writer.book=book
bcom_baseline.to_excel(writer,'bcom_baseline')
bcom_ls.to_excel(writer,'bcom_ls')
(asset_ret.mean() * 52).to_excel(writer,'mean')
(asset_ret.std()* (52**0.5)).to_excel(writer,'vol')
writer.save()

