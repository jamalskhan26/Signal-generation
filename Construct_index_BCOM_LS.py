# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:28:26 2020

@author: Jamal Khan
"""

'''
Egg & Chicken 
    construct index
'''
 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 

#############################################################
# Import regression models class
#############################################################
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
wd = os.getcwd()
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
from regression_models import regression_models
os.chdir(wd)

#############################################################
# Load data
#############################################################
import pandas as pd
import numpy as np
import os
import _pickle as pickle
import datetime as dt
### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Data'
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Output'

    ### BCOM ticker weights
fname = os.path.join(data_dir, 'ticker_weights.xlsx')
bcom_tickers = pd.read_excel(io=fname,skiprows=range(0))
bcom_tickers.index=bcom_tickers.name
bcom_tickers = bcom_tickers.drop(['name','ticker'],axis=1)

fname=os.path.join(data_dir, 'encModels_1Ylookback.pickle')
model = pickle.load( open( fname, "rb" ))
model.ensemble()

bcom_weights = pd.DataFrame(index = model.actual_returns.index, columns=model.actual_returns.columns)
bcom_weights.loc[:, :] = bcom_tickers.T[model.actual_returns.columns].values

strategy_long = model.actual_returns*bcom_weights 
strategy_eq = pd.DataFrame() 
        
p = model.predictions[model.models.index('Ensemble')]
temp = pd.concat([p[y][[y]] for y in model.actual_returns.columns], axis = 1)
    # construct index
portfolio = strategy_long.mul(temp.iloc[strategy_long.index].values, axis = 0)    
strategy = portfolio.sum(axis = 1)
strategy = (pd.concat([pd.Series([100]), 1 + strategy])).cumprod().iloc[1:]

bcom_ls = pd.concat([model.dates,strategy],axis=1)
bcom_ls.columns = ['Date','BCOM_LS']


with open(os.path.join(data_dir, 'bcom_ls.pickle'), 'wb') as output:
    pickle.dump(bcom_ls, output)
    

### merge with BCOM
    
from eqd import bbg
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import DataPrepFunctions as fns


start=model.dates.iloc[51].values[0].astype(str)
end  =dt.date.today().strftime('%Y%m%d')

bcom = bbg.bdh(['BCOM Index'],'px_last',start,end)
bcom = fns.create_date(bcom)

bcom.index = pd.DatetimeIndex(bcom.Date)
bcom = bcom.reindex(pd.date_range(start, end), fill_value=np.nan).reset_index()
bcom.drop(['Date'],axis=1,inplace=True)
bcom = bcom.fillna(method='ffill')
bcom.columns = ['Date','BCOM']

    # weekly returns
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import HelperFunctions as fns

bcom = pd.merge(bcom_ls[['Date']],bcom,on='Date',how='left')
bcom  = fns.ret(bcom)
bcom.BCOM = (pd.concat([pd.Series([100]), 1 + bcom.BCOM])).cumprod().iloc[1:]

bcom_ls = pd.merge(bcom_ls,bcom,on='Date',how='outer')

writer=pd.ExcelWriter(os.path.join(data_dir,'BCOM_LS.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'BCOM_LS.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'BCOM_LS.xlsx'),engine='openpyxl')
writer.book=book
bcom_ls.to_excel(writer,'BCOM_longshort')
writer.save()