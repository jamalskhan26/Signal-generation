# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 2019

@author: Jamal
"""

import pandas as pd

    ###create a list of column names, replacing tickers with commodity name
def clean_tickers(dat,tickerdat,label,target):
    '''
    input dataframe, dataframe with list of tickers and 
    corresponding column names, the label of the merge variable
    and target variable name
    '''
    colnames=dat.columns.tolist()
    colnames=pd.DataFrame(colnames)
    colnames.columns=[label]
    colnames=pd.merge(colnames,tickerdat,on=label,how='outer')
    colnames[colnames[label].str.contains('^Unnamed:')]='Date'
    dat.columns=colnames[target]
    dat.columns.name=""
    
    ###create date column
def create_date(dat):
    dat=dat.reset_index()
    dat['Date']=pd.to_datetime(dat['Date'])
    return dat

    ###align dates across each timeseries
def align_dates(df):
    '''Note: df should be structured as [date1,series1,date2,series2,...]
    make sure to name the date columnn in df "Date"
    '''

    ###set parameters#
    start_date=df.Date.iloc[:,0].dropna().iloc[0]
    end_date  =df.Date.iloc[:,0].dropna().iloc[-1]

    ###generate date series between start and end including every calendar day
    base=pd.DataFrame(pd.date_range(start_date,end_date))
    base.columns=['Date']

    ###merge each monopair series with master date series
    for x in range(0,len(df.columns),2):
        temp1=df.iloc[:,x:x+2]
        base=pd.merge(base,temp1,on='Date',how='left')

    ###subset merged dataset to remove rows with no observations
    outdf=base.dropna(how='all',subset=base.columns[1:]).reset_index(drop=True)
    
    return outdf
