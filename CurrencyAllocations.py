# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:52:12 2019

@author: ZK463GK
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
import datetime as dt
from datetime import timedelta
import seaborn

########UPDATE PATH############
data_dir = '//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies/Data'
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
###########################

#############################################################
#1. Load Data
#############################################################

fname=os.path.join(out_dir, 'equity_fx_indices.pickle')
assets = pickle.load( open( fname, "rb" ) )

    ###Import Currency Weights in FX Factors
fname=os.path.join(data_dir, 'wt_ppp_dm.pickle')
wt_ppp_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'wt_ppp_em.pickle')
wt_ppp_em = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'wt_carry_dm.pickle')
wt_carry_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'wt_carry_em.pickle')
wt_carry_em = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'wt_dollar_dm.pickle')
wt_dollar_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'wt_dollar_em.pickle')
wt_dollar_em = pickle.load( open( fname, "rb" ) )

    ###Import FX Factors Weights in Baskets
fname=os.path.join(data_dir, 'wt_ivol.pickle')
wt_ivol = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'wt_reg.pickle')
wt_reg = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'fxrp_reg_hedge_b_m_fac.pickle')
wt_reg_as = pickle.load( open( fname, "rb" ) )


dateseries=pd.DataFrame(assets.Date)
dateseries['month']=dateseries['Date'].dt.month
    #signal on last day of month
dateseries['temp']=(dateseries.month)-(dateseries.month).shift(-1)
dateseries['signalDay']=0
dateseries.loc[dateseries['temp']!=0,'signalDay']=1
    #trade on first day of month
dateseries['temp']=(dateseries.month)-(dateseries.month).shift(1)
dateseries['tradeDay']=0
dateseries.loc[dateseries['temp']!=0,'tradeDay']=1

dateseries=dateseries[['Date','signalDay','tradeDay']]

dm=list(wt_ppp_dm.columns.drop('tradeDate'))
em=list(wt_ppp_em.columns.drop('tradeDate'))

#############################################################
#2. Compute currency weights in baskets
#############################################################

def currency_wt(basket,factor,nick):
    basket['merge']=(basket['Date'].dropna().dt.month.astype(int).astype(str))+(basket['Date'].dropna().dt.year.astype(int).astype(str))
    factor['merge']=(factor['tradeDate'].dropna().dt.month.astype(int).astype(str))+(factor['tradeDate'].dropna().dt.year.astype(int).astype(str))
    
    wt=pd.merge(basket[['merge',nick]],factor.drop(['tradeDate'],axis=1),on='merge',how='inner')
    wt=pd.concat([wt['merge'],wt.iloc[:,2:].mul(wt.iloc[:,1].values,axis=0)],axis=1)
    
    return wt

def extract_signal(dat):
    out=pd.merge(dateseries[dateseries['signalDay']==1].drop(['signalDay','tradeDay'],axis=1),dat,on='Date').rename(columns={'Date':'tradeDate'})
    return out

###iVol
    #Value
wt_ivol_val_dm=currency_wt(wt_ivol,wt_ppp_dm,'Value_DM')
wt_ivol_val_em=currency_wt(wt_ivol,wt_ppp_em,'Value_EM')
    #Carry
wt_ivol_car_dm=currency_wt(wt_ivol,extract_signal(wt_carry_dm),'Carry_DM')
wt_ivol_car_em=currency_wt(wt_ivol,extract_signal(wt_carry_em),'Carry_EM')
    #Dollar
wt_ivol_dol_dm=currency_wt(wt_ivol,extract_signal(wt_dollar_dm),'DollarCarry_DM')
wt_ivol_dol_em=currency_wt(wt_ivol,extract_signal(wt_dollar_em),'DollarCarry_EM')

temp=pd.merge(pd.merge(wt_ivol_val_dm,wt_ivol_car_dm,on='merge',how='outer',suffixes=('','_x')),wt_ivol_dol_dm,on='merge',how='outer')
wt_ivol_dm=temp[dm].fillna(0)+temp[[c+'_x' for c in dm]].fillna(0).values
wt_ivol_dm=wt_ivol_dm.add(temp['Dollar'].fillna(0).values,axis=0)

temp=pd.merge(pd.merge(wt_ivol_val_em,wt_ivol_car_em,on='merge',how='outer',suffixes=('','_x')),wt_ivol_dol_em,on='merge',how='outer')
wt_ivol_em=temp[em].fillna(0)+temp[[c+'_x' for c in em]].fillna(0).values
wt_ivol_em=wt_ivol_em.add(temp['Dollar'].fillna(0).values,axis=0)

wt_ivol_all=pd.concat([temp['merge'],wt_ivol_dm,wt_ivol_em],axis=1)

###Reg
    #Value
wt_reg_val_dm=currency_wt(wt_reg,wt_ppp_dm,'Value_DM')
wt_reg_val_em=currency_wt(wt_reg,wt_ppp_em,'Value_EM')
    #Carry
wt_reg_car_dm=currency_wt(wt_reg,extract_signal(wt_carry_dm),'Carry_DM')
wt_reg_car_em=currency_wt(wt_reg,extract_signal(wt_carry_em),'Carry_EM')
    #Dollar
wt_reg_dol_dm=currency_wt(wt_reg,extract_signal(wt_dollar_dm),'DollarCarry_DM')
wt_reg_dol_em=currency_wt(wt_reg,extract_signal(wt_dollar_em),'DollarCarry_EM')

temp=pd.merge(pd.merge(wt_reg_val_dm,wt_reg_car_dm,on='merge',how='outer',suffixes=('','_x')),wt_reg_dol_dm,on='merge',how='outer')
wt_reg_dm=temp[dm].fillna(0)+temp[[c+'_x' for c in dm]].fillna(0).values
wt_reg_dm=wt_reg_dm.add(temp['Dollar'].fillna(0).values,axis=0)

temp=pd.merge(pd.merge(wt_reg_val_em,wt_reg_car_em,on='merge',how='outer',suffixes=('','_x')),wt_reg_dol_em,on='merge',how='outer')
wt_reg_em=temp[em].fillna(0)+temp[[c+'_x' for c in em]].fillna(0).values
wt_reg_em=wt_reg_em.add(temp['Dollar'].fillna(0).values,axis=0)

wt_reg_all=pd.concat([temp['merge'],wt_reg_dm,wt_reg_em],axis=1)

##transaction costs
def t_costs(dat):
    test=dat
    test=test.diff(1)
    test.columns
    rebal_cost=[0.5]*9+[3.5,3,5,5,3.5,2,2,3,3.5,1,2,3,3.5]
    roll_cost=[6]*9+[42,36,60,60,42,24,24,36,42,12,24,36,42]
    
    rebal=(abs(test.replace(0,np.nan).dropna().reset_index(drop=True)))*rebal_cost/1E4
    rebal=rebal.rolling(12).sum()
    rebal=rebal.mean()
    
    roll=(abs(dat.replace(0,np.nan).dropna().reset_index(drop=True)).rolling(12).mean())*roll_cost/1E4
    roll=roll.mean()
    
    total=rebal+roll
    
    print(total.sum())

t_costs(wt_reg_all.drop(['merge'],axis=1))
t_costs(wt_ivol_all.drop(['merge'],axis=1))

###SUMMARY
    ###last 10 years
def summary_wt(lastn):    
    reg=wt_reg_all.tail(lastn).drop(['merge'],axis=1)
    reg['USD (Net)']=-1*reg.sum(1)
    reg=reg.mean()

    ivol=wt_ivol_all.tail(lastn).drop(['merge'],axis=1)
    ivol['USD (Net)']=-1*ivol.sum(1)
    ivol=ivol.mean()
    
    out=pd.concat([reg,ivol],axis=1)
    out.columns=['FXRP_$','FXRP_iVol']
    temp=out[out.index!='USD (Net)'].sort_values(by='FXRP_$')
    temp=out[out.index=='USD (Net)'].append(temp)
    return temp

full=summary_wt(1000)
last5=summary_wt(60)

writer=pd.ExcelWriter(os.path.join(data_dir,'currency_allocations.xlsx'))
full.to_excel(writer,'full')
last5.to_excel(writer,'last5')
writer.save()

#############################################################
#3. Compute currency weights in asset specific hedges
#############################################################

asset_wts_dict={}

def asset_merge(factor,nick):
    temp=wt_reg_as[p].drop(['const'],axis=1)
    temp['merge']=(temp['Date'].dropna().dt.month.astype(int).astype(str))+(temp['Date'].dropna().dt.year.astype(int).astype(str))
    factor['merge']=(factor['tradeDate'].dropna().dt.month.astype(int).astype(str))+(factor['tradeDate'].dropna().dt.year.astype(int).astype(str))
    wt=pd.merge(temp[['merge',nick]],factor.drop(['tradeDate'],axis=1),on='merge',how='inner')
    return wt

def asset_wts(dat,nick,fx):
    out=dat[fx].mul(dat[nick].values,axis=0)
    out=pd.concat([dat[['merge']],out],axis=1)
    return out

for p in wt_reg_as.keys():
    as_ppp_dm=asset_wts(asset_merge(wt_ppp_dm,'Value_DM'),'Value_DM',dm)
    as_car_dm=asset_wts(asset_merge(extract_signal(wt_carry_dm),'Carry_DM'),'Carry_DM',dm)
    as_dol_dm=asset_wts(asset_merge(extract_signal(wt_dollar_dm),'DollarCarry_DM'),'DollarCarry_DM',['Dollar'])
    
    temp=pd.merge(pd.merge(as_ppp_dm,as_car_dm,on='merge',how='outer',suffixes=('','_x')),as_dol_dm,on='merge',how='outer')
    as_dm=temp[dm].fillna(0)+temp[[c+'_x' for c in dm]].fillna(0).values
    as_dm=as_dm.add(temp['Dollar'].fillna(0).values,axis=0)
    
    as_ppp_em=asset_wts(asset_merge(wt_ppp_em,'Value_EM'),'Value_EM',em)
    as_car_em=asset_wts(asset_merge(extract_signal(wt_carry_em),'Carry_EM'),'Carry_EM',em)
    as_dol_em=asset_wts(asset_merge(extract_signal(wt_dollar_em),'DollarCarry_EM'),'DollarCarry_EM',['Dollar'])
    
    temp=pd.merge(pd.merge(as_ppp_em,as_car_em,on='merge',how='outer',suffixes=('','_x')),as_dol_em,on='merge',how='outer')
    as_em=temp[em].fillna(0)+temp[[c+'_x' for c in em]].fillna(0).values
    as_em=as_em.add(temp['Dollar'].fillna(0).values,axis=0)
    
    as_all=pd.concat([as_dm,as_em],axis=1)
    as_all['USD (Net)']=-1*as_all.sum(1)
    
    ###last 5y
    last5=as_all.tail(60).mean()
    ###last 10y
    last10=as_all.tail(120).mean()
    ###full
    full=as_all.tail(1000).mean()
    
    summary=-1*pd.concat([last5,last10,full],axis=1)
    summary.columns=['last5y','last10y','full']
    
    asset_wts_dict[p]=summary

writer=pd.ExcelWriter(os.path.join(data_dir,'currency_allocations_byasset.xlsx'))
writer.save()
for p in wt_reg_as.keys():
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir,'currency_allocations_byasset.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir,'currency_allocations_byasset.xlsx'),engine='openpyxl')
    writer.book=book
    asset_wts_dict[p].to_excel(writer,(p))
    writer.save()


temp=wt_dollar_dm
temp['year']=temp['Date'].dt.year
temp.groupby(by='year').mean()

temp=wt_dollar_em
temp['year']=temp['Date'].dt.year
temp.groupby(by='year').mean()


wt_ivol.drop(['Date'],axis=1).tail(120).median()
wt_reg.drop(['Date'],axis=1).tail(120).median()

wt_ivol.tail(1).T