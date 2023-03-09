# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:17:20 2019

@author: ZK463GK
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import os
import pickle
import datetime as dt
from eqd import bbg
from scipy import stats
import matplotlib.pyplot as plt

###import helper functions
os.chdir('//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies/Code/FX_RPproject')
import HelperFunctions as fns

########OUTPUT PATH############
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
###############################
########UPDATE PATH############
data_dir = '//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies/Data'
###############################

'''
Construct FX Factors
1. Value
2. Carry
3. Dollar
4. Momentum
5. Mean Reversion
'''

#############################################################
#0. Import data: FX monopairs
#                PPP weights
#############################################################

    ###import FX monopairs
fname=os.path.join(out_dir, 'fxforward_index_bb.pickle')
fxmonopairs=pickle.load( open( fname, "rb" ) )

    ###import FX spot
fname=os.path.join(out_dir, 'fx_spot.pickle')
fx_spot=pickle.load( open( fname, "rb" ) )

    ###import FX forward contracts
fname=os.path.join(out_dir, 'fx_fwd.pickle')
fx_fwd=pickle.load( open( fname, "rb" ) )

#########################################

    ### calculate mono returns in t+1
fxmonopairs = fxmonopairs.fillna(method='ffill')
fxmono_ret=fns.ret_next(fxmonopairs)

    ### calculate mono returns in t (contemparaneous)
fxmono_ret_contamp=fns.ret(fxmonopairs)

    ###import PPP exchange rates
fname = os.path.join(data_dir, 'PPP_OECD.xlsx')
ppp = pd.read_excel(io=fname,sheet_name='ppp',skiprows=range(0))
ppp = ppp.T.reset_index()
ppp.columns = list(ppp.iloc[0,:])
ppp = ppp.iloc[1:,:].reset_index(drop=True)
ppp['Year']=ppp['Year'].astype(int)


#############################################################
#   Create Date Series for Signal and Trade days
#############################################################

    ### 1-month holding period for strategy
        # generate signal on last trading day of month
        # trade/rebalance on first trading day of month

dateseries=pd.DataFrame(fxmonopairs.Date)
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


#############################################################
#   IGARCH model for portfolio covariance variance estimation
#   used in vol targeting strategies (for EM strats)
#############################################################

#Calculate portfolio variance
from itertools import combinations
 
def port_var(cov,wt,curr):    

#List of covariance combinations
    cov_pairs=list(combinations(curr, 2))

#Portfolio variance component
    var_portfolio=pd.DataFrame(0,index=range(len(wt)),columns=['var'])
    for c in curr:
        temp=pd.DataFrame(cov[c][c]*(wt[c]**2).values)
        temp[np.isnan(temp)]=0
        var_portfolio = var_portfolio.add(temp.values,axis=0)
#Portfolio covariance component (2 times sum of covariances times square of weights)        
    cov_portfolio=pd.DataFrame(0,index=range(len(wt)),columns=['var'])
    for c in cov_pairs:
        temp=pd.DataFrame(2*cov[c[0]][c[1]]*((wt[c[0]]*wt[c[1]]).values))
        temp[np.isnan(temp)]=0
        cov_portfolio = cov_portfolio.add(temp.values,axis=0)
        
    var_portfolio=cov_portfolio.add(var_portfolio.values,axis=0)
    return var_portfolio


#Calculate Covariance Matrix of mono returns using IGARCH


curr=list(fx_spot.columns.drop('Date'))
delta=0.06
r2={}
cov_d={}
cov={}

for c in curr:
    r2[c]=fxmono_ret_contamp[curr].iloc[1:,:].multiply(fxmono_ret_contamp.iloc[1:,:][c].values,axis=0)
    r2[c]=r2[c].fillna(0)

    #  set initial value
for c in curr:
    cov_d[c]=pd.DataFrame(r2[c]).copy()
    cov_d[c]=cov_d[c]*delta
    cov_d[c].iloc[1:,]=0

for c in curr:
    for i in range(1,len(cov_d[c])):
        cov_d[c].iloc[i,:]=(1-delta)*cov_d[c].iloc[i-1,:]+delta*r2[c].iloc[i,:]
        
for c in curr:  
    cov[c]=cov_d[c]
#LAST DAY OF THE MONTH     
    cov[c]['signalDay']=dateseries['signalDay'].iloc[1:]
    cov[c]=21*cov[c].loc[cov[c]['signalDay']==1].drop(['signalDay'],axis=1).reset_index(drop=True)

for c in curr:  
    cov_d[c]=pd.concat([fxmono_ret_contamp['Date'].iloc[1:],cov_d[c]],axis=1)

with open(os.path.join(out_dir, 'cov_bbmono.pickle'), 'wb') as output:
    pickle.dump(cov, output)

with open(os.path.join(out_dir, 'cov_d_bbmono.pickle'), 'wb') as output:
    pickle.dump(cov_d, output)



#############################################################
#1. Value
#############################################################

    #fxmono cov matrix
fname=os.path.join(out_dir, 'cov_bbmono.pickle')
cov=pickle.load( open( fname, "rb" ) )

#### Calculate PPP weights

temp = fx_spot.copy()
temp['Year'] = temp['Date'].dt.year
temp = pd.merge(temp,ppp,on='Year',how='left',suffixes=('','_ppp'))
temp = temp.dropna().reset_index(drop=True)
spots = fx_spot.columns.drop('Date')

##calculate smoothed exchange rate (30 DMA) for ppp scores##
smooth=pd.concat([temp['Date'],temp.iloc[:,1:].rolling(window=30).mean()],axis=1)

### ppp  scores
ppp_fx = smooth[spots].div(temp[spots+'_ppp'].values,axis=1)
ppp_fx = pd.concat([temp['Date'],ppp_fx],axis=1)

##!!Rebalance monthly first day of month!!##

##!!extract scores on signal days!!##
### first merge rolling hedge ratios with date series
signal=pd.merge(dateseries,ppp_fx,on='Date',how='left')
### next forward fill signals to ensure non-missing values on signal days
signal=signal.fillna(method='ffill')
### subset to signal days and create trade date series (t+1)
signal=signal[(signal['signalDay']==1) | (signal['tradeDay']==1)]
signal['tradeDate']=signal['Date'].shift(-1)
signal=signal[(signal['signalDay']==1)]
signal=signal.drop(['signalDay','tradeDay'],axis=1)


### Compute weights for each currency based on PPP scores

def ppp_wt(currencies):
   
    df=signal.drop(['Date','tradeDate'],axis=1)
    ### harmonisation
    geo_mean = df[currencies].prod(axis=1).pow(1/df[currencies].count(axis=1))
    ### calculate harmonised ppp scores
    ppp_wt = df[currencies].div(geo_mean.values,axis=0)
    ### calculate currency weights as distance from fair value
    ppp_wt = ppp_wt-1
    ### scale weights to sum to 100%
#    ppp_wt = ppp_wt.div(abs(ppp_wt).sum(axis=1),axis=0)
#    ppp_wt.sum(axis=1)
    
    ppp_wt = pd.concat([signal['tradeDate'],ppp_wt],axis=1).reset_index(drop=True)
    
    return ppp_wt


    ### DM 
dm=['European Union','Australia','Canada','Japan','New Zealand','Norway','Sweden','Switzerland','United Kingdom']    
ppp_dm=ppp_wt(currencies=dm)
ppp_dm.iloc[:,1:].mean().sort_values()

    ### EM 
em=list(set(ppp_fx.columns.drop('Date'))-set(dm))
ppp_em=ppp_wt(currencies=em)
ppp_em.iloc[:,1:].mean().sort_values()


## Add Vol Control Overlay for EM

#stdev threshold (5% monthly vol)
rr_std=0.05/pow(12,0.5)

## Cap wts in  (-1,1) ?

###remove last obs: cov matrix not updated with Nov 19 data yet
#ppp_em=ppp_em.iloc[:-1,:]

#VALUE EM VOL OVERLAY
wt_em=ppp_em.drop(['tradeDate'],axis=1).reset_index(drop=True)
var_portfolio_em=port_var(cov=cov,wt=wt_em,curr=list(wt_em.columns))
std_portfolio_em=pow(var_portfolio_em,0.5)
leverage_em=pow(std_portfolio_em,-1)*rr_std

####require at least 1 year of covariance estimation to apply leverage (after beginning of period with > half mono series) 
begin_strat_date=fxmono_ret[em+['Date']].dropna(thresh=8).iloc[0,:]['Date']
ppp_em[ppp_em['tradeDate']>=begin_strat_date] # index=33
leverage_em.iloc[0:45]=1
ppp_em_unlev=ppp_em.copy()
ppp_em.iloc[:,1:]=np.asarray(ppp_em.iloc[:,1:])*np.asarray(leverage_em)


#####OUTPUT CURRENCY WEIGHTS IN VALUE
with open(os.path.join(data_dir, 'wt_ppp_dm.pickle'), 'wb') as output:
    pickle.dump(ppp_dm, output)
with open(os.path.join(data_dir, 'wt_ppp_em.pickle'), 'wb') as output:
    pickle.dump(ppp_em, output)


########CONSTRUCT FACTOR AS LONG SHORT STRATEGY##############

def construct_val_factor(value,name):

    ##merge with fx monopairs
    temp=value.copy()
    temp['merge']=value['tradeDate'].apply(lambda x: [x.year,x.month]).astype(str) 
    temp=temp.drop(['tradeDate'],axis=1)
    fxmono_ret['merge']=fxmono_ret['Date'].apply(lambda x: [x.year,x.month]).astype(str) 
    cols = ['Date'] + list(temp.columns)
        ### require at least half (7 out of 13) EM currencies to have mono returns to compute strategy
        ### this occurrs in Oct 2001 -- the rest of the mono returns begin in 2005 (RUB, IDR, ILS) and 2008 (INR, HUF, KRW)
        ### thresh = 7 + 2datecolumns 
    temp=pd.merge(fxmono_ret[cols].dropna(thresh=9),temp,on='merge',how='left',suffixes=('','_w'))              
    fxmono_ret.drop(['merge'],axis=1,inplace=True)
    temp=temp.drop(['merge'],axis=1)
    temp =temp.fillna(method='ffill')
    temp2=temp[list(value.iloc[:,1:].columns)].mul(temp[list(value.iloc[:,1:].columns+'_w')].values,axis=1)
    temp2=temp2.sum(axis=1)
    temp2=temp2+1
    temp2[0]=100
    out=temp2.cumprod()
    out=pd.concat([temp['Date'],out],axis=1)
    out.columns=['Date',name] 
    
    return out

value_dm =construct_val_factor(value=ppp_dm,name='Value_DM')
value_em =construct_val_factor(value=ppp_em,name='Value_EM')
value_em_unlev =construct_val_factor(value=ppp_em_unlev,name='Value_EM')

    #Global Value as DM/EM average
value_glo = pd.merge(value_dm,value_em,on='Date',how='inner')  
value_glo['Value_Global']=(value_glo['Value_DM']+value_glo['Value_EM'])/2
value_glo=value_glo.drop(['Value_DM','Value_EM'],axis=1)

plt.plot(value_dm.Date,value_dm.Value_DM)
plt.plot(value_em.Date,value_em.Value_EM)
plt.plot(value_glo.Date,value_glo.Value_Global)

with open(os.path.join(out_dir, 'fx_value_dm.pickle'), 'wb') as output:
    pickle.dump(value_dm, output)

with open(os.path.join(out_dir, 'fx_value_em.pickle'), 'wb') as output:
    pickle.dump(value_em, output)

with open(os.path.join(out_dir, 'fx_value_gl.pickle'), 'wb') as output:
    pickle.dump(value_glo, output)

### POTENTIAL IMPROVEMENT: Update PPP rates monthly using recent CPI data

#############################################################
#2. Cross-Sectional Carry : !! UPDATE -- REBALANCE DAILY !!
    # TRY LRV 2014 POTFOLIO BASED METHOD - HML FACTOR
#############################################################

    #fxmono cov matrix
fname=os.path.join(out_dir, 'cov_d_bbmono.pickle')
cov_d=pickle.load( open( fname, "rb" ) )


curr=list(fx_spot.columns.drop('Date'))
dm  =['European Union','Australia','Canada','Japan','New Zealand','Norway','Sweden','Switzerland','United Kingdom']    
em  =list(set(curr)-set(dm))

spot_fwd=pd.merge(fx_spot,fx_fwd,on='Date',how='inner',suffixes=('','_f'))
    ### convert to dollars per unit of foreign currency
spot_fwd.iloc[:,1:]=1/spot_fwd.iloc[:,1:]

#calculate forward discount
fwd_discount=spot_fwd[['Date']+curr].copy()
fwd_discount[curr]=spot_fwd[curr].sub(spot_fwd[[f+'_f' for f in curr]].values)

## buy forward t sell spot t+1 returns
carry_ret=spot_fwd[curr].shift(-1).div(spot_fwd[[f+'_f' for f in curr]].values)-1
carry_ret=pd.concat([spot_fwd['Date'],carry_ret],axis=1)

#get directional positions
def _sign(df):
    df_sign=df.copy()
    df_sign[df_sign > 1e-10]=1 #forward discount Ft<St 
    df_sign[df_sign < -1e-10]=-1 #forward premium Ft>St
    df_sign[abs(df_sign) < 1e-10]=0
    return df_sign

fwd_discount_sign=_sign(fwd_discount.iloc[:,1:])

def wt_xs_carry(fx,spread_weighted=True,vol_overlay=False,vol_target=1):
    '''
    compute carry trade weights:
    go long currencies trading with forward discount
    with a weight proportional to the spread (if spread_weighted)
    otherwise equal weighted in the direction of fwd discount
    Optional param: vol_overlay to target portfolio 
    volatility at vol_target annually
    '''
    if spread_weighted==True:
        wt_xs_carry=fwd_discount.copy()     
        
        ##calculate forward discount spread scaled by realized volatility
        realvol=pd.concat([fxmono_ret['Date'],fxmono_ret[fx].fillna(method='ffill').rolling(window=252).std()],axis=1)
        ##calculate spread-weighted currency allocations
        temp=pd.merge(wt_xs_carry,realvol,on='Date',how='inner',suffixes=('','_v'))
        wt_xs_carry=temp[fx].div(temp[[c+'_v' for c in fx]].values)

        ### include only top 8 currencies by magnitude of weights
#        rank=abs(wt_xs_carry[fx]).rank(axis=1,ascending=False)
#        rank[rank<=8]=1
#        rank[rank>8]=0    
#        wt_xs_carry[fx]=wt_xs_carry[fx].mul(rank)

        ##calculate carry trade expected return (as recent mean)
        expret=fxmono_ret_contamp[fx].fillna(method='ffill').rolling(window=30).mean()
        expret=expret.mul(wt_xs_carry)
    
        ## only invest in currencies with positive expected return    
        expret[expret>0]=1
        expret[expret<0]=0
        
        wt_xs_carry=wt_xs_carry.mul(expret)        
        
        #apply cap
        cap=0.25
        
        wt_xs_carry=wt_xs_carry.clip_upper(cap)
        wt_xs_carry=wt_xs_carry.clip_lower(-cap)

        
        ##calculate forward discount spread scaled by sum of absolute weights
        spread=abs(wt_xs_carry[fx]).sum(axis=1)
        ##calculate spread-weighted currency allocations
        wt_xs_carry=wt_xs_carry.div(spread,axis=0)
        
    if spread_weighted==False:
        wt_xs_carry=fwd_discount_sign.copy()
        ##number of currencies
        ncurr=fwd_discount_sign[fx].count(axis=1)
        ##calculate equal-weighted currency allocations
        wt_xs_carry[fx]=wt_xs_carry[fx].div(ncurr,axis=0)        
        ###lag weights time series
        wt_xs_carry=pd.concat([fwd_discount['Date'],wt_xs_carry[fx].shift(1)],axis=1)
          

    ###lag weights time series
    wt_xs_carry=pd.concat([fxmono_ret['Date'],wt_xs_carry.shift(1)],axis=1)

    ### add vol target overlay   
    if vol_overlay==True:
        temp=pd.merge(wt_xs_carry,cov_d[fx[0]][['Date',fx[0]]],on='Date',how='right',suffixes=('','_y')).drop(fx[0]+'_y',axis=1)
        var_portfolio=port_var(cov=cov_d,wt=temp,curr=fx)
        std_portfolio=pow(var_portfolio,0.5)
        # stdev threshold (vol_target=yearly vol)
        rr_std=vol_target/pow(252,0.5)
        leverage=pow(std_portfolio,-1)*rr_std
        # use leverage only after 1 year of cov matrix estimation
        leverage.iloc[0:252]=1
        # vol adjusted weights
        temp.iloc[:,1:]=np.asarray(temp.iloc[:,1:])*np.asarray(leverage)    
        wt_xs_carry=temp

    wt_xs_carry.iloc[:,1:]=wt_xs_carry.iloc[:,1:].clip_upper(cap)
    wt_xs_carry.iloc[:,1:]=wt_xs_carry.iloc[:,1:].clip_lower(-cap)

    return wt_xs_carry

def _strat(wt, fx):
    '''
    compute carry trade returns (or any other strategy) for each currency
    Use carry trade weights on last day of month t-1 to trade currencies in month t
    output cross-sectional carry trade returns for each currency
    '''    
    ##merge carry signal (weights) from previous trading day to realized returns (on next day)
    cols = list(wt.columns)
    temp=pd.merge(fxmono_ret[cols],wt,on='Date',how='left',suffixes=('','_w'))              

    xs_carry=pd.concat([temp['Date'],temp[fx].mul(temp[[c+'_w' for c in fx]].values)],axis=1)

    return xs_carry

def construct_factor(factor,name):
    '''
    compute carry trade strategy (or any other strategy) cumulative performance
    '''
    strat=factor.iloc[:,1:].sum(axis=1)+1
    strat[0]=100
    strat=strat.cumprod()
    strat=pd.concat([factor['Date'],strat],axis=1)
    strat.columns=['Date',name]
    
    return strat
 
    
#####OUTPUT CURRENCY WEIGHTS IN XS CARRY
wt_carry_dm=wt_xs_carry(fx=dm,spread_weighted=True,vol_overlay=True,vol_target=0.05)
wt_carry_em=wt_xs_carry(fx=em,spread_weighted=True,vol_overlay=True,vol_target=0.05) 
with open(os.path.join(data_dir, 'wt_carry_dm.pickle'), 'wb') as output:
    pickle.dump(wt_carry_dm, output)
with open(os.path.join(data_dir, 'wt_carry_em.pickle'), 'wb') as output:
    pickle.dump(wt_carry_em, output)

#APPLY VOL OVERLAY TO CARRY STRATEGIES -- APPLY 5% VOL TARGET

xs_carry_dm=construct_factor(
                _strat(
                        wt_xs_carry(fx=dm,spread_weighted=True,vol_overlay=True,vol_target=0.05),fx=dm),
            'Carry_DM')

xs_carry_em=construct_factor(
                _strat(
                        wt_xs_carry(fx=em,spread_weighted=True,vol_overlay=True,vol_target=0.05),fx=em),
            'Carry_EM')

    #Global Value as DM/EM average
ncurr_dm=fxmono_ret[dm].count(axis=1)
ncurr_em=fxmono_ret[em].count(axis=1)
wt_dm=ncurr_dm/(ncurr_dm+ncurr_em)
wt_em=1-wt_dm

xs_carry_glo = pd.merge(xs_carry_dm,xs_carry_em,on='Date',how='inner')  
xs_carry_glo['Carry_Global']=(xs_carry_glo['Carry_DM'].mul(wt_dm)+xs_carry_glo['Carry_EM'].mul(wt_em))
xs_carry_glo=xs_carry_glo.drop(['Carry_DM','Carry_EM'],axis=1)

plt.plot(xs_carry_dm.Date,xs_carry_dm.Carry_DM)
plt.plot(xs_carry_em.Date,xs_carry_em.Carry_EM)
plt.plot(xs_carry_glo.Date,xs_carry_glo.Carry_Global)

plt.plot(xs_carry_dm.Date[xs_carry_dm.Date>dt.datetime(2005,7,1)],xs_carry_dm.Carry_DM[xs_carry_dm.Date>dt.datetime(2005,7,1)])
plt.plot(xs_carry_em.Date[xs_carry_em.Date>dt.datetime(2005,7,1)],xs_carry_em.Carry_EM[xs_carry_em.Date>dt.datetime(2005,7,1)])
plt.plot(xs_carry_glo.Date[xs_carry_glo.Date>dt.datetime(2005,7,1)],xs_carry_glo.Carry_Global[xs_carry_glo.Date>dt.datetime(2005,7,1)])


with open(os.path.join(out_dir, 'xs_carry_dm.pickle'), 'wb') as output:
    pickle.dump(xs_carry_dm, output)

with open(os.path.join(out_dir, 'xs_carry_em.pickle'), 'wb') as output:
    pickle.dump(xs_carry_em, output)

with open(os.path.join(out_dir, 'xs_carry_glo.pickle'), 'wb') as output:
    pickle.dump(xs_carry_glo, output)

#############################################################
                        #DOLLAR CARRY#
#############################################################

def dollar_carry(fx,vol_overlay=False,vol_target=1):
    '''
    compute dollar carry trade weights:
    go long all currencies trading when forward discount > 0
    / short all currencies trading when avg forward discount < 0
    Optional param: vol_overlay to target portfolio 
    volatility at vol_target annually
    '''
    afd=fwd_discount[fx].mean(axis=1)
    afd[afd > 0] = 1
    afd[afd < 0] =-1
    afd[afd.isnull()] = 0    
   
    ##number of currencies
    ncurr=fwd_discount[fx].count(axis=1)
    
    ##calculate equal-weighted currency allocations
    wt_dollar_carry=afd.div(ncurr,axis=0)            
    wt_dollar_carry=pd.concat([fwd_discount['Date'],wt_dollar_carry.shift(1)],axis=1)
    wt_dollar_carry=pd.merge(fxmono_ret.iloc[:,:2],wt_dollar_carry,on='Date',how='left')[wt_dollar_carry.columns]
    wt_dollar_carry.columns=['Date','Dollar']
    
    ### add vol target overlay   
    if vol_overlay==True:
        temp1=pd.DataFrame()
        for c in fx:
            temp1[c]=wt_dollar_carry['Dollar']
        temp1=pd.concat([wt_dollar_carry['Date'],temp1],axis=1)
        
        temp=pd.merge(temp1,cov_d[fx[0]][['Date',fx[0]]],on='Date',how='right',suffixes=('','_y')).drop(fx[0]+'_y',axis=1)
        var_portfolio=port_var(cov=cov_d,wt=temp,curr=fx)
        std_portfolio=pow(var_portfolio,0.5)
        # stdev threshold (vol_target=yearly vol)
        rr_std=vol_target/pow(252,0.5)
        leverage=pow(std_portfolio,-1)*rr_std
        # use leverage only after 1 year of cov matrix estimation
        leverage.iloc[0:252]=1
        # vol adjusted weights
        wt_dollar_carry.iloc[1:,1:]=np.asarray(wt_dollar_carry.iloc[1:,1:])*np.asarray(leverage)    

    ##merge carry signal (weights) from previous trading day to realized returns (on next day)
    dollar_carry=pd.concat([fxmono_ret['Date'],fxmono_ret[fx].mul(wt_dollar_carry['Dollar'],axis=0)],axis=1)

    return dollar_carry, wt_dollar_carry

#####OUTPUT CURRENCY WEIGHTS IN DOLLAR CARRY
wt_dollar_dm=dollar_carry(fx=dm,vol_overlay=True,vol_target=0.05)[1]
wt_dollar_em=dollar_carry(fx=em,vol_overlay=True,vol_target=0.05)[1]
with open(os.path.join(data_dir, 'wt_dollar_dm.pickle'), 'wb') as output:
    pickle.dump(wt_dollar_dm, output)
with open(os.path.join(data_dir, 'wt_dollar_em.pickle'), 'wb') as output:
    pickle.dump(wt_dollar_em, output)


dollar_carry_dm=construct_factor(
        dollar_carry(fx=dm,vol_overlay=True,vol_target=0.05)[0],
        'DollarCarry_DM')

dollar_carry_em=construct_factor(
        dollar_carry(fx=em,vol_overlay=True,vol_target=0.05)[0],
        'DollarCarry_EM')

dollar_carry_glo = pd.merge(dollar_carry_dm,dollar_carry_em,on='Date',how='inner')  
dollar_carry_glo['DollarCarry_Global']=(dollar_carry_glo['DollarCarry_DM'].mul(wt_dm)+dollar_carry_glo['DollarCarry_EM'].mul(wt_em))
dollar_carry_glo=dollar_carry_glo.drop(['DollarCarry_DM','DollarCarry_EM'],axis=1)

plt.plot(dollar_carry_dm.Date,dollar_carry_dm.DollarCarry_DM)
plt.plot(dollar_carry_em.Date,dollar_carry_em.DollarCarry_EM)
plt.plot(dollar_carry_glo.Date,dollar_carry_glo.DollarCarry_Global)


with open(os.path.join(out_dir, 'dollar_carry_dm.pickle'), 'wb') as output:
    pickle.dump(dollar_carry_dm, output)

with open(os.path.join(out_dir, 'dollar_carry_em.pickle'), 'wb') as output:
    pickle.dump(dollar_carry_em, output)

with open(os.path.join(out_dir, 'dollar_carry_glo.pickle'), 'wb') as output:
    pickle.dump(dollar_carry_glo, output)

#############################################################
        #Time series MOMENTUM#
#############################################################

    ### convert to dollars per unit of foreign currency
spots=fx_spot.copy()
spots.iloc[:,1:]=1/spots.iloc[:,1:]

    ### compute score (as z scores) for 12 time horizons -- 1m to 12m in monthly increments
spot_ret=fns.ret(spots).iloc[:,1:]

score=pd.DataFrame()
for i in range(1,13):
    roll_window=spot_ret.rolling(window=(21*i),min_periods=1)
    score_i=roll_window.mean()/roll_window.std()
    score=score.add(score_i,axis=0,fill_value=0)
    
    ###final score is the average of 12 z scores    
score=pd.concat([spots[['Date']],score],axis=1)

def momentum(fx,vol_overlay=False,vol_target=1):
    '''
    compute time series momentum weights:
    weights equal to the average z score
    Optional param: vol_overlay to target portfolio 
    volatility at vol_target annually
    '''
  
    ### add vol target overlay   
    if vol_overlay==True:
        temp=pd.merge(score,cov_d[fx[0]][['Date',fx[0]]],on='Date',how='right',suffixes=('','_y')).drop(fx[0]+'_y',axis=1)
        var_portfolio=port_var(cov=cov_d,wt=temp,curr=fx)
        std_portfolio=pow(var_portfolio,0.5)
        # stdev threshold (vol_target=yearly vol)
        rr_std=vol_target/pow(252,0.5)
        leverage=pow(std_portfolio,-1)*rr_std
        # use leverage only after 1 year of cov matrix estimation
        leverage.iloc[0:252]=1
        # vol adjusted weights
        temp.iloc[:,1:]=np.asarray(temp.iloc[:,1:])*np.asarray(leverage)    
        wt_mom=temp

    if vol_overlay==False:
        wt_mom=score

        #apply cap -- 2x leverage
    cap=2
        
    wt_mom.iloc[:,1:]=wt_mom.iloc[:,1:].clip_upper(cap)
    wt_mom.iloc[:,1:]=wt_mom.iloc[:,1:].clip_lower(-cap)

    return wt_mom

#####OUTPUT CURRENCY WEIGHTS IN DOLLAR CARRY
wt_mom_dm=momentum(fx=dm,vol_overlay=True,vol_target=0.05)
wt_mom_em=momentum(fx=em,vol_overlay=True,vol_target=0.05)

with open(os.path.join(data_dir, 'wt_mom_dm.pickle'), 'wb') as output:
    pickle.dump(wt_mom_dm, output)
with open(os.path.join(data_dir, 'wt_mom_em.pickle'), 'wb') as output:
    pickle.dump(wt_mom_em, output)


mom_dm=construct_factor(
                _strat(
                        momentum(fx=dm,vol_overlay=True,vol_target=0.05),fx=dm),
            'Momentum_DM')

mom_em=construct_factor(
                _strat(
                        momentum(fx=em,vol_overlay=True,vol_target=0.05),fx=em),
            'Momentum_EM')

plt.plot(mom_dm.Date,mom_dm.Momentum_DM)
plt.plot(mom_em.Date,mom_em.Momentum_EM)

with open(os.path.join(out_dir, 'mom_dm.pickle'), 'wb') as output:
    pickle.dump(mom_dm, output)

with open(os.path.join(out_dir, 'mom_em.pickle'), 'wb') as output:
    pickle.dump(mom_em, output)

#############################################################
        #Mean Reversion#       
#############################################################

score=pd.DataFrame()
    ### compute score spot - X day MA
roll_mean=spots.iloc[:,1:].rolling(window=(252),min_periods=30).mean()
roll_std =spots.iloc[:,1:].rolling(window=(252),min_periods=30).std()
score=spots.iloc[:,1:].sub(roll_mean,axis=1)
score[score > 0] = 1
score[score < 0] =-1

    ### test for stationarity (ADF)
# time series library
import statsmodels.tsa.stattools as ts

#lookback period for stationarity test (months)
lookback=6
n=lookback*21

stationarity=pd.DataFrame(columns=spot_ret.columns, index=spot_ret.index)

for i in range(len(spot_ret.columns)):
    for t in range(n,len(spot_ret)):
        stationarity.iloc[t,i]=ts.adfuller(spots.iloc[:,1:].iloc[t-n:t,i].fillna(0))[1]

with open(os.path.join(out_dir, 'stationarity.pickle'), 'wb') as output:
    pickle.dump(stationarity, output)

fname=os.path.join(out_dir, 'stationarity.pickle')
stationarity=pickle.load( open( fname, "rb" ) )

stationarity[stationarity>0.01]=np.nan
stationarity[stationarity<0.01]=1.0
score=score.mul(stationarity.astype(float),axis=1)

'''
    ##test stationarity with monthly series
spot_ret_m=fns.monthly_ret(spots).iloc[:,1:]
stationarity_m=pd.DataFrame(columns=spot_ret_m.columns, index=spot_ret_m.index)

n=lookback*12
for i in range(len(spot_ret_m.columns)):
    for t in range(n,len(spot_ret_m)):
        stationarity_m.iloc[t,i]=ts.adfuller(spot_ret_m.iloc[t-n:t,i].fillna(0))[1]


###expand monthly stationarity to daily data
stationarity_m=pd.concat([fns.monthly_ret(spots).iloc[:,0],stationarity_m],axis=1)
stationarity_m=pd.merge(spots[['Date']],stationarity_m,on='Date',how='left').fillna(method='ffill')
stationarity_m=stationarity_m.drop('Date',axis=1)
with open(os.path.join(out_dir, 'stationarity_m.pickle'), 'wb') as output:
    pickle.dump(stationarity_m, output)
fname=os.path.join(out_dir, 'stationarity_m.pickle')
stationarity_m=pickle.load( open( fname, "rb" ) )

stationarity_m[stationarity_m>0.01]=np.nan
stationarity_m[stationarity_m<0.01]=1.0

score=score.mul(stationarity_m.astype(float),axis=1)
'''    
    ### compute binary weights with opposite sign as score
wt_meanrev=pd.concat([spots[['Date']],-1*score],axis=1)

wt_meanrev_dm=wt_meanrev.copy()
wt_meanrev_dm[dm]=wt_meanrev_dm[dm].div(wt_meanrev[dm].count(1).values,axis=0)
wt_meanrev_em=wt_meanrev.copy()
wt_meanrev_em[em]=wt_meanrev_em[em].div(wt_meanrev[em].count(1).values,axis=0)

meanrev_dm=construct_factor(
                _strat(
                        wt_meanrev_dm,fx=dm),
            'MeanRev_DM')

meanrev_em=construct_factor(
                _strat(
                        wt_meanrev_em,fx=em),
            'MeanRev_EM')

plt.plot(meanrev_dm.Date,meanrev_dm.MeanRev_DM)
plt.plot(meanrev_em.Date,meanrev_em.MeanRev_EM)
   
with open(os.path.join(out_dir, 'meanrev_dm.pickle'), 'wb') as output:
    pickle.dump(meanrev_dm, output)

with open(os.path.join(out_dir, 'meanrev_em.pickle'), 'wb') as output:
    pickle.dump(meanrev_em, output)
    
             
#############################################################
        #LRV 2014 - PORTFOLIO BASED CARRY FACTOR#
#############################################################

def _carry(fx):
    '''
    compute cross-sectional carry trade weights:
    go long currencies trading with forward discount
    with a weight proportional to the spread (if spread_weighted)
    otherwise equal weighted in the direction of fwd discount
    Optional param: vol_overlay to target portfolio 
    volatility at vol_target annually
    '''            
    ###lag weights time series
    wt_xs_carry=pd.concat([fwd_discount['Date'],fwd_discount[fx].shift(1)],axis=1)
    wt_xs_carry=pd.merge(fxmono_ret[['Date']+fx],wt_xs_carry,on='Date',how='left',suffixes=('_x',''))[['Date']+fx]

    temp1=wt_xs_carry[fx]
    temp1['index']=temp1.index
    temp1=pd.melt(temp1,id_vars=['index'],var_name='curr', value_name='wt')
    temp1=temp1.sort_values(by=['index','wt']).reset_index(drop=True)
    temp1.dropna(inplace=True)
       
    temp2=pd.merge(wt_xs_carry,fxmono_ret,on='Date',how='left',suffixes=('_x',''))[fx]
    temp2['index']=temp2.index
    temp2=pd.melt(temp2,id_vars=['index'],var_name='curr', value_name='rx')
    
    temp=pd.merge(temp1,temp2,on=['index','curr'],how='left')
    temp.dropna(inplace=True)

    ncurrs=pd.DataFrame(temp.groupby(['index']).size())
    ncurrs.columns=['ncurrs']
    ncurrs.ncurrs.unique()
    
    temp=temp.merge(ncurrs,how='left',on='index')
    temp['rank']=temp.groupby(['index']).cumcount()+1
    temp=temp[temp['ncurrs']>=8].reset_index(drop=True)

    
    conditions= [
    
        (temp['ncurrs']<=9) & (temp['rank']<3),
        (temp['ncurrs']<=9) & (temp['rank']>=3) & (temp['rank']<5),
        (temp['ncurrs']<=9) & (temp['rank']>=5) & (temp['rank']<7),
        (temp['ncurrs']<=9) & (temp['rank']>=7) & (temp['rank']<10),
    
        (temp['ncurrs']>9) & (temp['ncurrs']<=11) & (temp['rank']<4),
        (temp['ncurrs']>9) & (temp['ncurrs']<=11) & (temp['rank']>=4) & (temp['rank']<6),
        (temp['ncurrs']>9) & (temp['ncurrs']<=11) & (temp['rank']>=6) & (temp['rank']<9),
        (temp['ncurrs']>9) & (temp['ncurrs']<=11) & (temp['rank']>=9) & (temp['rank']<12),
    
        (temp['ncurrs']>11) & (temp['ncurrs']<=14) & (temp['rank']<5),
        (temp['ncurrs']>11) & (temp['ncurrs']<=14) & (temp['rank']>=5) & (temp['rank']<8),
        (temp['ncurrs']>11) & (temp['ncurrs']<=14) & (temp['rank']>=8) & (temp['rank']<11),
        (temp['ncurrs']>11) & (temp['ncurrs']<=14) & (temp['rank']>=11) & (temp['rank']<15),
        
    ]
    
    choices=['p1','p2','p3','p4','p1','p2','p3','p4','p1','p2','p3','p4']    
    temp['portfolio'] = np.select(conditions, choices)
    
    xs_portfolios=temp.groupby(['index','portfolio']).mean().reset_index()
    xs_portfolios=xs_portfolios.pivot(index='index',columns='portfolio',values='rx')

    xs_portfolios['Dollar']=xs_portfolios.mean(axis=1)
    xs_portfolios['Carry']=(xs_portfolios['p4']-xs_portfolios['p1'])
    
    xs_portfolios['index']=xs_portfolios.index
    wt_xs_carry['index']=wt_xs_carry.index

    carry=pd.merge(xs_portfolios,wt_xs_carry,on='index',how='left').reset_index()[['Date','Dollar','Carry']]

    return carry

p_dollar_carry_dm=construct_carry_factor(
        _carry(fx=dm).drop(['Carry'],axis=1),
        'DollarCarry_DM')

p_dollar_carry_em=construct_carry_factor(
        _carry(fx=em).drop(['Carry'],axis=1),
        'DollarCarry_EM')

plt.plot(dollar_carry_dm.Date,dollar_carry_dm.DollarCarry_DM)
plt.plot(dollar_carry_em.Date,dollar_carry_em.DollarCarry_EM)
plt.plot(p_dollar_carry_dm.Date,p_dollar_carry_dm.DollarCarry_DM)
plt.plot(p_dollar_carry_em.Date,p_dollar_carry_em.DollarCarry_EM)

p_dollar_carry_dm.DollarCarry_DM.corr(dollar_carry_dm.DollarCarry_DM)

p_carry_dm=construct_carry_factor(
        _carry(fx=dm).drop(['Dollar'],axis=1),
        'Carry_DM')

p_carry_em=construct_carry_factor(
        _carry(fx=em).drop(['Dollar'],axis=1),
        'Carry_EM')


plt.plot(xs_carry_dm.Date,xs_carry_dm.Carry_DM)
plt.plot(xs_carry_em.Date,xs_carry_em.Carry_EM)
plt.plot(p_carry_dm.Date,p_carry_dm.Carry_DM)
plt.plot(p_carry_em.Date,p_carry_em.Carry_EM)

p_carry_dm.Carry_DM.corr(xs_carry_dm.Carry_DM)

#############################################################
                        #TEST#
#############################################################

    ###Import ML FX RP indices
fname=os.path.join(out_dir, 'fx_factors_ml.pickle')
test = pickle.load( open( fname, "rb" ) )
test=test[['Date','XS_Carry_Global']].dropna()
test=fns.ret(pd.merge(test,xs_carry_glo,on='Date',how='inner'))
test_ml=construct_carry_factor(test[['Date','XS_Carry_Global']],'ML')
test_rec=construct_carry_factor(test[['Date','Carry_Global']],'Rec')

plt.plot(test_ml.Date,test_ml.ML)
plt.plot(test_rec.Date,test_rec.Rec)

plt.plot(test_ml.Date[(test_ml.Date>dt.datetime(2014,4,1))][(test_ml.Date<dt.datetime(2015,12,1))],test_ml.ML[(test_ml.Date>dt.datetime(2014,4,1))][(test_ml.Date<dt.datetime(2015,12,1))])
plt.plot(test_rec.Date[(test_rec.Date>dt.datetime(2014,4,1))][(test_rec.Date<dt.datetime(2015,12,1))],test_rec.Rec[(test_rec.Date>dt.datetime(2014,4,1))][(test_rec.Date<dt.datetime(2015,12,1))])

wt_dm=wt_xs_carry(fx=dm,spread_weighted=True,vol_overlay=True,vol_target=0.05)
wt_dm=wt_dm[wt_dm.Date>dt.datetime(2014,4,1)][wt_dm.Date<dt.datetime(2015,7,1)]
carry_dm=carry_strat(wt_dm,dm)
carry_dm=carry_dm[carry_dm.Date>dt.datetime(2014,4,1)][carry_dm.Date<dt.datetime(2015,7,1)]

wt_dm.mean()
carry_dm.mean()

wt_em=wt_xs_carry(fx=em,spread_weighted=True,vol_overlay=True,vol_target=0.05)
carry_em=carry_strat(wt_em,em)
carry_em=carry_em[carry_em.Date>dt.datetime(2014,4,1)][carry_em.Date<dt.datetime(2015,7,1)]

### strategy info ratios
def info(ser):
    ret=fns.ret(ser).iloc[:,1]
    info= (ret.mean() / ret.std()) * pow(252,0.5)
    return info

info(xs_carry_dm)
info(xs_carry_em)
info(xs_carry_glo)
info(test_ml)

info(value_dm)
info(value_em)

info(dollar_carry_dm)
info(dollar_carry_em)
info(dollar_carry_glo)

xs_carry_glo.Carry_Global.corr(dollar_carry_glo.DollarCarry_Global)

x=pd.concat([1/fx_spot[["European Union"]],1/fx_spot[["European Union"]].rolling(window=125).mean().shift(2)],axis=1)
y=x.diff(axis=1).iloc[::5,1].dropna()
y[y>0].count()
plt.plot(x)

