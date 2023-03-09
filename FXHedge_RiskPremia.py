# -*- coding: utf-8 -*-
"""
Created on Sep 6 2019

@author: jamal
"""
'''
To Update Optimal Hedge Ratio calculation, run script till part #4 &
add function call for additional hedging instruments at the end
'''


from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
import datetime as dt
###import helper functions
os.chdir('//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies/Code/FX_RPproject')
import HelperFunctions as fns

########UPDATE PATH############
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/Datasets'
out_dir = '//corp.bankofamerica.com/london/Researchshared/Share4/Sector/Commodities/Publications/FICC Portfolio Monthly/2019_enhanced hedging/FX risk premia strategies/Data'
###############################


#############################################################
#1. Load Data
#############################################################

    ###Import Equity and FI indices
fname=os.path.join(data_dir, 'equity_fx_indices.pickle')
assets = pickle.load( open( fname, "rb" ) )

    ###Import ML FX RP indices
fname=os.path.join(data_dir, 'fx_factors_ml.pickle')
fx_factors_ml = pickle.load( open( fname, "rb" ) )

    ###Import US trade weighted reconstructed indices
fname=os.path.join(data_dir, 'ustw_afe_index.pickle')
ustw_afe = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'ustw_index.pickle')
ustw_broad = pickle.load( open( fname, "rb" ) )

    ###Import dollar spot
fname=os.path.join(data_dir, 'dxy.pickle')
dxy = pickle.load( open( fname, "rb" ) )

ustw=pd.merge(ustw_afe,ustw_broad,how='outer')

    ###Import JK FX RP indices
#Value    
fname=os.path.join(data_dir, 'fx_value_dm.pickle')
fx_value_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'fx_value_em.pickle')
fx_value_em = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'fx_value_gl.pickle')
fx_value_gl = pickle.load( open( fname, "rb" ) )

#Carry
fname=os.path.join(data_dir, 'xs_carry_dm.pickle')
xs_carry_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'xs_carry_em.pickle')
xs_carry_em = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'xs_carry_glo.pickle')
xs_carry_glo = pickle.load( open( fname, "rb" ) )

#Dollar
fname=os.path.join(data_dir, 'dollar_carry_dm.pickle')
dollar_carry_dm = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'dollar_carry_em.pickle')
dollar_carry_em = pickle.load( open( fname, "rb" ) )
fname=os.path.join(data_dir, 'dollar_carry_glo.pickle')
dollar_carry_glo = pickle.load( open( fname, "rb" ) )

def combine_factors(df1,df2):
    out=pd.merge(df1,df2,how='outer',on='Date')
    return out

fx_factors=combine_factors(
        combine_factors(
                combine_factors(
                        combine_factors(
                                combine_factors(
                                        combine_factors(
                                                combine_factors(
                                                        combine_factors(fx_value_dm,fx_value_em),fx_value_gl),xs_carry_dm),xs_carry_em),xs_carry_glo),dollar_carry_dm),dollar_carry_em),dollar_carry_glo)

with open(os.path.join(out_dir, 'fx_factors.pickle'), 'wb') as output:
    pickle.dump(fx_factors, output)

#writer=pd.ExcelWriter(os.path.join(out_dir,'factor_correl.xlsx'))
#fx_factors.corr().to_excel(writer,'corr_daily')
#writer.save()

############################################
    ### CALCULATE RETURNS
############################################

assets=fns.ret(assets)
ustw_afe=fns.ret(ustw_afe)
ustw_broad=fns.ret(ustw_broad)
dxy=fns.ret(dxy)

temp1=pd.merge(dxy,assets,how='inner')[dxy.columns]
temp2=pd.merge(dxy,assets,how='inner')[assets.columns]

dxy=temp1
assets=temp2

##########################################
###calculate asset returns in DXY terms###
###(what a foriegn investor buying #######
### with DXY currency basket would make)##
##########################################

#perfect hedge i.e. returns in USD
assets_ph=assets.copy()

temp=pd.DataFrame((1+assets.iloc[:,1:].values)*(1+dxy.iloc[:,1:].values)-1,columns=assets.columns[1:],index=assets.index)
assets.iloc[:,1:]=temp


#############################################################
#2. Factor USTWER Correlations
#############################################################
    ### daily returns
ustw_factors=pd.merge(fns.ret(ustw),fns.ret(fx_factors),how='left')
corr_fx_all=pd.concat([ustw_factors,fns.ret(ustw)], axis=1, keys=['ustw_factors','ustw']).corr()
corr_fx=corr_fx_all.loc['ustw_factors','ustw']
'''
writer=pd.ExcelWriter(os.path.join(out_dir,'corr_fx.xlsx'))
corr_fx.to_excel(writer,'corr_fx_daily')
writer.save()
'''

#############################################################
#3. Construct FX Risk Premia Baskets 
#############################################################

    ## forward fill fx factors data
fx_factors[fx_factors['Date']==dt.datetime(2017,5,1)]
fx_factors=fx_factors.fillna(method='ffill')
### keep Global versions of carry and Value
    ### high correlation between EM, G-10 and Global factors
    ### restrict factor span to minimize multicollinearity    
fx_factors.columns
fx_factors_reg=fx_factors.drop(['Value_Global','Carry_Global','DollarCarry_Global'],axis=1)
fx_factors_reg.columns


#COMPUTE INVERSE VOL WEIGHTED FX RP BASKET
    ## 1 year rolling vol with at least 1 month history
fxrp_basket_ivol=1/fx_factors.iloc[:,1:].rolling(window=252,min_periods=30).std()
fxrp_basket_ivol=fxrp_basket_ivol.div(fxrp_basket_ivol.sum(axis=1),axis=0)
fxrp_basket_ivol=fx_factors.iloc[:,1:].mul(fxrp_basket_ivol,axis=1)
fxrp_basket_ivol=fxrp_basket_ivol.sum(axis=1)
fxrp_basket_ivol=pd.concat([fx_factors['Date'],fxrp_basket_ivol],axis=1)
fxrp_basket_ivol.rename(columns={0:'fxrp_basket_ivol'},inplace=True)

with open(os.path.join(out_dir, 'fxrp_basket_ivol.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_ivol, output)

fxrp_basket_ivol=fns.ret(fxrp_basket_ivol)

#DETERMINE OPTIMAL FACTOR WEIGHTS FROM ROLLING REGRESSIONS

    ## 1 year rolling OLS regression: DXY returns ~ FXRP returns 

dxy_fxrp_b,dxy_fxrp_p=fns.roll_reg(ydat=dxy,xdat=fns.ret(fx_factors_reg),name='DXY Curncy',roll_window=252)

#normalize weights
fxrp_basket_reg=dxy_fxrp_b.drop(['Date','const'],axis=1)
minim=fxrp_basket_reg.min(axis=1).replace(0, np.nan)
maxim=fxrp_basket_reg.max(axis=1).replace(0, np.nan)
diff=maxim-minim
count=fxrp_basket_reg.count(axis=1)

#take factor value when count=1
fxrp_basket_reg[count<=1]=np.sign(fx_factors_reg.iloc[:,1:][count<=1])
#form basket when >1 factor series available
fxrp_basket_reg[count>1]=(fxrp_basket_reg[count>1].sub(minim[count>1],axis=0)).div(diff[count>1],axis=0)
    #scale weights to sum = 1
temp=fxrp_basket_reg.sum(axis=1).replace(0, np.nan)    
fxrp_basket_reg = fxrp_basket_reg.div(temp,axis=0)  

#multiply weights by factor values
fxrp_basket_reg=fx_factors_reg.iloc[:,1:].mul(fxrp_basket_reg,axis=1)
fxrp_basket_reg=fxrp_basket_reg.sum(axis=1)
fxrp_basket_reg=pd.concat([fx_factors_reg['Date'],fxrp_basket_reg],axis=1).replace(0, np.nan)
fxrp_basket_reg.rename(columns={0:'fxrp_basket_reg'},inplace=True)

fxrp_basket_reg['fxrp_basket_reg'].plot(subplots=True)

#writer=pd.ExcelWriter(os.path.join(out_dir,'test2.xlsx'))
#fx_factors_reg.to_excel(writer,'factors')
#fxrp_basket_reg.to_excel(writer,'betas')
#writer.save()


with open(os.path.join(out_dir, 'fxrp_basket_reg.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_reg, output)

fxrp_basket_reg=fns.ret(fxrp_basket_reg)

    ## 1 year rolling Ridge regression: DXY returns ~ FXRP returns 

dxy_fxrp_b_ridge =fns.roll_reg_ridge(ydat=dxy,xdat=fns.ret(fx_factors_reg),name='DXY Curncy',roll_window=252)

#normalize weights
fxrp_basket_reg_rid=dxy_fxrp_b_ridge .drop(['Date'],axis=1)
minim=fxrp_basket_reg_rid.min(axis=1).replace(0, np.nan)
maxim=fxrp_basket_reg_rid.max(axis=1).replace(0, np.nan)
diff=maxim-minim
count=fxrp_basket_reg_rid.count(axis=1)

#take factor value when count=1
fxrp_basket_reg_rid[count<=1]=np.sign(fx_factors_reg.iloc[:,1:][count<=1])
#form basket when >1 factor series available
fxrp_basket_reg_rid[count>1]=(fxrp_basket_reg_rid[count>1].sub(minim[count>1],axis=0)).div(diff[count>1],axis=0)
    #scale weights to sum = 1
temp=fxrp_basket_reg_rid.sum(axis=1).replace(0, np.nan)    
fxrp_basket_reg_rid = fxrp_basket_reg_rid.div(temp,axis=0)  

#multiply weights by factor values
fxrp_basket_reg_rid=fx_factors_reg.iloc[:,1:].mul(fxrp_basket_reg_rid,axis=1)
fxrp_basket_reg_rid=fxrp_basket_reg_rid.sum(axis=1)
fxrp_basket_reg_rid=pd.concat([fx_factors_reg['Date'],fxrp_basket_reg_rid],axis=1).replace(0, np.nan)
fxrp_basket_reg_rid.rename(columns={0:'fxrp_basket_reg_rid'},inplace=True)

fxrp_basket_reg_rid['fxrp_basket_reg_rid'].plot(subplots=True)

with open(os.path.join(out_dir, 'fxrp_basket_reg_rid.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_reg_rid, output)

fxrp_basket_reg_rid=fns.ret(fxrp_basket_reg_rid)

fxrp_basket_reg_rid.min()

#writer=pd.ExcelWriter(os.path.join(out_dir,'test3.xlsx'))
#fx_factors_reg.to_excel(writer,'factors')
#ustw_fxrp_b_ridge.to_excel(writer,'reg')
#writer.save()



#########
#Regress daily asset returns on USTW index returns  
#Calculate 1-year rolling beta to get optimal hedge ratio for that asset
#Invest in $beta amount of USTW to hedge 
#Rebalance monthly 
#Track net basket returns (asset+currency) 
#########

#############################################################
#4. Calculate Optimal Hedge Ratios --  
#    Rolling regressions > Asset Returns ~ Hedging Instrument Returns
#############################################################

asset_tickers=list(assets.columns)[1:]

def run_reg(ydat,xdat,names,roll_window):
    '''
    Main OLS regression function 
    input: dep & indep vars, variable names, window (days) for rolling regressions (window=ydat.shape[0] for full period)
    output: if rolling, two dictionaries (key:value pairs as asset name:dataframe) 
    containing parameters and corresponding pvalues for each asset
    '''
    ### Rolling Regressions
    ### SET PARAMS
            
    # Regression window size
    n = roll_window # days / year
    # Index counter for rolling regressions
    t_start = n
    t_end = xdat.shape[0] 
                   
    ### Storage Dicts
    roll_reg_b = {}
    roll_reg_p = {}
                
    for p in names:
          
        X = xdat.drop(['Date'],axis=1)
        X = sm.add_constant(X)
        X = X.replace(np.inf, np.nan)
        Y = pd.merge(xdat,ydat,on='Date',how='left')[p]

        ols_roll   = pd.DataFrame(index=list(range(t_end)),columns=X.columns)
        ols_roll_p = pd.DataFrame(index=list(range(t_end)),columns=X.columns)    
            
        for t in range(t_start,t_end+1):
            X_roll = X.iloc[t-n:t,:]
            Y_roll = Y.iloc[t-n:t]
            try:
                rollreg = fns.ols_call(Y_roll,X_roll)
                ols_roll.iloc[t-1,:]   = rollreg.params
                ols_roll_p.iloc[t-1,:] = rollreg.pvalues
            except:
                continue
            
        roll_reg_b[p]=pd.concat([xdat['Date'],ols_roll],axis=1)
        roll_reg_p[p]=pd.concat([xdat['Date'],ols_roll_p],axis=1)
        
    return roll_reg_b, roll_reg_p

#remove NAs from dfs
def dropna_(df):
    df=df.dropna(thresh=2)
    return df

#############################################################
#5.   Function calls for dollar hedge
#############################################################

##########USTW_AFE##################

### 1-Year Rolling Regression
ustw_hedge_b,ustw_hedge_p = run_reg(ydat=assets,xdat=ustw_afe,names=asset_tickers,roll_window=252)

ustw_hedge_b = {k: dropna_(v) for k, v in ustw_hedge_b.items()}
ustw_hedge_p = {k: dropna_(v) for k, v in ustw_hedge_p.items()}

with open(os.path.join(out_dir, 'ustw_hedge_b.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_b, output)
with open(os.path.join(out_dir, 'ustw_hedge_p.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_p, output)

### Full Period Regression
ustw_hedge_b_full,ustw_hedge_p_full= run_reg(ydat=assets,xdat=ustw_afe,names=asset_tickers,roll_window=ustw_afe.shape[0])

#remove NAs from dfs
ustw_hedge_b_full = {k: dropna_(v) for k, v in ustw_hedge_b_full.items()}
ustw_hedge_p_full = {k: dropna_(v) for k, v in ustw_hedge_p_full.items()}

with open(os.path.join(out_dir, 'ustw_hedge_b_full.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_b_full, output)
with open(os.path.join(out_dir, 'ustw_hedge_p_full.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_p_full, output)


##########USTW_BROAD##################
    
### 1-Year Rolling Regression
ustwbroa_hedge_b,ustwbroa_hedge_p = run_reg(ydat=assets,xdat=ustw_broad,names=asset_tickers,roll_window=252)

#remove NAs from dfs
ustwbroa_hedge_b = {k: dropna_(v) for k, v in ustwbroa_hedge_b.items()}
ustwbroa_hedge_p = {k: dropna_(v) for k, v in ustwbroa_hedge_p.items()}

with open(os.path.join(out_dir, 'ustwbroa_hedge_b.pickle'), 'wb') as output:
    pickle.dump(ustwbroa_hedge_b, output)
with open(os.path.join(out_dir, 'ustwbroa_hedge_p.pickle'), 'wb') as output:
    pickle.dump(ustwbroa_hedge_p, output)

### Full Period Regression
ustw_broa_hedge_b_full,ustw_broa_hedge_p_full= run_reg(ydat=assets,xdat=ustw_broad,names=asset_tickers,roll_window=ustw_broad.shape[0])

#remove NAs from dfs
ustw_broa_hedge_b_full = {k: dropna_(v) for k, v in ustw_broa_hedge_b_full.items()}
ustw_broa_hedge_p_full = {k: dropna_(v) for k, v in ustw_broa_hedge_p_full.items()}

with open(os.path.join(out_dir, 'ustw_broa_hedge_b_full.pickle'), 'wb') as output:
    pickle.dump(ustw_broa_hedge_b_full, output)
with open(os.path.join(out_dir, 'ustw_broa_hedge_p_full.pickle'), 'wb') as output:
    pickle.dump(ustw_broa_hedge_p_full, output)
 
        
#############################################################
#6.   Function calls for FX Risk Premia hedge
#############################################################
 
########## Max Correlation Vol Basket ##################
    
### 1-Year Rolling Ridge Regression
fxrp_reg_rid_hedge_b,fxrp_reg_rid_hedge_p = run_reg(ydat=assets,xdat=fxrp_basket_reg_rid,names=asset_tickers,roll_window=252)

#remove NAs from dfs
fxrp_reg_rid_hedge_b = {k: dropna_(v) for k, v in fxrp_reg_rid_hedge_b.items()}
fxrp_reg_rid_hedge_p = {k: dropna_(v) for k, v in fxrp_reg_rid_hedge_p.items()}

with open(os.path.join(out_dir, 'fxrp_reg_rid_hedge_b.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_rid_hedge_b, output)
with open(os.path.join(out_dir, 'fxrp_reg_rid_hedge_p.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_rid_hedge_p, output)

### Full Period Regression
fxrp_reg_rid_hedge_b_full,fxrp_reg_rid_hedge_p_full= run_reg(ydat=assets,xdat=fxrp_basket_reg_rid,names=asset_tickers,roll_window=fxrp_basket_reg_rid.shape[0])

#remove NAs from dfs
fxrp_reg_rid_hedge_b_full = {k: dropna_(v) for k, v in fxrp_reg_rid_hedge_b_full.items()}
fxrp_reg_rid_hedge_p_full = {k: dropna_(v) for k, v in fxrp_reg_rid_hedge_p_full.items()}

with open(os.path.join(out_dir, 'fxrp_reg_rid_hedge_b_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_rid_hedge_b_full, output)
with open(os.path.join(out_dir, 'fxrp_reg_rid_hedge_p_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_rid_hedge_p_full, output)
       
### 1-Year Rolling OLS Regression
fxrp_reg_hedge_b,fxrp_reg_hedge_p = run_reg(ydat=assets,xdat=fxrp_basket_reg,names=asset_tickers,roll_window=252)

#remove NAs from dfs
fxrp_reg_hedge_b = {k: dropna_(v) for k, v in fxrp_reg_hedge_b.items()}
fxrp_reg_hedge_p = {k: dropna_(v) for k, v in fxrp_reg_hedge_p.items()}

with open(os.path.join(out_dir, 'fxrp_reg_hedge_b.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_b, output)
with open(os.path.join(out_dir, 'fxrp_reg_hedge_p.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_p, output)

### Full Period Regression
fxrp_reg_hedge_b_full,fxrp_reg_hedge_p_full= run_reg(ydat=assets,xdat=fxrp_basket_reg,names=asset_tickers,roll_window=fxrp_basket_reg.shape[0])

#remove NAs from dfs
fxrp_reg_hedge_b_full = {k: dropna_(v) for k, v in fxrp_reg_hedge_b_full.items()}
fxrp_reg_hedge_p_full = {k: dropna_(v) for k, v in fxrp_reg_hedge_p_full.items()}

with open(os.path.join(out_dir, 'fxrp_reg_hedge_b_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_b_full, output)
with open(os.path.join(out_dir, 'fxrp_reg_hedge_p_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_p_full, output)
    
##########Inverse Vol Basket##################
    
### 1-Year Rolling Regression
fxrp_ivol_hedge_b,fxrp_ivol_hedge_p = run_reg(ydat=assets,xdat=fxrp_basket_ivol,names=asset_tickers,roll_window=252)

#remove NAs from dfs
fxrp_ivol_hedge_b = {k: dropna_(v) for k, v in fxrp_ivol_hedge_b.items()}
fxrp_ivol_hedge_p = {k: dropna_(v) for k, v in fxrp_ivol_hedge_p.items()}

with open(os.path.join(out_dir, 'fxrp_ivol_hedge_b.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_b, output)
with open(os.path.join(out_dir, 'fxrp_ivol_hedge_p.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_p, output)

### Full Period Regression
fxrp_ivol_hedge_b_full,fxrp_ivol_hedge_p_full= run_reg(ydat=assets,xdat=fxrp_basket_ivol,names=asset_tickers,roll_window=fxrp_basket_ivol.shape[0])

#remove NAs from dfs
fxrp_ivol_hedge_b_full = {k: dropna_(v) for k, v in fxrp_ivol_hedge_b_full.items()}
fxrp_ivol_hedge_p_full = {k: dropna_(v) for k, v in fxrp_ivol_hedge_p_full.items()}

with open(os.path.join(out_dir, 'fxrp_ivol_hedge_b_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_b_full, output)
with open(os.path.join(out_dir, 'fxrp_ivol_hedge_p_full.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_p_full, output)
    

#############################################################
#7. Calculate Optimal Hedge Ratios --  
###Repeat with monthly data
#############################################################

    ###Import Equity and FI indices
fname=os.path.join(data_dir, 'equity_fx_indices.pickle')
assets = pickle.load( open( fname, "rb" ) )

    ###Import ML FX RP indices
fname=os.path.join(data_dir, 'fx_factors_ml.pickle')
fx_factors_ml = pickle.load( open( fname, "rb" ) )

    ###Import JK FX RP indices
fx_factors=fx_factors_reg

    ###Import US trade weighted reconstructed indices
fname=os.path.join(data_dir, 'ustw_afe_index.pickle')
ustw_afe = pickle.load( open( fname, "rb" ) )

    ###Import dollar spot
fname=os.path.join(data_dir, 'dxy.pickle')
dxy = pickle.load( open( fname, "rb" ) )


    ### 1-month holding period for strategy
        # generate signal on last trading day of month
        # trade/rebalance on first trading day of month

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

#############################################################
#7a. Generate Monthly Returns Series to generate signal
#############################################################

def monthly_ret(dat):
    dat=pd.merge(dateseries,dat,on='Date',how='left')
        #forward fill missing asset prices for monthly return calculation
    dat=dat.fillna(method='ffill')
        #subset to monthly observations
    dat_m=dat[dat['signalDay']==1].drop(['signalDay','tradeDay'],axis=1)
        #calculate monthly returns for previous month (t-1 to t)
    dat_m=fns.ret(dat_m).reset_index(drop=True)
    
    return dat_m

    # asset returns
assets_m = monthly_ret(assets)
    # ustw_afe returns
ustw_afe_m = monthly_ret(ustw_afe)
    # dxy returns
dxy_m = monthly_ret(dxy)

##########################################
###calculate asset returns in DXY terms###
###(what a foriegn investor buying #######
### with DXY currency basket would make)##
##########################################

#perfect hedge i.e. returns in USD
assets_m_ph=assets_m.copy()

temp=pd.DataFrame((1+assets_m.iloc[:,1:].values)*(1+dxy_m.iloc[:,1:].values)-1,columns=assets_m.columns[1:],index=assets_m.index)
assets_m.iloc[:,1:]=temp


#############################################################
#7b. Construct FX Risk Premia Baskets 
#############################################################
    #get monthly prices
def monthly_px(dat):
    dat=pd.merge(dateseries,dat,on='Date',how='left')
        #forward fill missing asset prices for monthly return calculation
    dat=dat.fillna(method='ffill')
        #subset to monthly observations
    dat_m=dat[dat['signalDay']==1].drop(['signalDay','tradeDay'],axis=1).reset_index(drop=True)
    
    return dat_m


fx_factors_m = monthly_px(fx_factors)
fx_factors_reg_m = monthly_px(fx_factors)

'''
### keep Global versions of carry and Value
    ### high correlation between EM, G-10 and Global factors
    ### restrict factor span to minimize multicollinearity    
fx_factors_reg_m=fx_factors_m.drop(['XS_Carry_EM','PPP_Value_EM','PPP_Value_G10'],axis=1)
fx_factors_reg_m.columns
'''

#COMPUTE INVERSE VOL WEIGHTED FX RP BASKET
    ## 5 year rolling vol with at least 1 year history

fxrp_basket_ivol_m=1/fx_factors_m.iloc[:,1:].rolling(window=60,min_periods=12).std()
fxrp_basket_ivol_m=fxrp_basket_ivol_m.div(fxrp_basket_ivol_m.sum(axis=1),axis=0)

###OUTPUT FXRP iVol Weights
with open(os.path.join(out_dir, 'wt_ivol.pickle'), 'wb') as output:
    pickle.dump(pd.concat([fx_factors_m['Date'],fxrp_basket_ivol_m],axis=1), output)

fxrp_basket_ivol_m=fx_factors_m.iloc[:,1:].mul(fxrp_basket_ivol_m,axis=1)
fxrp_basket_ivol_m=fxrp_basket_ivol_m.sum(axis=1)
fxrp_basket_ivol_m=pd.concat([fx_factors_m['Date'],fxrp_basket_ivol_m],axis=1)
fxrp_basket_ivol_m.rename(columns={0:'fxrp_basket_ivol'},inplace=True)

fxrp_basket_ivol_m['fxrp_basket_ivol'].plot(subplots=True)

with open(os.path.join(out_dir, 'fxrp_basket_ivol_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_ivol_m, output)

fxrp_basket_ivol_m=fns.ret(fxrp_basket_ivol_m)

#DETERMINE OPTIMAL FACTOR WEIGHTS FROM ROLLING REGRESSIONS

    ## 5 year rolling OLS regression: DXY returns ~ FXRP returns 

dxy_fxrp_b_m,ustw_fxrp_p_m=fns.roll_reg(ydat=dxy_m,xdat=fns.ret(fx_factors_reg_m).reset_index(drop=True),name='DXY Curncy',roll_window=60)

#normalize weights
fxrp_basket_reg_m=dxy_fxrp_b_m.drop(['Date','const'],axis=1)
#minim=abs(fxrp_basket_reg_m.min(axis=1).replace(0, np.nan))
#maxim=abs(fxrp_basket_reg_m.max(axis=1).replace(0, np.nan))
#diff=maxim-minim
#count=fxrp_basket_reg_m.count(axis=1)

#take factor value when count=1
#fxrp_basket_reg_m[count<=1]=np.sign(fx_factors_reg_m.iloc[:,1:][count<=1])
#form basket when >1 factor series available
#fxrp_basket_reg_m[count>1]=(fxrp_basket_reg_m[count>1].sub(minim[count>1],axis=0)).div(diff[count>1],axis=0)
    #scale weights to sum = 1
#temp=abs(fxrp_basket_reg_m).sum(axis=1).replace(0, np.nan)    
#fxrp_basket_reg_m = fxrp_basket_reg_m.div(temp,axis=0)  

###OUTPUT FXRP$ Weights
with open(os.path.join(out_dir, 'wt_reg.pickle'), 'wb') as output:
    pickle.dump(pd.concat([fx_factors_m['Date'],fxrp_basket_reg_m],axis=1), output)


#multiply weights by factor values
fxrp_basket_reg_m=fns.ret(fx_factors_reg_m).iloc[:,1:].mul(fxrp_basket_reg_m,axis=1)
fxrp_basket_reg_m=fxrp_basket_reg_m.sum(axis=1)

fxrp_basket_reg_m=pd.concat([pd.Series([100]),1+fxrp_basket_reg_m]).cumprod().iloc[1:]

fxrp_basket_reg_m=pd.concat([fx_factors_reg_m['Date'],fxrp_basket_reg_m],axis=1).replace(0, np.nan)
fxrp_basket_reg_m.rename(columns={0:'fxrp_basket_reg'},inplace=True)

fxrp_basket_reg_m['fxrp_basket_reg'].plot(subplots=True)

#writer=pd.ExcelWriter(os.path.join(out_dir,'test2.xlsx'))
#fx_factors_reg.to_excel(writer,'factors')
#fxrp_basket_reg.to_excel(writer,'betas')
#writer.save()


with open(os.path.join(out_dir, 'fxrp_basket_reg_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_reg_m, output)

fxrp_basket_reg_m=fns.ret(fxrp_basket_reg_m)

fxrp_basket_reg_m.min()


#############################################################
#7c.   Function calls 
#############################################################

##########USTW_AFE##################

### 5-Year Rolling Regression
ustw_hedge_b_m,ustw_hedge_p_m = run_reg(ydat=assets_m,xdat=ustw_afe_m,names=asset_tickers,roll_window=60)

ustw_hedge_b_m = {k: dropna_(v) for k, v in ustw_hedge_b_m.items()}
ustw_hedge_p_m = {k: dropna_(v) for k, v in ustw_hedge_p_m.items()}

with open(os.path.join(out_dir, 'ustw_hedge_b_m.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_b_m, output)
with open(os.path.join(out_dir, 'ustw_hedge_p_m.pickle'), 'wb') as output:
    pickle.dump(ustw_hedge_p_m, output)


##########FXRP_IVOL##################

### 5-Year Rolling Regression
fxrp_ivol_hedge_b_m,fxrp_ivol_hedge_p_m = run_reg(ydat=assets_m,xdat=fxrp_basket_ivol_m,names=asset_tickers,roll_window=60)

fxrp_ivol_hedge_b_m = {k: dropna_(v) for k, v in fxrp_ivol_hedge_b_m.items()}
fxrp_ivol_hedge_p_m = {k: dropna_(v) for k, v in fxrp_ivol_hedge_p_m.items()}

with open(os.path.join(out_dir, 'fxrp_ivol_hedge_b_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_b_m, output)
with open(os.path.join(out_dir, 'fxrp_ivol_hedge_p_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_ivol_hedge_p_m, output)

##########FXRP$_Optimal##################
#Update: regress on individual factors instead of basket and then form asset-specific basket

### 5-Year Rolling OLS Regression
fxrp_reg_hedge_b_m,fxrp_reg_hedge_p_m = run_reg(ydat=assets_m,xdat=fxrp_basket_reg_m,names=asset_tickers,roll_window=60)

#remove NAs from dfs
fxrp_reg_hedge_b_m = {k: dropna_(v) for k, v in fxrp_reg_hedge_b_m.items()}
fxrp_reg_hedge_p_m  = {k: dropna_(v) for k, v in fxrp_reg_hedge_p_m.items()}

with open(os.path.join(out_dir, 'fxrp_reg_hedge_b_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_b_m, output)
with open(os.path.join(out_dir, 'fxrp_reg_hedge_p_m.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_p_m, output)


##########FXRP_Optimal##################
#Update: regress on individual factors instead of basket and then form asset-specific basket

### 5-Year Rolling OLS Regression
fxrp_reg_hedge_b_m,fxrp_reg_hedge_p_m = run_reg(ydat=assets_m,xdat=fns.ret(fx_factors_reg_m),names=asset_tickers,roll_window=60)

#remove NAs from dfs
fxrp_reg_hedge_b_m_fac = {k: dropna_(v) for k, v in fxrp_reg_hedge_b_m.items()}
fxrp_reg_hedge_p_m_fac  = {k: dropna_(v) for k, v in fxrp_reg_hedge_p_m.items()}

with open(os.path.join(out_dir, 'fxrp_reg_hedge_b_m_fac.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_b_m_fac, output)
with open(os.path.join(out_dir, 'fxrp_reg_hedge_p_m_fac.pickle'), 'wb') as output:
    pickle.dump(fxrp_reg_hedge_p_m_fac, output)
    
    
fxrp_basket_reg_m_fac={}
for p in fxrp_reg_hedge_b_m.keys():
    basket=fxrp_reg_hedge_b_m[p].drop(['Date','const'],axis=1)
    #apply cap = 2x
    cap=2 
    basket=basket.clip_upper(cap)
    basket=basket.clip_lower(-cap)

    #multiply weights by factor values
    basket=fns.ret_next(fx_factors_reg_m).iloc[:,1:].mul(basket,axis=1)
    basket=basket.sum(axis=1)
    basket=pd.concat([fxrp_reg_hedge_b_m[p][['Date','const']],basket],axis=1).replace(0, np.nan)
    basket.rename(columns={0:'fxrp_basket_reg'},inplace=True)
    fxrp_basket_reg_m_fac[p]=basket


#remove NAs from dfs
fxrp_basket_reg_m_fac = {k: dropna_(v) for k, v in fxrp_basket_reg_m_fac.items()}

with open(os.path.join(out_dir, 'fxrp_basket_reg_m_fac.pickle'), 'wb') as output:
    pickle.dump(fxrp_basket_reg_m_fac, output)












