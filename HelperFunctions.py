# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:48:59 2019

@author: ZK463GK
"""


import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import Ridge

#############################################################
#4.  Helper Functions
#############################################################

def ret(df):
    '''
    calculate returns
    '''
    temp=df.copy()
    temp.iloc[:,1:]=temp.iloc[:,1:].div(temp.iloc[:,1:].shift(1),axis=1)-1
    return temp

def ret_next(df):
    '''
    calculate realized returns over next period
    '''
    temp=df.copy()
    temp.iloc[:,1:]=temp.iloc[:,1:].shift(-1).div(temp.iloc[:,1:],axis=1)-1
    return temp

def ols_call(Y,X):
    '''
    input dep and indep vars df
    call sm.OLS and calculate HAC standard errors 
    missing values dropped from regression
    return fitted model
    '''
    mod=sm.OLS(Y,X,missing='drop')
    results=mod.fit()
    results=results.get_robustcov_results(cov_type='HAC',maxlags=3)
    results.summary()

    return results


def stack_results(results,name):
    '''
    input sm.ols.fit() (univariate) and var name 
    output stacked df with parameter estimates including pvalues and model R2
    '''
    param=results.params.copy()
    param=pd.Series(param)
    formatter_param= lambda x: round(x*1E4,3) ###scale parameters to bps
    param=pd.DataFrame(param.apply(formatter_param))
    param.columns=[name]
            
    PV=pd.Series(results.pvalues.copy())
    formatter_PV = lambda x: round(x,3)
    PV=pd.DataFrame(PV.apply(formatter_PV))
    PV.columns=['pval']
            
    Rsq=pd.DataFrame([np.nan,round(results.rsquared*100,3)]) ###scale in %
    Rsq.columns=['Rsq']
    Rsq.index=param.index
            
    stacked=pd.concat([pd.concat([param,PV],axis=1),Rsq],axis=1)
    stacked=stacked.T
    
    return stacked


def roll_reg(ydat,xdat,name,roll_window):
    '''
    OLS regression function 
    input: dep & indep vars, Y variable name, window (days) for rolling regressions (window=ydat.shape[0] for full period)
    output: two dataframes with parameters and corresponding pvalues 
    '''
    ### Rolling Regressions
    ### SET PARAMS
            
    # Regression window size
    n = roll_window # days / year
    # Index counter for rolling regressions
    t_start = n
    t_end = xdat.shape[0] 
                                   
    X = xdat.drop(['Date'],axis=1)
    X = sm.add_constant(X)
    X = X.replace(np.inf, np.nan)
    Y = pd.merge(xdat,ydat,on='Date',how='left')[name]

    ols_roll   = pd.DataFrame(index=list(range(t_end)),columns=X.columns)
    ols_roll_p = pd.DataFrame(index=list(range(t_end)),columns=X.columns)    
            
    for t in range(t_start,t_end+1):
            # require all observations in window to include an X var in the regression
        X_roll = X.iloc[t-n:t,:].dropna(axis=1,thresh=n)
        Y_roll = Y.iloc[t-n:t]
        try:
            rollreg = ols_call(Y_roll,X_roll)
            ols_roll.loc[t-1,list(X_roll.columns)]   = rollreg.params
            ols_roll_p.loc[t-1,list(X_roll.columns)] = rollreg.pvalues
        except:
            continue
            
    ols_roll=pd.concat([xdat['Date'],ols_roll],axis=1)
    ols_roll_p=pd.concat([xdat['Date'],ols_roll_p],axis=1)
        
    return ols_roll, ols_roll_p



def roll_reg_ridge(ydat,xdat,name,roll_window):
    '''
    Ridge regression function 
    input: dep & indep vars, Y variable name, window (days) for rolling regressions (window=ydat.shape[0] for full period)
    output: two dataframes with parameters and corresponding pvalues 
    '''
    ### Rolling Regressions
    ### SET PARAMS
            
    # Regression window size
    n = roll_window # days / year
    # Index counter for rolling regressions
    t_start = n
    t_end = xdat.shape[0] 
                                   
    X = xdat.drop(['Date'],axis=1)
    X = X.replace(np.inf, np.nan)
    Y = pd.merge(xdat,ydat,on='Date',how='left')[name]

    rid_roll   = pd.DataFrame(index=list(range(t_end)),columns=X.columns)
            
    for t in range(t_start,t_end+1):
            # require all observations in window to include an X var in the regression
        X_roll = X.iloc[t-n:t,:].dropna(axis=1,thresh=n)
        Y_roll = Y.iloc[t-n:t]
        try:
            clf = Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, random_state=None, solver='auto', tol=0.001)
            rollreg = clf.fit(X_roll,Y_roll)
            rid_roll.loc[t-1,list(X_roll.columns)]   = rollreg.coef_
        except:
            continue
            
    rid_roll=pd.concat([xdat['Date'],rid_roll],axis=1)
        
    return rid_roll


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
                rollreg = ols_call(Y_roll,X_roll)
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
