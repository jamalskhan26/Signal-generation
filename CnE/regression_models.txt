# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2020

@author: Jamal Khan
"""

import os
wd = os.getcwd()
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
from meta_models import MetaModel
os.chdir(wd)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

class regression_models(MetaModel):

    def __init__(self, X, Y, freq, window, datevars_x, datevars_y, mergevar,
                 intercept = True, demeaned = False, ncomp_pca = None, ncomp_pls = None,
                 ols = True, lasso = True, ridge = True, corr_f = True, corr_d = True, 
                 pca = True, pls = True, rf = True, gb = True, lsvm = True, psvm = True, rsvm = True):
        """
        Initialize the class and data (X and Y dataframes & storage frames for rolling coefficients, predictions, and tstats)
    
        X: exogenous variable(s)
        Y: endogeous variable(s)
        (note: NAs not handled explicitly so should be dealt with before passing in)
        window: interval for rolling regressions 
        datevars_x and datevars_y: names of date variable(s) in X and Y passed in an array (e.g. [signalDate, tradeDate]) to be dropped from regressions but used for aligning time series
        mergevar: specify how to merge X and Y variables (typically mergevar = date)
        intercept: specify whether to include intercept in regression 
        demean: specify whether to demean the series
        model universe: 
            OLS (ols)
            Ridge (ridge)
            Lasso (lasso)
            Correlation filter: fixed threshold (corr_f)
            Correlation filter: dynamic threshold (corr_d)
            PCA (pca) (first sqrt(X vars) components used in regression -- can be specified alternatively in _init_ as ncomp_pca)
            PLS (pls) (first ncomp/2 dimensions used in regression -- can be specified alternatively in _init_ as ncomp_pls)
            Random forest (rf)
            Gradient boost (gb)
        """
        self.datevars_x = datevars_x
        self.datevars_y = datevars_y
        self.mergevar = mergevar
        self.dates = X[self.mergevar].reset_index(drop = True)
        self.X = X.drop(datevars_x, axis = 1).reset_index(drop = True).astype(float)
        self.Y = pd.merge(X, Y, on = self.mergevar, how = 'left').drop((datevars_y + list(X.columns)), axis = 1).reset_index(drop = True).astype(float)
        self.freq = freq
        self.window = window
        self.yvars = self.Y.columns
        self._intercept = intercept
        self._demeaned = demeaned
        self.ncomp_pca = ncomp_pca
        self.ncomp_pls = ncomp_pls

        if (self.freq == 'daily'):
            self.N = 252
        elif (self.freq == 'weekly'):
            self.N = 52
        elif (self.freq == 'monthly'):
            self.N = 12
        elif (self.freq == 'quarterly'):
            self.N = 4
            
        self._nobs = len(self.Y)
        self.X = self.X.replace(np.inf, np.nan).replace(np.nan, 0)
        self.Y = self.Y.replace(np.inf, np.nan).replace(np.nan, 0)     

        self.actual_returns = self.Y
        self.big_moves = abs(self.actual_returns)
        self.big_moves[self.big_moves > 0.03] =  1
        self.big_moves[self.big_moves <= 0.03] = 0
       
        if self._demeaned:
            self.X = self.X.sub(self.X.mean(axis = 0), axis = 1)
            self.Y = self.Y.sub(self.Y.mean(axis = 0), axis = 1)
        elif self._intercept:
            self.X = sm.add_constant(self.X)
                    
        self.ols =    ols                    
        self.lasso =  lasso
        self.ridge =  ridge
        self.corr_f = corr_f
        self.corr_d = corr_d
        self.pca =    pca
        self.pls =    pls
        self.rf =     rf
        self.gb =     gb
        self.lsvm =   lsvm
        self.psvm =   psvm
        self.rsvm =   rsvm
                            
        #1
        self.beta_ols, self.tstats_ols, self.score_ols, self.predict_ols = ({} for _ in range(4))
        #2
        self.beta_lasso, self.score_lasso, self.predict_lasso = ({} for _ in range(3))
        #3
        self.beta_ridge, self.score_ridge, self.predict_ridge = ({} for _ in range(3))
        #4
        self.beta_corr_f, self.tstats_corr_f, self.score_corr_f, self.predict_corr_f = ({} for _ in range(4))
        #5
        self.beta_corr_d, self.tstats_corr_d, self.score_corr_d, self.predict_corr_d = ({} for _ in range(4))
        #6
        self.beta_pcr, self.tstats_pcr, self.score_pcr, self.predict_pcr = ({} for _ in range(4))
        #7
        self.beta_pls, self.score_pls, self.predict_pls = ({} for _ in range(3))
        #8
        self.beta_rf, self.score_rf, self.predict_rf = ({} for _ in range(3))
        #9
        self.beta_gb, self.score_gb, self.predict_gb = ({} for _ in range(3))
        #10
        self.beta_lsvm, self.score_lsvm, self.predict_lsvm = ({} for _ in range(3))
        #11
        self.score_psvm, self.predict_psvm = ({} for _ in range(2))
        #12
        self.score_rsvm, self.predict_rsvm = ({} for _ in range(2))

        #parameters for PCA & PLS
        if self.ncomp_pca == None:
            self.ncomp_pca = max(math.ceil(len(self.X.columns)**0.5), 1)
        if self.ncomp_pls == None:
            self.ncomp_pls = max(math.ceil(self.ncomp_pca/2), 1)

        #scale X for PCA and PLS
        if self._intercept:
            self.X_noconst = self.X.drop(['const'], axis = 1)
        else:
            self.X_noconst = self.X

        #threshold for fixed correlation filter
        self.corr_thresh_f = 0.2
        #threshold for dynamic correlation filter (computed in method)
        #Criterion: if abs(rolling correlation) > corr_thresh_f AND
                    #greater than 1 SD decrease in abs(rolling correlation) => decoupling => exclude 
        self.roll_corr, self.roll_corr_std, self.roll_corr_d = ({} for _ in range(3))
        
        for y in self.yvars:
            self.roll_corr[y] =     self.X_noconst.rolling(self.window).corr(self.Y[y])
            self.roll_corr_std[y] = self.roll_corr[y].rolling(self.window).std()
            self.roll_corr_d[y] =   self.roll_corr[y].diff(axis = 0)
    
        
    def fit(self):
        '''
        run rolling regressions -- 
            currently set up as nested loop over X (rolling) and Y
        for each model, set values for rolling betas, predictions and tstats 
        '''
        for y in self.yvars:
            
            if self.ols:
                self.beta_ols[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.tstats_ols[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.score_ols[y] =      pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_ols[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.lasso:    
                self.beta_lasso[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.score_lasso[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_lasso[y] =  pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.ridge:
                self.beta_ridge[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.score_ridge[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_ridge[y] =  pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.corr_f:
                self.beta_corr_f[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.tstats_corr_f[y] =  pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns) 
                self.score_corr_f[y] =   pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_corr_f[y] = pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.corr_d:
                self.beta_corr_d[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns)
                self.tstats_corr_d[y] =  pd.DataFrame(index=list(range(self._nobs)),columns=self.X.columns) 
                self.score_corr_d[y] =   pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_corr_d[y] = pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.pca:
                self.beta_pcr[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=range(len(self.X_noconst.columns)))
                self.tstats_pcr[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=range(len(self.X_noconst.columns)))
                self.score_pcr[y] =      pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_pcr[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.pls:
                self.beta_pls[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=range(len(self.X_noconst.columns)))
                self.score_pls[y] =      pd.DataFrame(index=list(range(self._nobs)),columns=[y]) 
                self.predict_pls[y] =    pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.rf:
                self.beta_rf[y] =        pd.DataFrame(index=list(range(self._nobs)),columns=self.X_noconst.columns)
                self.score_rf[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_rf[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.gb:
                self.beta_gb[y] =        pd.DataFrame(index=list(range(self._nobs)),columns=self.X_noconst.columns)
                self.score_gb[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_gb[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.lsvm:
                self.beta_lsvm[y] =        pd.DataFrame(index=list(range(self._nobs)),columns=range(len(self.X_noconst.columns)))
                self.score_lsvm[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_lsvm[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.psvm:
                self.score_psvm[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_psvm[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=[y])
            if self.rsvm:
                self.score_rsvm[y] =       pd.DataFrame(index=list(range(self._nobs)),columns=[y])
                self.predict_rsvm[y] =     pd.DataFrame(index=list(range(self._nobs)),columns=[y])

        #######################

            for t in range(self.window, self._nobs):
                    # require all observations in window to include an X var in the regression
                X_roll = self.X.iloc[t - self.window : t, :].dropna(axis = 1, thresh = self.window)
                Y_roll = self.Y.loc[t - self.window : t-1, y]

                X_roll_noconst = self.X_noconst.iloc[t - self.window : t, :].dropna(axis = 1, thresh = self.window)
                
                pca=PCA()
                self.X_test_scale = pd.DataFrame(scale(self.X_noconst.iloc[: t + 1, :]))
                self.X_test_reduced = pd.DataFrame(pca.fit_transform(self.X_test_scale)).iloc[:,:(self.ncomp_pca+1)]
                
                #1
                if self.ols:
                    try:
                        rollreg = self.ols_call(X_roll, Y_roll)
                        
                        self.beta_ols[y].loc[t-1, list(X_roll.columns)]   = rollreg.params
                        self.tstats_ols[y].loc[t-1, list(X_roll.columns)] = rollreg.tvalues
                        self.score_ols[y].loc[t-1] = rollreg.rsquared
                        self.predict_ols[y].loc[t]  = rollreg.predict(list(self.X.loc[t, list(X_roll.columns)]))
                    except:
                        continue         
                #2
                if self.lasso:
                    try:
                        rollreg = self.lasso_call(X_roll, Y_roll)
                        
                        self.beta_lasso[y].loc[t-1, list(X_roll.columns)]   = rollreg.coef_
                        self.score_lasso[y].loc[t-1] = rollreg.score(X_roll, Y_roll)
                        self.predict_lasso[y].loc[t]  = rollreg.predict([list(self.X.loc[t, list(X_roll.columns)])])
                    except:
                        continue                         

                #3
                if self.ridge:
                    try:
                        rollreg = self.ridge_call(X_roll, Y_roll)
                        
                        self.beta_ridge[y].loc[t-1, list(X_roll.columns)]   = rollreg.coef_
                        self.score_ridge[y].loc[t-1] = rollreg.score(X_roll, Y_roll)
                        self.predict_ridge[y].loc[t]  = rollreg.predict([list(self.X.loc[t, list(X_roll.columns)])])
                    except:
                        continue         
                
                #4
                if self.corr_f:
                    try:
                        rollreg = self.corr_f_call(X_roll, Y_roll, self.roll_corr[y].iloc[t-1, :])
                        
                        vars_included = list(rollreg.params.index)
                        self.beta_corr_f[y].loc[t-1, vars_included]   = rollreg.params
                        self.tstats_corr_f[y].loc[t-1, vars_included] = rollreg.tvalues
                        self.score_corr_f[y].loc[t-1] = rollreg.rsquared_adj
                        self.predict_corr_f[y].loc[t]  = rollreg.predict(list(self.X.loc[t, vars_included]))
                    except:
                        continue         
                
                #5
                if self.corr_d:
                    try:
                        rollreg = self.corr_d_call(X_roll, Y_roll, self.roll_corr[y].iloc[t-1, :], self.roll_corr_d[y].iloc[t-1, :], self.roll_corr_std[y].iloc[t-1, :])
                        
                        vars_included = list(rollreg.params.index)
                        self.beta_corr_d[y].loc[t-1, vars_included]   = rollreg.params
                        self.tstats_corr_d[y].loc[t-1, vars_included] = rollreg.tvalues
                        self.score_corr_d[y].loc[t-1] = rollreg.rsquared_adj
                        self.predict_corr_d[y].loc[t]  = rollreg.predict(list(self.X.loc[t, vars_included]))
                    except:
                        continue         
                
                #6
                if self.pca:
                    try:
                        rollreg, X_reduced, X_exp_var = self.pcr_call(X_roll_noconst, Y_roll)
                        
                        self.beta_pcr[y].loc[t-1, list(X_reduced.columns)]   = rollreg.coef_
                        self.tstats_pcr[y].loc[t-1, list(X_reduced.columns)] = X_exp_var
                        self.score_pcr[y].loc[t-1] = rollreg.score(X_reduced, Y_roll)
                        self.predict_pcr[y].loc[t]  = rollreg.predict([list(self.X_test_reduced.iloc[t,:])])
                    except:
                        continue         
                
                #7
                if self.pls:
                    try:
                        rollreg, X_reduced = self.pls_call(X_roll_noconst, Y_roll)
                        
                        self.beta_pls[y].loc[t-1, list(X_reduced.columns)]   = rollreg.coef_.reshape(1,-1)[0]
                        self.score_pls[y].loc[t-1] = rollreg.score(X_reduced, Y_roll)
                        self.predict_pls[y].loc[t]  = rollreg.predict([list(self.X_test_scale.loc[t, list(X_reduced.columns)])])
                    except:
                        continue         
                
                #8
                if self.rf:
                    try:
                        rollreg = self.rf_call(X_roll_noconst, Y_roll)
                        
                        self.beta_rf[y].loc[t-1, list(X_roll_noconst.columns)]   = rollreg.feature_importances_
                        self.score_rf[y].loc[t-1] = rollreg.score(X_roll_noconst, Y_roll)
                        self.predict_rf[y].loc[t]  = rollreg.predict(pd.DataFrame(list(self.X.loc[t, list(X_roll_noconst.columns)])).T)
                    except:
                        continue         

                #9
                if self.gb:
                    try:
                        rollreg = self.gb_call(X_roll_noconst, Y_roll)
                        
                        self.beta_gb[y].loc[t-1, list(X_roll_noconst.columns)]   = rollreg.feature_importances_
                        self.score_gb[y].loc[t-1] = rollreg.score(X_roll_noconst, Y_roll)
                        self.predict_gb[y].loc[t]  = rollreg.predict(pd.DataFrame(list(self.X.loc[t, list(X_roll_noconst.columns)])).T)
                    except:
                        continue         

                #10
                if self.lsvm:
                    try:
                        rollreg, X_reduced, y_reduced = self.svm_call(X_roll_noconst, Y_roll, 'linear')
                        
                        self.beta_lsvm[y].loc[t-1, list(X_reduced.columns)]   = rollreg.coef_.reshape(1,-1)[0]
                        self.score_lsvm[y].loc[t-1] = rollreg.score(X_reduced, y_reduced)
                        self.predict_lsvm[y].loc[t]  = rollreg.predict([list(self.X_noconst.loc[t, list(X_roll_noconst.columns)])])
                    except:
                        continue         

                #11
                if self.psvm:
                    try:
                        rollreg, X_reduced, y_reduced = self.svm_call(X_roll_noconst, Y_roll, 'poly')
                        
                        self.score_psvm[y].loc[t-1] = rollreg.score(X_reduced, y_reduced)
                        self.predict_psvm[y].loc[t]  = rollreg.predict([list(self.X_noconst.loc[t, list(X_roll_noconst.columns)])])
                    except:
                        continue         

                #12
                if self.rsvm:
                    try:
                        rollreg, X_reduced, y_reduced = self.svm_call(X_roll_noconst, Y_roll, 'rbf')
                        
                        self.score_rsvm[y].loc[t-1] = rollreg.score(X_reduced, y_reduced)
                        self.predict_rsvm[y].loc[t]  = rollreg.predict([list(self.X_noconst.loc[t, list(X_roll_noconst.columns)])])
                    except:
                        continue         
                
        #######################
        
            if self.ols:
                self.beta_ols[y] =       pd.concat([self.dates, self.beta_ols[y]], axis = 1)
                self.tstats_ols[y] =     pd.concat([self.dates, self.tstats_ols[y]], axis = 1)                
                self.score_ols[y] =      pd.concat([self.dates, self.score_ols[y]], axis = 1)                
                self.predict_ols[y] =    pd.concat([self.dates, self.predict_ols[y]], axis = 1)                
            if self.lasso:
                self.beta_lasso[y] =     pd.concat([self.dates, self.beta_lasso[y]], axis = 1)
                self.score_lasso[y] =    pd.concat([self.dates, self.score_lasso[y]], axis = 1)                
                self.predict_lasso[y] =  pd.concat([self.dates, self.predict_lasso[y]], axis = 1)                
            if self.ridge:
                self.beta_ridge[y] =     pd.concat([self.dates, self.beta_ridge[y]], axis = 1)
                self.score_ridge[y] =    pd.concat([self.dates, self.score_ridge[y]], axis = 1)                
                self.predict_ridge[y] =  pd.concat([self.dates, self.predict_ridge[y]], axis = 1)                
            if self.corr_f:
                self.beta_corr_f[y] =    pd.concat([self.dates, self.beta_corr_f[y]], axis = 1)
                self.tstats_corr_f[y] =  pd.concat([self.dates, self.tstats_corr_f[y]], axis = 1)                
                self.score_corr_f[y] =   pd.concat([self.dates, self.score_corr_f[y]], axis = 1)                
                self.predict_corr_f[y] = pd.concat([self.dates, self.predict_corr_f[y]], axis = 1)                
            if self.corr_d:
                self.beta_corr_d[y] =    pd.concat([self.dates, self.beta_corr_d[y]], axis = 1)
                self.tstats_corr_d[y] =  pd.concat([self.dates, self.tstats_corr_d[y]], axis = 1)                
                self.score_corr_d[y] =   pd.concat([self.dates, self.score_corr_d[y]], axis = 1)                
                self.predict_corr_d[y] = pd.concat([self.dates, self.predict_corr_d[y]], axis = 1)                
            if self.pca:
                self.beta_pcr[y] =       pd.concat([self.dates, self.beta_pcr[y]], axis = 1)
                self.tstats_pcr[y] =     pd.concat([self.dates, self.tstats_pcr[y]], axis = 1)                
                self.score_pcr[y] =      pd.concat([self.dates, self.score_pcr[y]], axis = 1)                
                self.predict_pcr[y] =    pd.concat([self.dates, self.predict_pcr[y]], axis = 1)                
            if self.pls:
                self.beta_pls[y] =       pd.concat([self.dates, self.beta_pls[y]], axis = 1)
                self.score_pls[y] =      pd.concat([self.dates, self.score_pls[y]], axis = 1)                
                self.predict_pls[y] =    pd.concat([self.dates, self.predict_pls[y]], axis = 1)                
            if self.rf:
                self.beta_rf[y] =        pd.concat([self.dates, self.beta_rf[y]], axis = 1)
                self.score_rf[y] =       pd.concat([self.dates, self.score_rf[y]], axis = 1)                
                self.predict_rf[y] =     pd.concat([self.dates, self.predict_rf[y]], axis = 1)                
            if self.gb:
                self.beta_gb[y] =        pd.concat([self.dates, self.beta_gb[y]], axis = 1)
                self.score_gb[y] =       pd.concat([self.dates, self.score_gb[y]], axis = 1)                
                self.predict_gb[y] =     pd.concat([self.dates, self.predict_gb[y]], axis = 1)                
            if self.lsvm:
                self.beta_lsvm[y] =        pd.concat([self.dates, self.beta_lsvm[y]], axis = 1)
                self.score_lsvm[y] =       pd.concat([self.dates, self.score_lsvm[y]], axis = 1)                
                self.predict_lsvm[y] =     pd.concat([self.dates, self.predict_lsvm[y]], axis = 1)                
            if self.psvm:
                self.score_psvm[y] =       pd.concat([self.dates, self.score_psvm[y]], axis = 1)                
                self.predict_psvm[y] =     pd.concat([self.dates, self.predict_psvm[y]], axis = 1)                
            if self.rsvm:
                self.score_rsvm[y] =       pd.concat([self.dates, self.score_rsvm[y]], axis = 1)                
                self.predict_rsvm[y] =     pd.concat([self.dates, self.predict_rsvm[y]], axis = 1)                
                
        ## if new models added update model inventory and add model in fit(), output()
        self.models = ['Long_Only', 'OLS', 'Lasso', 'Ridge', 'Correlation_Filter_F', 'Correlation_Filter_D', 'PCR', 'PLS', 'Random_Forest', 'Gradient_Boost', 'SVM (L)', 'SVM (P)', 'SVM (R)']
        self.predictions = [{x : abs(self.actual_returns)[[x]] for x in self.actual_returns.columns}, 
                             self.predict_ols, self.predict_lasso, self.predict_ridge, self.predict_corr_f, 
                             self.predict_corr_d, self.predict_pcr, self.predict_pls, self.predict_rf, self.predict_gb, 
                             self.predict_lsvm, self.predict_psvm, self.predict_rsvm]
        self.scores = [{x : abs(self.actual_returns)[[x]] for x in self.actual_returns.columns}, 
                        self.score_ols, self.score_lasso, self.score_ridge, self.score_corr_f, 
                        self.score_corr_d, self.score_pcr, self.score_pls, self.score_rf, self.score_gb, 
                        self.score_lsvm, self.score_psvm, self.score_rsvm]
            
    def ensemble(self):
        from copy import deepcopy
        if bool(set(['LR', 'RR', 'SVD', 'DT','Ensemble', 'Ensemble2', 'Ensemble3']).intersection(set(self.models))):           
            self.models = [x for x in self.models if x not in set(['LR', 'RR', 'SVD', 'DT', 'SVM', 'Ensemble', 'Ensemble2', 'Ensemble3']).intersection(set(self.models))]
            self.predictions = [{x : abs(self.actual_returns)[[x]] for x in self.actual_returns.columns}, 
                                 self.predict_ols, self.predict_lasso, self.predict_ridge, self.predict_corr_f, 
                                 self.predict_corr_d, self.predict_pcr, self.predict_pls, self.predict_rf, self.predict_gb,
                                 self.predict_lsvm, self.predict_psvm, self.predict_rsvm]

        # model family ensembles
        e_ols = deepcopy(self.predictions[2])
        e_rr = deepcopy(self.predictions[2])
        e_svd = deepcopy(self.predictions[2])
        e_dtr = deepcopy(self.predictions[2])
        e_svm = deepcopy(self.predictions[2])

        for p in self.predictions[0].keys():
            # OLS based methods -- OLS, correlation filters (dynamic and static)
            e_ols[p][p] = (self.predictions[1][p][p] + self.predictions[4][p][p] + self.predictions[5][p][p])/3

            # Regularized regressions -- (lasso and ridge)
            e_rr[p][p] = (self.predictions[2][p][p] + self.predictions[3][p][p])/2
            
            # SVD -- singular value decomposition (PCR and PLS)
            e_svd[p][p] = (self.predictions[6][p][p] + self.predictions[7][p][p])/2

            # DTR -- Decision Trees (RF and GB)
            e_dtr[p][p] = (self.predictions[8][p][p] + self.predictions[9][p][p])/2

            # SVM -- Support Vector Machines (polynomial kernel)
            e_svm[p][p] = (self.predictions[10][p][p] + self.predictions[11][p][p])/2

        prediction_sign = []
        temp = deepcopy([e_ols,e_rr,e_svd,e_dtr,e_svm])    
        for p in temp:
            for s in p.values():
                r = s.iloc[:, 1:]
                r[r > 0] =  1
                r[r <= 0] = 0
                s.iloc[:, 1:] = r
            prediction_sign.append(p)
        
        ensemble = deepcopy(self.predictions[2])
        ensemble2 = deepcopy(self.predictions[2])
        ensemble3 = deepcopy(self.predictions[2])
        comb_score = deepcopy(self.scores[2])    

      
        for p in prediction_sign[0].keys():
            # Ensemble 1 -- modal/max voting
            x = prediction_sign[0][p][p] + prediction_sign[1][p][p] + prediction_sign[2][p][p] + prediction_sign[3][p][p] + prediction_sign[4][p][p] 
            y = self.scores[2][p][p] + self.scores[3][p][p] + self.scores[4][p][p] + self.scores[5][p][p] + self.scores[6][p][p] + self.scores[7][p][p] + self.scores[8][p][p] + self.scores[9][p][p] + self.scores[11][p][p] 
            x[x < 3] = -1
            x[x >= 3] = 1            
            ensemble[p][p] = x
            comb_score[p][p] = y
            
            # Ensemble 2 -- simple average
            ensemble2[p][p] = 0.15*e_ols[p][p].add(0.10*e_svm[p][p], fill_value=0).add(0.25*e_rr[p][p], fill_value=0).add(0.25*e_svd[p][p], fill_value=0).add(0.25*e_dtr[p][p], fill_value=0) 
            
            # Ensemble 3 -- weighted average (by rolling modified residuals)
            r_lasso = (self.actual_returns[p] - self.predictions[2][p][p]) * np.sign(self.predictions[2][p][p].replace(np.nan,0))*-1
            r_ridge = (self.actual_returns[p] - self.predictions[3][p][p]) * np.sign(self.predictions[3][p][p].replace(np.nan,0))*-1
            r_corrf = (self.actual_returns[p] - self.predictions[4][p][p]) * np.sign(self.predictions[4][p][p].replace(np.nan,0))*-1
            r_corrd = (self.actual_returns[p] - self.predictions[5][p][p]) * np.sign(self.predictions[5][p][p].replace(np.nan,0))*-1
            r_pcr =   (self.actual_returns[p] - self.predictions[6][p][p]) * np.sign(self.predictions[6][p][p].replace(np.nan,0))*-1
            r_pls =   (self.actual_returns[p] - self.predictions[7][p][p]) * np.sign(self.predictions[7][p][p].replace(np.nan,0))*-1
            r_rf =    (self.actual_returns[p] - self.predictions[8][p][p]) * np.sign(self.predictions[8][p][p].replace(np.nan,0))*-1
            r_gb =    (self.actual_returns[p] - self.predictions[9][p][p]) * np.sign(self.predictions[9][p][p].replace(np.nan,0))*-1
            r_lsvm =  (self.actual_returns[p] - self.predictions[10][p][p]) * np.sign(self.predictions[10][p][p].replace(np.nan,0))*-1
            r_psvm =  (self.actual_returns[p] - self.predictions[11][p][p]) * np.sign(self.predictions[11][p][p].replace(np.nan,0))*-1
                ## calculate weights as standardized residuals
            w_lasso = (1/((r_lasso.rolling(window = self.N).mean()) / r_lasso.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_ridge = (1/((r_ridge.rolling(window = self.N).mean()) / r_ridge.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_corrf = (1/((r_corrf.rolling(window = self.N).mean()) / r_corrf.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_corrd = (1/((r_corrd.rolling(window = self.N).mean()) / r_corrd.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_pcr =   (1/((r_pcr.rolling(window = self.N).mean()) / r_pcr.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_pls =   (1/((r_pls.rolling(window = self.N).mean()) / r_pls.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_rf =    (1/((r_rf.rolling(window = self.N).mean()) / r_rf.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_gb =    (1/((r_gb.rolling(window = self.N).mean()) / r_gb.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_lsvm =  (1/((r_lsvm.rolling(window = self.N).mean()) / r_lsvm.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
            w_psvm =  (1/((r_psvm.rolling(window = self.N).mean()) / r_psvm.rolling(window = self.N).std())).replace([np.inf, -np.inf], np.nan).replace(np.nan,0)
                ## normalize weights between 0 and 1
            w_min = pd.concat([w_lasso, w_ridge, w_corrf, w_corrd, w_pcr, w_pls, w_rf, w_gb, w_lsvm, w_psvm], axis = 1).min(axis = 1)
            w_max = pd.concat([w_lasso, w_ridge, w_corrf, w_corrd, w_pcr, w_pls, w_rf, w_gb, w_lsvm, w_psvm], axis = 1).max(axis = 1)
            w_range = w_max - w_min
            w_lasso = ((w_lasso - w_min) / w_range).shift(1)
            w_ridge = ((w_ridge - w_min) / w_range).shift(1)
            w_corrf = ((w_corrf - w_min) / w_range).shift(1)
            w_corrd = ((w_corrd - w_min) / w_range).shift(1)
            w_pcr =   ((w_pcr - w_min) / w_range).shift(1)
            w_pls =   ((w_pls - w_min) / w_range).shift(1)
            w_rf =    ((w_rf - w_min) / w_range).shift(1)
            w_gb =    ((w_gb - w_min) / w_range).shift(1)
            w_lsvm =  ((w_lsvm - w_min) / w_range).shift(1)
            w_psvm =  ((w_psvm - w_min) / w_range).shift(1)
            ensemble3[p][p] = w_lasso*self.predictions[2][p][p].add(w_ridge*self.predictions[3][p][p], fill_value=0).add(w_corrf*self.predictions[4][p][p], fill_value=0).add(w_corrd*self.predictions[5][p][p], fill_value=0).add(w_pcr*self.predictions[6][p][p], fill_value=0).add(w_pls*self.predictions[7][p][p], fill_value=0).add(w_rf*self.predictions[8][p][p], fill_value=0).add(w_gb*self.predictions[9][p][p], fill_value=0).add(w_lsvm*self.predictions[10][p][p], fill_value=0).add(w_psvm*self.predictions[11][p][p], fill_value=0)  
                                            
        self.models.extend(['LR', 'RR', 'SVD', 'DT', 'SVM', 'Ensemble', 'Ensemble2', 'Ensemble3'])
        self.predictions.extend([e_ols, e_rr, e_svd, e_dtr, e_svm, ensemble, ensemble2, ensemble3])
        self.scores.extend([comb_score]*8)


    def performance_stats(self, signal_weighting = False):
        '''
        calculate (1) hit ratios (all and big moves), 
                  (2) annualized returns, 
                  (3) info ratios
        over (a) the full period, 
             (b) broken out by year
        '''
        import datetime as dt

        hit_ratio =           {}
        ann_returns =         {}
        info_ratio =          {}
        hit_ratio_by_year =   {}
        big_hit_by_year =     {}
        ann_returns_by_year = {}
        info_ratio_by_year =  {}
        signal_wt = {}

        for y in self.yvars:
            hit_ratio[y] =           pd.DataFrame()
            ann_returns[y] =         pd.DataFrame()
            info_ratio[y] =          pd.DataFrame()
            hit_ratio_by_year[y] =   pd.DataFrame(index = np.unique(self.dates.apply(lambda x: x.dt.year)))
            big_hit_by_year[y] =     pd.DataFrame(index = np.unique(self.dates.apply(lambda x: x.dt.year)))
            ann_returns_by_year[y] = pd.DataFrame(index = np.unique(self.dates.apply(lambda x: x.dt.year)))
            info_ratio_by_year[y] =  pd.DataFrame(index = np.unique(self.dates.apply(lambda x: x.dt.year)))
            signal_wt[y] = {}
            
            for p, q, r in zip(self.predictions, self.models, self.scores):
                try:
                    if signal_weighting: # trade only if prediction magnitude > 3%
                        wt = p[y][y].copy()
                        wt[abs(wt) >= 0.03] = 1
                        wt[abs(wt) < 0.03] =  0
#                        temp = self.actual_returns.loc[:,y].mul(p[y][y].iloc[self.actual_returns.index].values, axis = 0)                  
#                        temp[temp > 0] =  1
#                        temp[temp <= 0] = 0
#                        hr_df_rolling = temp.shift(1).rolling(window = self.N).mean()
#                        signal_1 = hr_df_rolling/0.5
#                        signal_2 = abs(p[y][y]) / abs(p[y][y]).shift(1).rolling(window = self.N).mean()
#                        signal_3 = r[y][y].shift(1) - r[y][y].shift(1).rolling(window = self.N).mean()
#                        wt = 0.25*signal_1.add(0.5*signal_2, fill_value = 0).add(0.25*signal_3, fill_value = 0)
                        signal_wt[y][q] = wt
                    else:
                        wt = 1

                    ## a. full period
                    # 1. hit ratios
                    if q != 'Long_Only':
                        hr_df = self.actual_returns.loc[:,y].mul(((p[y][y]*wt).iloc[self.actual_returns.index]).values, axis = 0)                  
                    else:
                        hr_df = self.actual_returns.loc[:,y].mul((p[y][y].iloc[self.actual_returns.index]).values, axis = 0)                  
                    hr_df[hr_df == 0] = np.nan
                    hr_df[hr_df > 0] =  1
                    hr_df[hr_df < 0] =  0            
                    big_hr_df = self.big_moves[[y]].mul(hr_df.values, axis = 0)
                    hr_yearly = pd.concat([self.dates, hr_df], axis = 1)
                    big_hr_yearly = pd.concat([self.dates, big_hr_df], axis = 1)
        
                    hr_df = hr_df.sum()/hr_df.count()
                    big_hr_df = big_hr_df.dropna().sum() / self.big_moves[[y]].sum()
                    hit_ratio[y].loc[q, 'Hit_Ratio'] = hr_df
                    hit_ratio[y].loc[q, 'Hit_Ratio_Big'] = big_hr_df.values

                    # 2. ann. returns
                    if q != 'Long_Only':
                        ret = self.actual_returns.loc[:,y].mul(np.sign(((p[y][y]*wt).iloc[self.actual_returns.index]).replace(np.nan,0)).values, axis = 0)
                    else:
                        ret = self.actual_returns.loc[:,y].mul(np.sign((p[y][y].iloc[self.actual_returns.index]).replace(np.nan,0)).values, axis = 0)
                    
                    ann_returns[y].loc[q, 'ann_ret'] = ret.mean(axis = 0) * self.N
                    
                    # 3. info ratios
                    info_ratio[y].loc[q, 'info_ratio'] = (ret.mean(axis = 0) / ret.std(axis = 0)) * (self.N**0.5)

                    ## b. by year
                    # 1. hit ratios
                    hr_yearly['year'] = hr_yearly[self.mergevar].apply(lambda x: x.dt.year)
                    hr_yearly = hr_yearly.drop(self.mergevar, axis = 1)
                    hr_yearly = hr_yearly.groupby('year')
                    hit_ratio_by_year[y].loc[:, q] = (hr_yearly.sum()/hr_yearly.count())[0]

                    big_moves_yearly = self.big_moves[[y]]
                    big_moves_yearly['year'] = big_hr_yearly[self.mergevar].apply(lambda x: x.dt.year) 
                    big_moves_yearly = big_moves_yearly.groupby('year')

                    big_hr_yearly['year'] = big_hr_yearly[self.mergevar].apply(lambda x: x.dt.year)
                    big_hr_yearly = big_hr_yearly.drop(self.mergevar, axis = 1)
                    big_hr_yearly = big_hr_yearly.groupby('year')
                    
                    big_hit_by_year[y].loc[:, q] = (big_hr_yearly.sum()/big_moves_yearly.sum())
                    
                    # 2. ann. returns
                    ret_yearly = pd.concat([self.dates, ret], axis = 1)
                    ret_yearly['year'] = ret_yearly[self.mergevar].apply(lambda x: x.dt.year)
                    ret_yearly = ret_yearly.drop(self.mergevar, axis = 1)
                    ret_yearly = ret_yearly.groupby('year')

                    ann_returns_by_year[y].loc[:, q] = (ret_yearly.mean() * self.N)[0]

                    # 3. info ratios
                    info_ratio_by_year[y].loc[:, q] = ((ret_yearly.mean() / ret_yearly.std()) * (self.N**0.5))[0]
                    
                except:
                    continue
            
        self.hit_ratio =   hit_ratio
        self.ann_returns = ann_returns
        self.info_ratio =  info_ratio        
        self.hit_ratio_by_year =   hit_ratio_by_year
        self.big_hit_by_year =     big_hit_by_year
        self.ann_returns_by_year = ann_returns_by_year
        self.info_ratio_by_year =  info_ratio_by_year       
        self.signal_wt = signal_wt
 

    def construct_strategy(self, basket, signal_weighting = False):
        '''
        construct equal weighted & inverse vol strategy
        return time series of index &
        strategy ann. returns and info ratios by year
        '''
        import datetime as dt
        
        strategy_long = self.actual_returns[basket]
        real_vol = strategy_long.rolling(window = self.N).std()
        strategy_eq = pd.DataFrame() 
        strategy_iv = pd.DataFrame() 
        
        for p, q in zip(self.predictions, self.models):
            try:
                if signal_weighting:
                    temp = pd.concat([p[y][[y]].mul(self.signal_wt[y][q].shift(1), axis = 0) for y in basket], axis = 1)
                else:
                    temp = pd.concat([p[y][[y]] for y in basket], axis = 1)
                temp = np.sign(temp.replace(np.nan,0))
                # equal weighted strategy
                portfolio = strategy_long.mul(temp.iloc[self.actual_returns.index].values, axis = 0)                
                strategy_eq[q] = portfolio.mean(axis = 1)
                strategy_eq[q] = (pd.concat([pd.Series([100]), 1 + strategy_eq[q]])).cumprod().iloc[1:]
                # inverse volatility weighted strategy
                portfolio_iv =   portfolio.mul((real_vol.div(real_vol.sum(axis = 1), axis = 0)).values, axis = 0)
                strategy_iv[q] = portfolio_iv.sum(axis = 1)
                strategy_iv[q] = (pd.concat([pd.Series([100]), 1 + strategy_iv[q]])).cumprod().iloc[1:]

            except:
                continue
        
        self.strategy_eq =      pd.concat([self.dates, strategy_eq], axis = 1)
        self.strategy_iv =      pd.concat([self.dates, strategy_iv], axis = 1)
        
        # calculate ann returns and info ratios by year for strategy
        ret_eq = self.strategy_eq.copy()
        ret_eq['year'] = ret_eq[self.mergevar].apply(lambda x: x.dt.year)
        ret_eq = ret_eq.drop(self.mergevar, axis = 1)
        ret_eq.loc[:, ret_eq.columns != 'year'] = ret_eq.drop('year', axis = 1)/ret_eq.drop('year', axis = 1).shift(1)-1
        ret_eq = ret_eq.groupby('year')
                    
        self.strategy_eq_ret =  (ret_eq.mean() * self.N)
        self.strategy_eq_info = ((ret_eq.mean() / ret_eq.std()) * (self.N**0.5))
   
        ret_iv = self.strategy_iv.copy()
        ret_iv['year'] = ret_iv[self.mergevar].apply(lambda x: x.dt.year)
        ret_iv = ret_iv.drop(self.mergevar, axis = 1)
        ret_iv.loc[:, ret_iv.columns != 'year'] = ret_iv.drop('year', axis = 1)/ret_iv.drop('year', axis = 1).shift(1)-1
        ret_iv = ret_iv.groupby('year')
                    
        self.strategy_iv_ret =  (ret_iv.mean() * self.N)
        self.strategy_iv_info = ((ret_iv.mean() / ret_iv.std()) * (self.N**0.5))
            
    def output(self):
        '''
        collection of model outputs: 
            dictionary with betas, predictions, and tstats for each model           
        '''
    
        self.ols =    dict({'model' :'ols',
                            'beta' : self.beta_ols,
                            'predict' : self.predict_ols,
                            'score' : self.score_ols,
                            'tstats' : self.tstats_ols})
        self.lasso =  dict({'model' :'lasso',
                            'beta' : self.beta_lasso,
                            'predict' : self.predict_lasso,
                            'score' : self.score_lasso})
        self.ridge =  dict({'model' :'ridge',
                            'beta' : self.beta_ridge,
                            'predict' : self.predict_ridge,
                            'score' : self.score_ridge})
        self.corr_f = dict({'model' :'corr_filter_fixed',
                            'beta' : self.beta_corr_f,
                            'predict' : self.predict_corr_f,
                            'score' : self.score_corr_f,
                            'tstats' : self.tstats_corr_f})
        self.corr_d = dict({'model' :'corr_filter_d',
                            'beta' : self.beta_corr_d,
                            'predict' : self.predict_corr_d,
                            'score' : self.score_corr_d,
                            'tstats' : self.tstats_corr_d})
        self.pcr =    dict({'model' :'pcr',
                            'beta' : self.beta_pcr,
                            'predict' : self.predict_pcr,
                            'score' : self.score_pcr,
                            'tstats' : self.tstats_pcr})
        self.pls =    dict({'model' :'pls',
                            'beta' : self.beta_pls,
                            'predict' : self.predict_pls,
                            'score' : self.score_pls})
        self.rf =     dict({'model' :'random_forest',
                            'beta' : self.beta_rf,
                            'predict' : self.predict_rf,
                            'score' : self.score_rf})
        self.gb =     dict({'model' :'gradient_boost',
                            'beta' : self.beta_gb,
                            'predict' : self.predict_gb,
                            'score' : self.score_gb})
        self.lsvm =   dict({'model' :'svm_linear',
                            'beta' : self.beta_lsvm,
                            'predict' : self.predict_lsvm,
                            'score' : self.score_lsvm})
        self.psvm =   dict({'model' :'svm_polynomial',
                            'beta' : self.beta_psvm,
                            'predict' : self.predict_psvm,
                            'score' : self.score_psvm})
        self.rsvm =   dict({'model' :'svm_radial',
                            'beta' : self.beta_rsvm,
                            'predict' : self.predict_rsvm,
                            'score' : self.score_rsvm})

    
    def ols_call(self, X, Y):
        '''
        input dep and indep vars df
        call sm.OLS and calculate HAC standard errors 
        missing values dropped from regression
        return fitted model
        '''
        mod = sm.OLS(Y.astype(float), X.astype(float),missing='drop')
        results = mod.fit()
        results = results.get_robustcov_results(cov_type='HAC',maxlags=3)
        return results

    def lasso_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn.linear_model Lasso 
        return fitted model
        '''
        mod = LassoCV() #built-in cross-validation function chooses optimal model
        results = mod.fit(X.astype(float), Y.astype(float))
        return results

    def ridge_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn.linear_model Ridge 
        return fitted model
        '''
        mod = RidgeCV() #built-in cross-validation function chooses optimal model
        results = mod.fit(X.astype(float), Y.astype(float))
        return results

    def corr_f_call(self, X, Y, corr):
        '''
        input dep and indep vars df
        apply static correlation filter for variable selection
        call sm.OLS and calculate HAC standard errors 
        return fitted model
        '''
        #threshold = corr_thresh_f 
        vars_excluded = list(corr[abs(corr) < self.corr_thresh_f].index)
        vars_to_include = [x for x in list(X.columns) if x not in vars_excluded]
        
        mod = sm.OLS(Y.astype(float), X[vars_to_include].astype(float),missing='drop')
        results = mod.fit()
        return results

    def corr_d_call(self, X, Y, corr, corr_d, corr_std):
        '''
        input dep and indep vars df
        apply dynamic correlation filter for variable selection
        call sm.OLS and calculate HAC standard errors 
        return fitted model
        '''
        #identify decoupling
        #if corr > 0 & corr_d < 0  & abs(corr_d) > corr_std => decouple
        decouple =      list(corr[(corr > 0) & (corr_d < 0) & (abs(corr_d) > corr_std)].index)
        #if corr < 0 & corr_d > 0  & abs(corr_d) > corr_std => decouple
        decouple.append(list(corr[(corr < 0) & (corr_d > 0) & (abs(corr_d) > corr_std)].index))
        #threshold = fixed thresh & decoupled
        vars_excluded = list(corr[abs(corr) < 1/len(self.X.columns)].index)
        vars_excluded.append(decouple)
        vars_to_include = [x for x in list(X.columns) if x not in vars_excluded]
        
        mod = sm.OLS(Y.astype(float), X[vars_to_include].astype(float),missing='drop')
        results = mod.fit()
        return results
        
    def pcr_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn.PCA to extract PCs (first ncomp_pca components)
        return linear regression fitted model
        '''
        pca = PCA()
        X_reduced = pd.DataFrame(pca.fit_transform(scale(X.astype(float)))).iloc[:,:(self.ncomp_pca + 1)]
        mod = LinearRegression()
        results = mod.fit(X_reduced, Y)         
        X_exp_var = pca.explained_variance_ratio_[:(self.ncomp_pca + 1)]
        return results, X_reduced, X_exp_var

    def pls_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn PLSRegression to extract PLS dimensions (first ncomp_pls components)
        return linear regression fitted model
        '''
        X_reduced = pd.DataFrame(scale(X.astype(float)))
        mod = PLSRegression(n_components = self.ncomp_pls)
        results = mod.fit(X_reduced, Y.astype(float))
        return results, X_reduced
    
    def rf_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn RandomForestRegressor 
        return fitted model
        '''
        mod = RandomForestRegressor(random_state = 0, n_jobs = -1)
        results = mod.fit(X.astype(float), Y.astype(float))
        return results            
              
    def gb_call(self, X, Y):
        '''
        input dep and indep vars df
        call sklearn GradientBoostRegressor 
        return fitted model
        '''
        mod = GradientBoostingRegressor(random_state = 0)
        results = mod.fit(X.astype(float), Y.astype(float))
        return results            

    def svm_call(self, X, Y, kernel):
        '''
        input dep and indep vars df
        call sklearn SVR 
        return fitted model
        '''
        X_reduced = pd.DataFrame(scale(X.astype(float)))
        y_reduced = scale(Y.astype(float))
        mod = SVR(kernel = kernel, epsilon = 0.01)
        results = mod.fit(X_reduced, y_reduced)
        return results, X_reduced, y_reduced
    
    def charting(self, specs, fname, out_dir):
        '''
        chart performance metrics (hit ratio, info ratios and ann. returns) by year for all models and ensembles
        must run self.fit() and self.performance_stats() before calling this function
        works only with full model suite 
        '''
        import os
        import matplotlib.backends.backend_pdf
        import matplotlib.pyplot as plt
        os.chdir(out_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
           
        for p in self.hit_ratio.keys():
           
                ###############
                    #Hit Ratio -- two pages: individual models and ensembles
                ###############
    
            hit_ratio = self.hit_ratio_by_year[p].dropna()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(hit_ratio.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, hit_ratio.loc[:,'LR'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, hit_ratio.loc[:,'RR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='RR')
                   
            rect3 = plt.bar(index + 2*bar_width, hit_ratio.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='SVD')
              
            rects4 = plt.bar(index + 3*bar_width, hit_ratio.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Tree-based')

            rects5 = plt.bar(index + 4*bar_width, hit_ratio.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM')

            rects6 = plt.bar(index + 5*bar_width, hit_ratio.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects7 = plt.bar(index + 6*bar_width, hit_ratio.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects8 = plt.bar(index + 7*bar_width, hit_ratio.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 8*bar_width, hit_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('hit ratio')
            plt.title('Ensemble hit ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, hit_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=8+1, fontsize=7)
            pdf.savefig()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(hit_ratio.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, hit_ratio.loc[:,'OLS'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, hit_ratio.loc[:,'Lasso'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='L1')
            
            rects3 = plt.bar(index + 2*bar_width, hit_ratio.loc[:,'Ridge'], bar_width,
            alpha=opacity,
            color=[0.8, 0.2, 1.0],
            label='L2')
            
            rects4 = plt.bar(index + 3*bar_width, hit_ratio.loc[:,'Correlation_Filter_F'], bar_width,
            alpha=opacity,
            color=[0.4, 0.4, 0.4],
            label='CF_F')
                
            rects5 = plt.bar(index + 4*bar_width, hit_ratio.loc[:,'Correlation_Filter_D'], bar_width,
            alpha=opacity,
            color=[0.8, 0.8, 0.8],
            label='CF_D')
        
            rects6 = plt.bar(index + 5*bar_width, hit_ratio.loc[:,'PCR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='PCR')
        
            rects7 = plt.bar(index + 6*bar_width, hit_ratio.loc[:,'PLS'], bar_width,
            alpha=opacity,
            color=[1.0, 0.6, 0.0],
            label='PLS')
    
            rects8 = plt.bar(index + 7*bar_width, hit_ratio.loc[:,'Random_Forest'], bar_width,
            alpha=opacity,
            color=[0.4, 0.8, 0.6],
            label='RF')
    
            rects9 = plt.bar(index + 8*bar_width, hit_ratio.loc[:,'Gradient_Boost'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='GB')

            rects10 = plt.bar(index + 9*bar_width, hit_ratio.loc[:,'SVM (L)'], bar_width,
            alpha=opacity,
            color=[0.4, 1.0, 0.6],
            label='SVM (L)')

            rects11 = plt.bar(index + 10*bar_width, hit_ratio.loc[:,'SVM (P)'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM (P)')

            rects12 = plt.bar(index + 11*bar_width, hit_ratio.loc[:,'SVM (R)'], bar_width,
            alpha=opacity,
            color=[0.8, 0.5, 0.8],
            label='SVM (R)')

            rects13 = plt.bar(index + 12*bar_width, hit_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('hit ratio')
            plt.title('Model hit ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, hit_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=hit_ratio.shape[1]-8, fontsize=7)
            pdf.savefig()

          
                ###############
                    #Info Ratio
                ###############
    
            info_ratio = self.info_ratio_by_year[p].dropna()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(info_ratio.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, info_ratio.loc[:,'LR'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, info_ratio.loc[:,'RR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='RR')
                   
            rect3 = plt.bar(index + 2*bar_width, info_ratio.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='SVD')
              
            rects4 = plt.bar(index + 3*bar_width, info_ratio.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Tree-based')

            rects5 = plt.bar(index + 4*bar_width, info_ratio.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM')

            rects6 = plt.bar(index + 5*bar_width, info_ratio.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects7 = plt.bar(index + 6*bar_width, info_ratio.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects8 = plt.bar(index + 7*bar_width, info_ratio.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 8*bar_width, info_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('info ratio')
            plt.title('Ensemble information ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, info_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=8+1, fontsize=7)
            pdf.savefig()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(info_ratio.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, info_ratio.loc[:,'OLS'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, info_ratio.loc[:,'Lasso'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='L1')
            
            rects3 = plt.bar(index + 2*bar_width, info_ratio.loc[:,'Ridge'], bar_width,
            alpha=opacity,
            color=[0.8, 0.2, 1.0],
            label='L2')
            
            rects4 = plt.bar(index + 3*bar_width, info_ratio.loc[:,'Correlation_Filter_F'], bar_width,
            alpha=opacity,
            color=[0.4, 0.4, 0.4],
            label='CF_F')
                
            rects5 = plt.bar(index + 4*bar_width, info_ratio.loc[:,'Correlation_Filter_D'], bar_width,
            alpha=opacity,
            color=[0.8, 0.8, 0.8],
            label='CF_D')
        
            rects6 = plt.bar(index + 5*bar_width, info_ratio.loc[:,'PCR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='PCR')
        
            rects7 = plt.bar(index + 6*bar_width, info_ratio.loc[:,'PLS'], bar_width,
            alpha=opacity,
            color=[1.0, 0.6, 0.0],
            label='PLS')
    
            rects8 = plt.bar(index + 7*bar_width, info_ratio.loc[:,'Random_Forest'], bar_width,
            alpha=opacity,
            color=[0.4, 0.8, 0.6],
            label='RF')
    
            rects9 = plt.bar(index + 8*bar_width, info_ratio.loc[:,'Gradient_Boost'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='GB')

            rects10 = plt.bar(index + 9*bar_width, info_ratio.loc[:,'SVM (L)'], bar_width,
            alpha=opacity,
            color=[0.4, 1.0, 0.6],
            label='SVM (L)')

            rects11 = plt.bar(index + 10*bar_width, info_ratio.loc[:,'SVM (P)'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM (P)')

            rects12 = plt.bar(index + 11*bar_width, info_ratio.loc[:,'SVM (R)'], bar_width,
            alpha=opacity,
            color=[0.8, 0.5, 0.8],
            label='SVM (R)')

            rects13 = plt.bar(index + 12*bar_width, info_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('info ratio')
            plt.title('Model information ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, info_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=info_ratio.shape[1]-8, fontsize=7)
            pdf.savefig()

         
                ###############
                    #Ann returns
                ###############
    
            ann_returns = self.ann_returns_by_year[p].replace(0, np.nan).dropna()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(ann_returns.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, ann_returns.loc[:,'LR'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, ann_returns.loc[:,'RR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='RR')
                   
            rect3 = plt.bar(index + 2*bar_width, ann_returns.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='SVD')
              
            rects4 = plt.bar(index + 3*bar_width, ann_returns.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Tree-based')

            rects5 = plt.bar(index + 4*bar_width, ann_returns.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM')

            rects6 = plt.bar(index + 5*bar_width, ann_returns.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects7 = plt.bar(index + 6*bar_width, ann_returns.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects8 = plt.bar(index + 7*bar_width, ann_returns.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 8*bar_width, ann_returns.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('ann. ret.')
            plt.title('Ensemble annualized returns: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, ann_returns.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=8+1, fontsize=7)
            pdf.savefig()
            
            # create plot
            fig, ax = plt.subplots(figsize=(10,6))
            index = np.arange(ann_returns.shape[0])
            bar_width = 0.05
            opacity = 0.8
            
            rects1 = plt.bar(index, ann_returns.loc[:,'OLS'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 0.0],
            label='OLS')
                
            rects2 = plt.bar(index + bar_width, ann_returns.loc[:,'Lasso'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='L1')
            
            rects3 = plt.bar(index + 2*bar_width, ann_returns.loc[:,'Ridge'], bar_width,
            alpha=opacity,
            color=[0.8, 0.2, 1.0],
            label='L2')
            
            rects4 = plt.bar(index + 3*bar_width, ann_returns.loc[:,'Correlation_Filter_F'], bar_width,
            alpha=opacity,
            color=[0.4, 0.4, 0.4],
            label='CF_F')
                
            rects5 = plt.bar(index + 4*bar_width, ann_returns.loc[:,'Correlation_Filter_D'], bar_width,
            alpha=opacity,
            color=[0.8, 0.8, 0.8],
            label='CF_D')
        
            rects6 = plt.bar(index + 5*bar_width, ann_returns.loc[:,'PCR'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='PCR')
        
            rects7 = plt.bar(index + 6*bar_width, ann_returns.loc[:,'PLS'], bar_width,
            alpha=opacity,
            color=[1.0, 0.6, 0.0],
            label='PLS')
    
            rects8 = plt.bar(index + 7*bar_width, ann_returns.loc[:,'Random_Forest'], bar_width,
            alpha=opacity,
            color=[0.4, 0.8, 0.6],
            label='RF')
    
            rects9 = plt.bar(index + 8*bar_width, ann_returns.loc[:,'Gradient_Boost'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='GB')

            rects10 = plt.bar(index + 9*bar_width, ann_returns.loc[:,'SVM (L)'], bar_width,
            alpha=opacity,
            color=[0.4, 1.0, 0.6],
            label='SVM (L)')

            rects11 = plt.bar(index + 10*bar_width, ann_returns.loc[:,'SVM (P)'], bar_width,
            alpha=opacity,
            color=[0.4, 0.6, 0.8],
            label='SVM (P)')

            rects12 = plt.bar(index + 11*bar_width, ann_returns.loc[:,'SVM (R)'], bar_width,
            alpha=opacity,
            color=[0.8, 0.5, 0.8],
            label='SVM (R)')

            rects13 = plt.bar(index + 12*bar_width, ann_returns.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('ann. ret.')
            plt.title('Model annualized returns: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, ann_returns.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=ann_returns.shape[1]-8, fontsize=7)
            pdf.savefig()
          
        pdf.close()

    def charting_strat(self, strat_nick, specs, fname, out_dir):
        '''
        chart performance of equal weighted and inverse volatility weighted strategies
        must run self.construct_strategy() and self.ensemble() before calling this function
        Note: works only with full model suite
        '''
        import os
        import matplotlib.backends.backend_pdf
        import matplotlib.pyplot as plt
        os.chdir(out_dir)
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
    
            ###############
                #Strategy Performance
            ###############
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Long_Only'],             '-', color=[1.0, 0.0, 0.0], label='Long-Only')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['LR'],                    '-', color=[0.0, 0.0, 0.0], label='OLS')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['RR'],                    '-', color=[1.0, 0.5, 0.8], label='RR')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['SVD'],                   '-', color=[1.0, 0.8, 0.2], label='SVD')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['DT'],                    '-', color=[0.0, 1.0, 0.0], label='Tree-based')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['SVM'],                   '-', color=[0.4, 0.6, 0.8], label='SVM')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble'],              '-', color=[0.0, 0.0, 1.0], label='Ensemble (max voting)')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble2'],             '-', color=[0.5, 0.5, 1.0], label='Ensemble (simple avg.)')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble3'],             '-', color=[0.5, 1.0, 1.0], label='Ensemble (weighted avg.)')
        plt.title('Ensemble equal-weighted strategies: ' + specs, fontsize=10)
        plt.suptitle(strat_nick)
        plt.ylabel('performance')
        plt.xlabel('')
        plt.legend()
        pdf.savefig()
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Long_Only'],             '-', color=[1.0, 0.0, 0.0], label='Long-Only')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['LR'],                    '-', color=[0.0, 0.0, 0.0], label='OLS')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['RR'],                    '-', color=[1.0, 0.5, 0.8], label='RR')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['SVD'],                   '-', color=[1.0, 0.8, 0.2], label='SVD')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['DT'],                    '-', color=[0.0, 1.0, 0.0], label='Tree-based')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['SVM'],                   '-', color=[0.4, 0.6, 0.8], label='SVM')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble'],              '-', color=[0.0, 0.0, 1.0], label='Ensemble (max voting)')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble2'],             '-', color=[0.5, 0.5, 1.0], label='Ensemble (simple avg.)')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble3'],             '-', color=[0.5, 1.0, 1.0], label='Ensemble (weighted avg.)')
        plt.title('Ensemble iVol-weighted strategies: ' + specs, fontsize=10)
        plt.suptitle(strat_nick)
        plt.ylabel('performance')
        plt.xlabel('')
        plt.legend()
        pdf.savefig()
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Long_Only'],             '-', color=[1.0, 0.0, 0.0], label='Long-Only')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['OLS'],                   '-', color=[0.0, 0.0, 0.0], label='OLS')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Lasso'],                 '-', color=[1.0, 0.5, 0.8], label='Lasso')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ridge'],                 '-', color=[0.8, 0.2, 1.0], label='Ridge')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Correlation_Filter_F'],  '-', color=[0.4, 0.4, 0.4], label='C_filter_F')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Correlation_Filter_D'],  '-', color=[0.8, 0.8, 0.8], label='C_filter_D')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['PCR'],                   '-', color=[1.0, 0.8, 0.2], label='PCR')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['PLS'],                   '-', color=[1.0, 0.6, 0.0], label='PLS')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Random_Forest'],         '-', color=[0.4, 0.8, 0.6], label='RF')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Gradient_Boost'],        '-', color=[0.0, 1.0, 0.0], label='GB')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['SVM (L)'],               '-', color=[0.4, 1.0, 0.6], label='SVM (L)')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['SVM (P)'],               '-', color=[0.4, 0.6, 0.8], label='SVM (P)')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['SVM (R)'],               '-', color=[0.8, 0.5, 0.8], label='SVM (R)')
        plt.title('Equal-weighted strategy: ' + specs, fontsize=10)
        plt.suptitle(strat_nick)
        plt.ylabel('performance')
        plt.xlabel('')
        plt.legend()
        pdf.savefig()
        
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Long_Only'],             '-', color=[1.0, 0.0, 0.0], label='Long-Only')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['OLS'],                   '-', color=[0.0, 0.0, 0.0], label='OLS')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Lasso'],                 '-', color=[1.0, 0.5, 0.8], label='Lasso')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ridge'],                 '-', color=[0.8, 0.2, 1.0], label='Ridge')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Correlation_Filter_F'],  '-', color=[0.4, 0.4, 0.4], label='C_filter_F')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Correlation_Filter_D'],  '-', color=[0.8, 0.8, 0.8], label='C_filter_D')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['PCR'],                   '-', color=[1.0, 0.8, 0.2], label='PCR')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['PLS'],                   '-', color=[1.0, 0.6, 0.0], label='PLS')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Random_Forest'],         '-', color=[0.4, 0.8, 0.6], label='RF')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Gradient_Boost'],        '-', color=[0.0, 1.0, 0.0], label='GB')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['SVM (L)'],               '-', color=[0.4, 1.0, 0.6], label='SVM (L)')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['SVM (P)'],               '-', color=[0.4, 0.6, 0.8], label='SVM (P)')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['SVM (R)'],               '-', color=[0.8, 0.5, 0.8], label='SVM (R)')
        plt.title('iVol-weighted strategy: ' + specs, fontsize=10)
        plt.suptitle(strat_nick)
        plt.ylabel('performance')
        plt.xlabel('')
        plt.legend()
        pdf.savefig()
        
        pdf.close()    