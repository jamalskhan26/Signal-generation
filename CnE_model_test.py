# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:28:26 2020

@author: Jamal Khan
"""

'''
Chicken and Egg model:
    forecast FX returns using commodity returns
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
import time
### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Data'
out_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CnE/Output'

fname=os.path.join(data_dir, 'signal_returns_co.pickle')
signal_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'signal_returns_fx.pickle')
signal_fx = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_co.pickle')
trade_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_fx.pickle')
trade_fx = pickle.load( open( fname, "rb" ) )


fname=os.path.join(data_dir, 'signal_returns_daily_co.pickle')
signal_d_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'signal_returns_daily_fx.pickle')
signal_d_fx = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_daily_co.pickle')
trade_d_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_daily_fx.pickle')
trade_d_fx = pickle.load( open( fname, "rb" ) )


fname=os.path.join(data_dir, 'signal_returns_monthly_co.pickle')
signal_m_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'signal_returns_monthly_fx.pickle')
signal_m_fx = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_monthly_co.pickle')
trade_m_co = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'trade_returns_monthly_fx.pickle')
trade_m_fx = pickle.load( open( fname, "rb" ) )

#############################################################
# Run models -- CNE &EnC
#############################################################

'''
CnE
'''

#############################################################
# Run models -- Weekly
#############################################################

#########################
###UPDATE with current###
#########################

cne = regression_models(X = signal_co.replace(np.nan, 0).iloc[-200:, :], Y = trade_fx.replace(np.nan, 0).iloc[-200:, :], freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

cne.fit()

with open(os.path.join(data_dir, 'cneModels_latest.pickle'), 'wb') as output:
    pickle.dump(cne, output)

#########################

start1 = time.time()

cne = regression_models(X = signal_co.replace(np.nan, 0), Y = trade_fx.replace(np.nan, 0), freq = 'weekly', window = 260, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

## save object so you don't have to fit it again and access all the object attributes any time by loading it back onto your wd

# save model object
with open(os.path.join(data_dir, 'cneModels_5Ylookback.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

start2 = time.time()

cne = regression_models(X = signal_co.replace(np.nan, 0), Y = trade_fx.replace(np.nan, 0), freq = 'weekly', window = 150, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

## save object so you don't have to fit it again and access all the object attributes any time by loading it back onto your wd

# save model object
with open(os.path.join(data_dir, 'cneModels_3Ylookback.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')

start3 = time.time()

cne = regression_models(X = signal_co.replace(np.nan, 0), Y = trade_fx.replace(np.nan, 0), freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

## save object so you don't have to fit it again and access all the object attributes any time by loading it back onto your wd

# save model object
with open(os.path.join(data_dir, 'cneModels_default.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end3 = time.time()
print('Time to execute:', round((end3-start3)/60), 'minutes')

#############################################################
# Run models -- Daily
#############################################################

start1 = time.time()

cne = regression_models(X = signal_d_co.replace(np.nan, 0), Y = trade_d_fx.replace(np.nan, 0), freq = 'daily', window = 252, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# save model object
with open(os.path.join(data_dir, 'cneModels_1Ydaily.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

start2 = time.time()

cne = regression_models(X = signal_d_co.replace(np.nan, 0), Y = trade_d_fx.replace(np.nan, 0), freq = 'daily', window = 126, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# save model object
with open(os.path.join(data_dir, 'cneModels_6mdaily.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')

start3 = time.time()

cne = regression_models(X = signal_d_co.replace(np.nan, 0), Y = trade_d_fx.replace(np.nan, 0), freq = 'daily', window = 63, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# save model object
with open(os.path.join(data_dir, 'cneModels_3mdaily.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end3 = time.time()
print('Time to execute:', round((end3-start3)/60), 'minutes')

start4 = time.time()

cne = regression_models(X = signal_d_co.replace(np.nan, 0), Y = trade_d_fx.replace(np.nan, 0), freq = 'daily', window = 1000, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# save model object
with open(os.path.join(data_dir, 'cneModels_4Ydaily.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end4 = time.time()
print('Time to execute:', round((end4-start4)/60), 'minutes')


#############################################################
# Run models -- Monthly 
#############################################################

start1 = time.time()

cne = regression_models(X = signal_m_co.replace(np.nan, 0), Y = trade_m_fx.replace(np.nan, 0), freq = 'monthly', window = 60, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# save model object
with open(os.path.join(data_dir, 'cneModels_5Ymonthly.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

'''
EnC
'''

#############################################################
# Run models -- Daily
#############################################################

#########################
###UPDATE with current###
#########################

enc = regression_models(X = signal_d_fx.replace(np.nan, 0).iloc[-600:,], Y = trade_d_co.replace(np.nan, 0).iloc[-600:,], freq = 'daily', window = 252, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

enc.fit()

with open(os.path.join(data_dir, 'encModels_daily_latest.pickle'), 'wb') as output:
    pickle.dump(enc, output)

#########################

start1 = time.time()

enc1 = regression_models(X = signal_d_fx.replace(np.nan, 0), Y = trade_d_co.replace(np.nan, 0), freq = 'daily', window = 252, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc1.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_1Ydaily.pickle'), 'wb') as output:
    pickle.dump(enc1, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

start2 = time.time()

enc2 = regression_models(X = signal_d_fx.replace(np.nan, 0), Y = trade_d_co.replace(np.nan, 0), freq = 'daily', window = 126, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc2.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_6mdaily.pickle'), 'wb') as output:
    pickle.dump(enc2, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')

start3 = time.time()

enc3 = regression_models(X = signal_d_fx.replace(np.nan, 0), Y = trade_d_co.replace(np.nan, 0), freq = 'daily', window = 63, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc3.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_3mdaily.pickle'), 'wb') as output:
    pickle.dump(enc3, output)

end3 = time.time()
print('Time to execute:', round((end3-start3)/60), 'minutes')

start4 = time.time()

enc4 = regression_models(X = signal_d_fx.replace(np.nan, 0), Y = trade_d_co.replace(np.nan, 0), freq = 'daily', window = 1000, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc4.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_4Ydaily.pickle'), 'wb') as output:
    pickle.dump(enc4, output)

end4 = time.time()
print('Time to execute:', round((end4-start4)/60), 'minutes')

#############################################################
# Run models -- Weekly
#############################################################


#########################
###UPDATE with current###
#########################

enc = regression_models(X = signal_fx.replace(np.nan, 0).iloc[-200:, :], Y = trade_co.replace(np.nan, 0).iloc[-200:, :], freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

enc.fit()

with open(os.path.join(data_dir, 'encModels_latest.pickle'), 'wb') as output:
    pickle.dump(enc, output)

#########################

start1 = time.time()

enc = regression_models(X = signal_fx.replace(np.nan, 0), Y = trade_co.replace(np.nan, 0), freq = 'weekly', window = 260, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_5Ylookback.pickle'), 'wb') as output:
    pickle.dump(enc, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

start2 = time.time()

enc = regression_models(X = signal_fx.replace(np.nan, 0), Y = trade_co.replace(np.nan, 0), freq = 'weekly', window = 150, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_3Ylookback.pickle'), 'wb') as output:
    pickle.dump(enc, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')

start3 = time.time()

enc = regression_models(X = signal_fx.replace(np.nan, 0), Y = trade_co.replace(np.nan, 0), freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_1Ylookback.pickle'), 'wb') as output:
    pickle.dump(enc, output)

end3 = time.time()
print('Time to execute:', round((end3-start3)/60), 'minutes')

#############################################################
# Run models -- Monthly 
#############################################################

start2 = time.time()

enc = regression_models(X = signal_m_fx.replace(np.nan, 0), Y = trade_m_co.replace(np.nan, 0), freq = 'monthly', window = 60, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc.fit()

# save model object
with open(os.path.join(data_dir, 'encModels_5Ymonthly.pickle'), 'wb') as output:
    pickle.dump(enc, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')


#############################################################
# Analyze models
#############################################################
# load model object
fname=os.path.join(data_dir, 'encModels_1Ylookback.pickle')
test = pickle.load( open( fname, "rb" ) )
test.ensemble()
test.performance_stats()

for p in cne.hit_ratio_by_year.keys(): 
    cne.hit_ratio_by_year[p] =  cne.hit_ratio_by_year[p][cne.hit_ratio_by_year[p].index!=2020]
    cne.info_ratio_by_year[p] = cne.info_ratio_by_year[p][cne.info_ratio_by_year[p].index!=2020]
    cne.ann_returns_by_year[p] = cne.ann_returns_by_year[p][cne.ann_returns_by_year[p].index!=2020]
    
cne.strategy_eq = cne.strategy_eq[cne.strategy_eq[cne.mergevar] < '2020-01-01']
cne.strategy_iv = cne.strategy_iv[cne.strategy_iv[cne.mergevar] < '2020-01-01']


cne.charting =       regression_models.charting
cne.charting_strat = regression_models.charting_strat

### access attributes

# performance metrics
hr = cne.hit_ratio
ar = cne.ann_returns
cne.ann_returns['Australia']
ir = cne.info_ratio
cne.info_ratio['Australia']
hry = cne.hit_ratio_by_year
bhy = cne.big_hit_by_year
ary = cne.ann_returns_by_year
iry = cne.info_ratio_by_year


# equal weighted strat 
s_e = cne.strategy_eq
s_e_ir = cne.strategy_eq_info
s_e_re = cne.strategy_eq_ret

# ivol weighted strat
s_i = cne.strategy_iv
s_i_ir = cne.strategy_iv_info
s_i_re = cne.strategy_iv_ret

# look at model outputs
cne.ols.keys()
cne.ols['tstats']
cne.ols['predict']
cne.ols['score']
cne.ols['beta']

# load model object
fname=os.path.join(data_dir, 'encModels_default.pickle')
enc = pickle.load( open( fname, "rb" ) )

# performance metrics
enc.hit_ratio
enc.ann_returns
enc.info_ratio
enc.hit_ratio_by_year
enc.big_hit_by_year
enc.ann_returns_by_year
enc.info_ratio_by_year
#############################################################
# Test
#############################################################



from copy import deepcopy

cne.ensemble=ensemble
cne.ensemble()
#cne.signal_weighting=True
cne.performance_stats=performance_stats
cne.performance_stats()

hr = cne.hit_ratio
ar = cne.ann_returns
ir = cne.info_ratio
hry = cne.hit_ratio_by_year
bhy = cne.big_hit_by_year
ary = cne.ann_returns_by_year
iry = cne.info_ratio_by_year


for p,q in zip(cne.hit_ratio.values(),cne.hit_ratio.keys()):
    print(q, round(p.loc['Ensemble'][0],2))
for p,q in zip(cne.ann_returns.values(),cne.ann_returns.keys()):
    print(q, round(p.loc['Ensemble'][0],2))
for p,q in zip(cne.ann_returns.values(),cne.ann_returns.keys()):
    print(q, round(p.loc['Ensemble'][0] - p.loc['Long_Only'][0],2))

dmfx = ['Australia','European Union','Canada','Switzerland','United Kingdom','Japan','Norway','New Zealand','Sweden']
emfx = ['Czech','Hungary','Poland','Romania',
        'South Africa','Israel','Russia','Turkey',
        'Korea','Indonesia','Philippines','Thailand','Taiwan','India','Singapore','China',
        'Brazil','Chile','Peru','Colombia','Mexico']
emfx_1 = ['Czech','Hungary','Poland','Romania', 'Israel']
emfx_2 = ['Korea','Indonesia','Philippines','Thailand','Taiwan','India','Singapore','China', 'Turkey']
emfx_3 = ['Brazil','Chile','Peru','Colombia','Mexico']
emfx_4 = ['Brazil','Russia','India','China','South Africa']
emfx_5 = np.unique(emfx_2 + emfx_3)
emfx_6 = np.unique(emfx_2 + emfx_3 + emfx_4)
      
# contruct strategies corresponding to a basket of currencies

#1 - all EMFX
cne.construct_strategy(basket = emfx)
emfx_strat_eq = cne.strategy_eq
emfx_strat_iv = cne.strategy_iv
emfx_strat_eq_ir = cne.strategy_eq_info
emfx_strat_iv_ir = cne.strategy_iv_info
emfx_strat_eq_re = cne.strategy_eq_ret
emfx_strat_iv_re = cne.strategy_iv_ret
cne.charting_strat(cne, strat_nick = "All EMFX", specs = "5Y lookback, no signal weighting", fname = "CnE_5Yroll_nw_all_strat.pdf", out_dir = out_dir)
cne.charting(specs = "5Y lookback, no signal weighting", fname = "CnE_5Yroll_nw.pdf", out_dir = out_dir)

#2 - EastEUR + ME
cne.construct_strategy(basket = emfx_1)
emfx1_strat_eq = cne.strategy_eq
emfx1_strat_iv = cne.strategy_iv
emfx1_strat_eq_ir = cne.strategy_eq_info
emfx1_strat_iv_ir = cne.strategy_iv_info
emfx1_strat_eq_re = cne.strategy_eq_ret
emfx1_strat_iv_re = cne.strategy_iv_ret
charting(emfx1_strat_eq)
charting(emfx1_strat_iv)

#3 - Asia - good
cne.construct_strategy(basket = emfx_2)
emfx2_strat_eq = cne.strategy_eq
emfx2_strat_iv = cne.strategy_iv
emfx2_strat_eq_ir = cne.strategy_eq_info
emfx2_strat_iv_ir = cne.strategy_iv_info
emfx2_strat_eq_re = cne.strategy_eq_ret
emfx2_strat_iv_re = cne.strategy_iv_ret
charting(emfx2_strat_eq)
charting(emfx2_strat_iv)

#4 - SouthAMR - good
cne.construct_strategy(basket = emfx_3)
emfx3_strat_eq = cne.strategy_eq
emfx3_strat_iv = cne.strategy_iv
emfx3_strat_eq_ir = cne.strategy_eq_info
emfx3_strat_iv_ir = cne.strategy_iv_info
emfx3_strat_eq_re = cne.strategy_eq_ret
emfx3_strat_iv_re = cne.strategy_iv_ret
charting(emfx3_strat_eq)
charting(emfx3_strat_iv)

#4 - BRICS - good
cne.construct_strategy(basket = emfx_4)
emfx4_strat_eq = cne.strategy_eq
emfx4_strat_iv = cne.strategy_iv
emfx4_strat_eq_ir = cne.strategy_eq_info
emfx4_strat_iv_ir = cne.strategy_iv_info
emfx4_strat_eq_re = cne.strategy_eq_ret
emfx4_strat_iv_re = cne.strategy_iv_ret
charting(emfx4_strat_eq)
charting(emfx4_strat_iv)
charting(cne.strategy_eq)

#5 - Asia + South AMR - good
cne.construct_strategy(basket = emfx_5)
emfx5_strat_eq = cne.strategy_eq
emfx5_strat_iv = cne.strategy_iv
emfx5_strat_eq_ir = cne.strategy_eq_info
emfx5_strat_iv_ir = cne.strategy_iv_info
emfx5_strat_eq_re = cne.strategy_eq_ret
emfx5_strat_iv_re = cne.strategy_iv_ret
charting(emfx5_strat_eq)
charting(emfx5_strat_iv)

#6 - Asia + South AMR + BRICS - good
cne.construct_strategy(basket = emfx_6)
emfx6_strat_eq = cne.strategy_eq
emfx6_strat_iv = cne.strategy_iv
emfx6_strat_eq_ir = cne.strategy_eq_info
emfx6_strat_iv_ir = cne.strategy_iv_info
emfx6_strat_eq_re = cne.strategy_eq_ret
emfx6_strat_iv_re = cne.strategy_iv_ret
charting(emfx6_strat_eq)
charting(emfx6_strat_iv)

#########################################


#########################################

    def ensemble(self):
        from copy import deepcopy
        if bool(set(['LR', 'SVM', 'SVD', 'DT','Ensemble', 'Ensemble2', 'Ensemble3']).intersection(set(self.models))):           
            self.models = [x for x in self.models if x not in set(['LR', 'SVM', 'SVD', 'DT', 'Ensemble', 'Ensemble2', 'Ensemble3']).intersection(set(self.models))]
            self.predictions = [{x : abs(self.actual_returns)[[x]] for x in self.actual_returns.columns}, 
                                 self.predict_ols, self.predict_lasso, self.predict_ridge, self.predict_corr_f, 
                                 self.predict_corr_d, self.predict_pcr, self.predict_pls, self.predict_rf, self.predict_gb]

        # model family ensembles
        e_ols = deepcopy(self.predictions[2])
        e_svm = deepcopy(self.predictions[2])
        e_svd = deepcopy(self.predictions[2])
        e_dtr = deepcopy(self.predictions[2])

        for p in self.predictions[0].keys():
            # OLS based methods -- OLS, correlation filters (dynamic and static)
            e_ols[p][p] = (self.predictions[1][p][p] + self.predictions[4][p][p] + self.predictions[5][p][p])/3

            # SVM -- support vector machines (lasso and ridge)
            e_svm[p][p] = (self.predictions[2][p][p] + self.predictions[3][p][p])/2
            
            # SVD -- singular value decomposition (PCR and PLS)
            e_svd[p][p] = (self.predictions[6][p][p] + self.predictions[7][p][p])/2

            # DTR -- Decision Trees (RF and GB)
            e_dtr[p][p] = (self.predictions[8][p][p] + self.predictions[9][p][p])/2

        prediction_sign = []
        temp = deepcopy([e_ols,e_svm,e_svd,e_dtr])    
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
            x = prediction_sign[0][p][p] + prediction_sign[1][p][p] + prediction_sign[2][p][p] + prediction_sign[3][p][p] 
            y = self.scores[2][p][p] + self.scores[3][p][p] + self.scores[4][p][p] + self.scores[5][p][p] + self.scores[6][p][p] + self.scores[7][p][p] + self.scores[8][p][p] + self.scores[9][p][p]
            x[x < 3] = -1
            x[x >= 3] = 1            
            ensemble[p][p] = x
            comb_score[p][p] = y
            
            # Ensemble 2 -- simple average
            ensemble2[p][p] = 0.25*e_ols[p][p] + 0.25*e_svm[p][p] + 0.25*e_ols[p][p] + 0.25*e_svd[p][p] + 0.25*e_dtr[p][p]
            
            # Ensemble 3 -- weighted average (by rolling modified residuals)
            r_lasso = (self.actual_returns[p] - self.predictions[2][p][p]) * np.sign(self.predictions[2][p][p].replace(np.nan,0))*-1
            r_ridge = (self.actual_returns[p] - self.predictions[3][p][p]) * np.sign(self.predictions[3][p][p].replace(np.nan,0))*-1
            r_corrf = (self.actual_returns[p] - self.predictions[4][p][p]) * np.sign(self.predictions[4][p][p].replace(np.nan,0))*-1
            r_corrd = (self.actual_returns[p] - self.predictions[5][p][p]) * np.sign(self.predictions[5][p][p].replace(np.nan,0))*-1
            r_pcr =   (self.actual_returns[p] - self.predictions[6][p][p]) * np.sign(self.predictions[6][p][p].replace(np.nan,0))*-1
            r_pls =   (self.actual_returns[p] - self.predictions[7][p][p]) * np.sign(self.predictions[7][p][p].replace(np.nan,0))*-1
            r_rf =    (self.actual_returns[p] - self.predictions[8][p][p]) * np.sign(self.predictions[8][p][p].replace(np.nan,0))*-1
            r_gb =    (self.actual_returns[p] - self.predictions[9][p][p]) * np.sign(self.predictions[9][p][p].replace(np.nan,0))*-1
                ## calculate weights as standardized residuals
            w_lasso = ((r_lasso.rolling(window = self.N).mean()) / r_lasso.rolling(window = self.N).std())
            w_ridge = ((r_ridge.rolling(window = self.N).mean()) / r_ridge.rolling(window = self.N).std())
            w_corrf = ((r_corrf.rolling(window = self.N).mean()) / r_corrf.rolling(window = self.N).std())
            w_corrd = ((r_corrd.rolling(window = self.N).mean()) / r_corrd.rolling(window = self.N).std())
            w_pcr =   ((r_pcr.rolling(window = self.N).mean()) / r_pcr.rolling(window = self.N).std())
            w_pls =   ((r_pls.rolling(window = self.N).mean()) / r_pls.rolling(window = self.N).std())
            w_rf =    ((r_rf.rolling(window = self.N).mean()) / r_rf.rolling(window = self.N).std())
            w_gb =    ((r_gb.rolling(window = self.N).mean()) / r_gb.rolling(window = self.N).std())
                ## normalize weights between 0 and 1
            w_min = pd.concat([w_lasso, w_ridge, w_corrf, w_corrd, w_pcr, w_pls, w_rf, w_gb], axis = 1).min(axis = 1)
            w_max = pd.concat([w_lasso, w_ridge, w_corrf, w_corrd, w_pcr, w_pls, w_rf, w_gb], axis = 1).max(axis = 1)
            w_range = w_max - w_min
            w_lasso = ((w_lasso - w_min) / w_range).shift(1)
            w_ridge = ((w_ridge - w_min) / w_range).shift(1)
            w_corrf = ((w_corrf - w_min) / w_range).shift(1)
            w_corrd = ((w_corrd - w_min) / w_range).shift(1)
            w_pcr =   ((w_pcr - w_min) / w_range).shift(1)
            w_pls =   ((w_pls - w_min) / w_range).shift(1)
            w_rf =    ((w_rf - w_min) / w_range).shift(1)
            w_gb =    ((w_gb - w_min) / w_range).shift(1)
            ensemble3[p][p] = w_lasso*self.predictions[2][p][p] + w_ridge*self.predictions[3][p][p] + w_corrf*self.predictions[4][p][p] + w_corrd*self.predictions[5][p][p] + w_pcr*self.predictions[6][p][p] + w_pls*self.predictions[7][p][p] + w_rf*self.predictions[8][p][p] + w_gb*self.predictions[9][p][p] 
                                            
        self.models.extend(['LR', 'SVM', 'SVD', 'DT', 'Ensemble', 'Ensemble2', 'Ensemble3'])
        self.predictions.extend([e_ols, e_svm, e_svd, e_dtr, ensemble, ensemble2, ensemble3])
        self.scores.extend([comb_score]*7)


    def charting(self, specs, fname, out_dir):
        '''
        chart performance metrics (hit ratio, info ratios and ann. returns) by year for all models
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
            label='Linear Regression')
                
            rects2 = plt.bar(index + bar_width, hit_ratio.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='Support Vector Machines')
                   
            rect3 = plt.bar(index + 2*bar_width, hit_ratio.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='Singular Value Decomposition')
              
            rects4 = plt.bar(index + 3*bar_width, hit_ratio.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Decision Trees')

            rects5 = plt.bar(index + 4*bar_width, hit_ratio.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects6 = plt.bar(index + 5*bar_width, hit_ratio.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects7 = plt.bar(index + 6*bar_width, hit_ratio.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 7*bar_width, hit_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('hit ratio')
            plt.title('Ensemble hit ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, hit_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=7+1, fontsize=7)
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

            rects10 = plt.bar(index + 9*bar_width, hit_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('hit ratio')
            plt.title('Model hit ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, hit_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=hit_ratio.shape[1]-7, fontsize=7)
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
            label='Linear Regression')
                
            rects2 = plt.bar(index + bar_width, info_ratio.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='Support Vector Machines')
                   
            rect3 = plt.bar(index + 2*bar_width, info_ratio.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='Singular Value Decomposition')
              
            rects4 = plt.bar(index + 3*bar_width, info_ratio.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Decision Trees')

            rects5 = plt.bar(index + 4*bar_width, info_ratio.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects6 = plt.bar(index + 5*bar_width, info_ratio.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects7 = plt.bar(index + 6*bar_width, info_ratio.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 7*bar_width, info_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('info ratio')
            plt.title('Ensemble information ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, info_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=7+1, fontsize=7)
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

            rects10 = plt.bar(index + 9*bar_width, info_ratio.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('info ratio')
            plt.title('Model information ratios: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, info_ratio.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=info_ratio.shape[1]-7, fontsize=7)
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
            label='Linear Regression')
                
            rects2 = plt.bar(index + bar_width, ann_returns.loc[:,'SVM'], bar_width,
            alpha=opacity,
            color=[1.0, 0.5, 0.8],
            label='Support Vector Machines')
                   
            rect3 = plt.bar(index + 2*bar_width, ann_returns.loc[:,'SVD'], bar_width,
            alpha=opacity,
            color=[1.0, 0.8, 0.2],
            label='Singular Value Decomposition')
              
            rects4 = plt.bar(index + 3*bar_width, ann_returns.loc[:,'DT'], bar_width,
            alpha=opacity,
            color=[0.0, 1.0, 0.0],
            label='Decision Trees')

            rects5 = plt.bar(index + 4*bar_width, ann_returns.loc[:,'Ensemble'], bar_width,
            alpha=opacity,
            color=[0.0, 0.0, 1.0],
            label='Ensemble (max voting)')
    
            rects6 = plt.bar(index + 5*bar_width, ann_returns.loc[:,'Ensemble2'], bar_width,
            alpha=opacity,
            color=[0.5, 0.5, 1.0],
            label='Ensemble (simple avg.)')

            rects7 = plt.bar(index + 6*bar_width, ann_returns.loc[:,'Ensemble3'], bar_width,
            alpha=opacity,
            color=[0.5, 1.0, 1.0],
            label='Ensemble (weighted avg.)')

            rects8 = plt.bar(index + 7*bar_width, ann_returns.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='Long-Only')
    
            plt.xlabel('')
            plt.ylabel('ann. ret.')
            plt.title('Ensemble annualized returns: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, ann_returns.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=7+1, fontsize=7)
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

            rects10 = plt.bar(index + 9*bar_width, ann_returns.loc[:,'Long_Only'], bar_width,
            alpha=opacity,
            color=[1.0, 0.0, 0.0],
            label='LO')
    
            plt.xlabel('')
            plt.ylabel('ann. ret.')
            plt.title('Model annualized returns: ' + specs, fontsize=10)
            plt.suptitle(str(p))
            plt.xticks(index + bar_width, ann_returns.index)
            plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=ann_returns.shape[1]-7, fontsize=7)
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
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['OLS'],                   '-', color=[0.0, 0.0, 0.0], label='OLS')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Lasso'],                 '-', color=[1.0, 0.5, 0.8], label='Lasso')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ridge'],                 '-', color=[0.8, 0.2, 1.0], label='Ridge')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Correlation_Filter_F'],  '-', color=[0.4, 0.4, 0.4], label='C_filter_F')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Correlation_Filter_D'],  '-', color=[0.8, 0.8, 0.8], label='C_filter_D')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['PCR'],                   '-', color=[1.0, 0.8, 0.2], label='PCR')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['PLS'],                   '-', color=[1.0, 0.6, 0.0], label='PLS')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Random_Forest'],         '-', color=[0.4, 0.8, 0.6], label='RF')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Gradient_Boost'],        '-', color=[0.0, 1.0, 0.0], label='GB')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble'],              '-', color=[0.0, 0.0, 1.0], label='Ensemble')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble2'],             '-', color=[0.5, 0.5, 1.0], label='Ensemble2')
        plt.plot(self.strategy_eq[self.mergevar], self.strategy_eq['Ensemble3'],             '-', color=[0.5, 1.0, 1.0], label='Ensemble3')
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
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble'],              '-', color=[0.0, 0.0, 1.0], label='Ensemble')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble2'],             '-', color=[0.5, 0.5, 1.0], label='Ensemble2')
        plt.plot(self.strategy_iv[self.mergevar], self.strategy_iv['Ensemble3'],             '-', color=[0.5, 1.0, 1.0], label='Ensemble3')
        plt.title('iVol-weighted strategy: ' + specs, fontsize=10)
        plt.suptitle(strat_nick)
        plt.ylabel('performance')
        plt.xlabel('')
        plt.legend()
        pdf.savefig()
        
        pdf.close()    




import time
start = time.time()
test = regression_models(X = signal_fx.replace(np.nan, 0), Y = trade_co.replace(np.nan, 0).iloc[:,:4], freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])
test.fit()
end = time.time()
print('Time to execute:', round((end-start)/60), 'minutes')

test.ensemble()
test.performance_stats()


test7 = deepcopy(test6) # fitted on subsample
test = deepcopy(test7)

test=deepcopy(cne)
test.signal_weighting = True

test.ensemble = ensemble
test.ensemble() 

test.performance_stats = performance_stats
test.performance_stats() # current iteration
test2=deepcopy(test) # winner
test3=deepcopy(test) # no signal weighting

test.signal_weighting = False
test.performance_stats()

test.hit_ratio['Korea']
test.ann_returns['European Union']
test.info_ratio
test.hit_ratio_by_year
test.big_hit_by_year
test.info_ratio_by_year

test2.hit_ratio
test2.ann_returns['Canada']


test.beta_pcr
test.beta_pls
test.beta_rf

test.predict_pcr
test.predict_pls
test.predict_rf

test.tstats_pcr
test.X_test_scale
test.X_test_reduced

for p,q in zip(test.hit_ratio.values(),test.hit_ratio.keys()):
    print(q, round(p.loc['Ensemble'][0],2))
for p,q in zip(test2.hit_ratio.values(),test2.hit_ratio.keys()):
    print(q, round(p.loc['Ensemble'][0],2))
for p,q in zip(test.ann_returns.values(),test.ann_returns.keys()):
    print(q, round(p.loc['Ensemble'][0],3))
for p,q in zip(test2.ann_returns.values(),test2.ann_returns.keys()):
    print(q, round(p.loc['Ensemble'][0],3))

for p,q,r in zip(test.hit_ratio.values(), test2.hit_ratio.values(),test.hit_ratio.keys()):
    print(r, 'improved signal', round(p.loc['Ensemble'][0]-q.loc['Ensemble'][0],3))

for p,q,r in zip(test.ann_returns.values(), test2.ann_returns.values(),test.ann_returns.keys()):
    print(r, 'improved return', round(p.loc['Ensemble'][0]-q.loc['Ensemble'][0],3))



emfx_strat_eq = cne.strategy_eq
emfx_strat_iv = cne.strategy_iv

cne.charting_strat = charting_strat
cne.charting_strat(cne, strat_nick = "All EMFX", specs = "1Y lookback, daily returns, no signal weighting", fname = "CnE_3Ydaily_nw_all_strat.pdf", out_dir = out_dir)

cne.charting = charting
cne.charting(cne, specs = "1Y lookback, daily returns, no signal weighting", fname = "CnE_1Ydaily_nw.pdf", out_dir = out_dir)
  
        

   