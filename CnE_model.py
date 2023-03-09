# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:28:26 2020

@author: Jamal Khan
"""

'''
Chicken and Egg model:
    forecast FX returns using commodity returns
'''

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

#############################################################
# Run models -- CnE & EnC
#############################################################
'''
CnE
'''
start1 = time.time()

cne = regression_models(X = signal_co.replace(np.nan, 0), Y = trade_fx.replace(np.nan, 0), freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
cne.fit()

# generate ensemble signal
cne.ensemble()

# compute performance stats
cne.performance_stats()

# save model outputs in a dictionary
cne.output()

## save object so you don't have to fit it again and access all the object attributes any time by loading it back onto your wd

# save model object
with open(os.path.join(data_dir, 'cneModels_default.pickle'), 'wb') as output:
    pickle.dump(cne, output)

end1 = time.time()
print('Time to execute:', round((end1-start1)/60), 'minutes')

'''
EnC
'''
start2 = time.time()

enc = regression_models(X = signal_fx.replace(np.nan, 0), Y = trade_co.replace(np.nan, 0), freq = 'weekly', window = 52, 
                        datevars_x = ['date', 'signalDate'], datevars_y = ['date', 'tradeDate'], mergevar = ['date'])

# fit models
enc.fit()

# generate ensemble signal
enc.ensemble()

# compute performance stats
enc.performance_stats()

# save model outputs in a dictionary
enc.output()

# save model object
with open(os.path.join(data_dir, 'encModels_default.pickle'), 'wb') as output:
    pickle.dump(enc, output)

end2 = time.time()
print('Time to execute:', round((end2-start2)/60), 'minutes')

#############################################################
# Analyze models
#############################################################
# load model object
fname=os.path.join(data_dir, 'cneModels_default.pickle')
cne = pickle.load( open( fname, "rb" ) )

### access attributes

# performance metrics
cne.hit_ratio
cne.ann_returns
cne.info_ratio
cne.hit_ratio_by_year
cne.big_hit_by_year
cne.ann_returns_by_year
cne.info_ratio_by_year

# contruct strategies corresponding to a basket of currencies
dmfx = ['Australia','European Union','Canada','Switzerland','United Kingdom','Japan','Norway','New Zealand','Sweden']

emfx = ['Czech','Hungary','Poland','Romania',
        'South Africa','Israel','Russia','Turkey',
        'Korea','Indonesia','Philippines','Thailand','Taiwan','India','Singapore','China',
        'Brazil','Chile','Peru','Colombia','Mexico']

cne.construct_strategy(basket = emfx)

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


