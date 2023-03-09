# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:45:02 2020

@author: zk463gk
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

#############################################################
# Load/import data
#############################################################
import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
import datetime as dt
from datetime import timedelta  
from eqd import bbg
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import DataPrepFunctions as fns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels import RandomEffects
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

#################
# import data
#################

fname = os.path.join(data_dir, 'BAC/cardspending.xlsx')
bac = pd.read_excel(fname,skiprows=range(1))
bac.iloc[:,2:] = bac.iloc[:,2:]*100

fname=os.path.join(data_dir, 'cases_state.pickle')
cases_state = pickle.load( open( fname, "rb" ) )

mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving']

fname = os.path.join(data_dir, 'Other/state_abb.xlsx')
state_abb = pd.read_excel(fname,skiprows=range(0))
state_abb.columns = ['state','location'] 

mob_state = cases_state[['date','location']+mobility_vars_a+mobility_vars_g]
mob_state = pd.merge(mob_state, state_abb, on = 'location', how = 'left')
mob_state.columns

spending = pd.merge(bac, mob_state, on = ['state','date'], how = 'left')

len(bac.dropna()['state'].unique())
len(spending.dropna()['location'].unique())

spending['month'] = spending['date'].apply(lambda x: x.month)

spending_summary = spending.groupby(['state','month']).mean().reset_index()
spending_top10 = spending_summary.loc[spending_summary['state'].isin(['CA','TX','FL','NY','PA','IL','OH','GA','NC','MI'])]

writer=pd.ExcelWriter(os.path.join(data_dir, 'spending_top10.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'spending_top10.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'spending_top10.xlsx'),engine='openpyxl')
writer.book=book    
spending_top10.to_excel(writer,'spending_top10')
writer.save()



###############GRAPH#######################

os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility/BAC/Charts')
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d-%b')

###############AUTO#######################

pdf = matplotlib.backends.backend_pdf.PdfPages("Auto_1wrolling.pdf")
for p in spending['location'].dropna().unique():
    temp = spending[spending['location'] == p]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel('card spending on auto (1w rolling)',color='b')
    ax1.plot(temp['date'], temp['Auto2020'].rolling(7).mean(), '-', label='auto spending (1w rolling)',color='b')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis   
    ax2.set_ylabel('driving mobility (1w rolling)',color='g')  # we already handled the x-label with ax1
    ax2.plot(temp['date'], temp['driving'].rolling(7).mean(), '-', label='driving mobility (1w rolling)',color='g')
    ax2.tick_params(axis='y')    

    ax1.xaxis.set_major_formatter(myFmt)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Driving mobility and BAC card spending on Auto \n', fontsize=10)
    plt.suptitle(p, fontsize=10)
   
    pdf.savefig()
pdf.close()

###############GAS#######################

pdf = matplotlib.backends.backend_pdf.PdfPages("Gas_1wrolling.pdf")
for p in spending['location'].dropna().unique():
    temp = spending[spending['location'] == p]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel('card spending on gas (1w rolling)',color='b')
    ax1.plot(temp['date'], temp['Gas2020'].rolling(7).mean(), '-', label='gas spending (1w rolling)',color='b')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis   
    ax2.set_ylabel('driving mobility (1w rolling)',color='g')  # we already handled the x-label with ax1
    ax2.plot(temp['date'], temp['driving'].rolling(7).mean(), '-', label='driving mobility (1w rolling)',color='g')
    ax2.tick_params(axis='y')    

    ax1.xaxis.set_major_formatter(myFmt)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Driving mobility and BAC card spending on Gas \n', fontsize=10)
    plt.suptitle(p, fontsize=10)
   
    pdf.savefig()
pdf.close()


###############TRANSIT#######################

pdf = matplotlib.backends.backend_pdf.PdfPages("Transit_1wrolling.pdf")
for p in spending['location'].dropna().unique():
    temp = spending[spending['location'] == p]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel('card spending on transit (1w rolling)',color='b')
    ax1.plot(temp['date'], temp['Transit2020'].rolling(7).mean(), '-', label='transit spending (1w rolling)',color='b')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis   
    ax2.set_ylabel('transit mobility (1w rolling)',color='g')  # we already handled the x-label with ax1
    ax2.plot(temp['date'], temp['transit_stations'].rolling(7).mean(), '-', label='transit mobility (1w rolling)',color='g')
    ax2.tick_params(axis='y')    

    ax1.xaxis.set_major_formatter(myFmt)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Transit mobility and BAC card spending on Transit \n', fontsize=10)
    plt.suptitle(p, fontsize=10)
   
    pdf.savefig()
pdf.close()



pdf = matplotlib.backends.backend_pdf.PdfPages("Auto.pdf")
for p in spending['location'].dropna().unique():
    temp = spending[spending['location'] == p]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('')
    ax1.set_ylabel('card spending on auto (1w rolling)',color='b')
    ax1.plot(temp['date'], temp['Auto2020'], '-', label='auto spending (1w rolling)',color='b')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis   
    ax2.set_ylabel('driving mobility (1w rolling)',color='g')  # we already handled the x-label with ax1
    ax2.plot(temp['date'], temp['driving'], '-', label='driving mobility (1w rolling)',color='g')
    ax2.tick_params(axis='y')    

    ax1.xaxis.set_major_formatter(myFmt)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Driving mobility and BAC card spending on Auto \n', fontsize=10)
    plt.suptitle(p, fontsize=10)
   
    pdf.savefig()
pdf.close()

