# -*- coding: utf-8 -*-
from IPython import get_ipython
get_ipython().magic('reset -sf')

#############################################################
# Load/import data
#############################################################
import pandas as pd
import numpy as np
import os
import pickle
import datetime as dt

### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility/Other'

#################
# import data
#################

# median income
fname = os.path.join(data_dir, 'median_income_us.xlsx')
income = pd.read_excel(fname,skiprows=range(0))
income = income[['Geographic Area Name','median income']] 
income.head(1)
income.columns = ['location','median_income']


# popover65
fname = os.path.join(data_dir, 'PopOver65_us.xlsx')
popover65 = pd.read_excel(fname,skiprows=range(0))
popover65 = popover65[['Geographic Area Name','median age', 'over65']] 
popover65.head(1)
popover65.columns = ['location','median_age','over65']

# household size
fname = os.path.join(data_dir, 'US_household_size.xlsx')
household_size = pd.read_excel(fname,skiprows=range(0))
household_size = household_size[['Geographic Area Name','average household size']] 
household_size.head(1)
household_size.columns = ['location','household_size']

# pop density
fname = os.path.join(data_dir, 'us_pop_density.csv')
USpop = pd.read_csv(fname,skiprows=range(0))
USpop = USpop[['State','Density','Pop']] 
USpop.head(1)
USpop.columns = ['location','pop_density','pop']

# testing
fname = os.path.join(data_dir, 'ustesting.xlsx')
UStesting = pd.read_excel(fname,skiprows=range(0))
UStesting = UStesting[['date','state','total']] 
UStesting.head(1)
UStesting.columns = ['date','location','tests']

fname = os.path.join(data_dir, 'state_abb.xlsx')
state_abb = pd.read_excel(fname,skiprows=range(0))

UStesting = pd.merge(state_abb,UStesting,on='location',how='outer')
UStesting = UStesting.drop(['location'],axis=1) 
UStesting = UStesting.rename(columns = {'state':'location'})
UStesting['date'] = UStesting['date'].apply(lambda x: dt.datetime(year=int(str(x)[0:4]), month=int(str(x)[4:6]), day=int(str(x)[6:8])))


#################
# merge
#################
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

fname=os.path.join(data_dir, 'cases_state.pickle')
cases_state = pickle.load( open( fname, "rb" ) )

_0 = pd.DataFrame(cases_state['location'].unique())
_0.columns = ['cases_state']

_1 = pd.DataFrame(USpop['location'].unique())
_1.columns = ['USpop']

_2 = pd.DataFrame(UStesting['location'].unique())
_2.columns = ['UStesting']

_3 = pd.DataFrame(household_size['location'].unique())
_3.columns = ['household_size']

fname=os.path.join(data_dir, 'mobility_google.pickle')
mobility_google = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'mobility_apple.pickle')
mobility_apple = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'case_stats.pickle')
cases = pickle.load( open( fname, "rb" ) )

_4 = pd.DataFrame(mobility_google.loc[((mobility_google['country']=='United States') & (mobility_google['geo_level']=='city_state')),'location'].unique())
_4.columns = ['mobility_google']

_5 = pd.DataFrame(mobility_apple.loc[(mobility_apple['geo_level']=='state'),'location'].unique())
_5.columns = ['mobility_apple']

_6 = pd.DataFrame(cases.loc[(cases['country']=='United States') & (cases['geo_level']=='state'),'location'].unique())
_6.columns = ['case_stats']

state_list = pd.concat([_1,_2,_3,_4,_5,_6],axis=1)

writer=pd.ExcelWriter(os.path.join(data_dir,'state_list.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'state_list.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'state_list.xlsx'),engine='openpyxl')
writer.book=book
_0.to_excel(writer,'cases')
state_list.to_excel(writer,'state_list')
writer.save()

#################
# merge data
#################
# country master list
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'
fname = os.path.join(data_dir, 'state_match.xlsx')
states = pd.read_excel(fname,skiprows=range(0))
states.columns

fname=os.path.join(data_dir, 'cases_state.pickle')
cases_state = pickle.load( open( fname, "rb" ) )

# merge
def mergedat(dat,var):
    temp = states[['cases_state', var]].dropna().reset_index(drop=True)
    temp.columns = ['cases_state','location']
    dat = pd.merge(dat,temp,on='location',how='left')
    dat['location'] = dat['cases_state']
    dat = dat.drop(['cases_state'],axis=1).dropna().reset_index(drop=True)
    return dat

USpop = mergedat(dat=USpop,var='USpop')
UStesting = mergedat(dat=UStesting,var='UStesting')
household_size = mergedat(dat=household_size,var='household_size')
income = mergedat(dat=income,var='median_income')
popover65 = mergedat(dat=popover65,var='over65')

def mergevars(dat1,dat2,var):
    return pd.merge(dat1,dat2,on=var,how='left')

cases_state = mergevars(cases_state, USpop, ['location'])
cases_state = mergevars(cases_state, household_size, ['location'])
cases_state = mergevars(cases_state, UStesting, ['date','location'])
cases_state = mergevars(cases_state, income, ['location'])
cases_state = mergevars(cases_state, popover65, ['location'])

#fill missing
cases_state.loc[cases_state['tests'].isna(),'tests'] = 0
#rename
cases_state.rename(columns = {'retail_and_recreation_percent_change_from_baseline' : 'retail_recreation',
                                'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery_pharmacy',
                                'parks_percent_change_from_baseline' : 'parks',
                                'transit_stations_percent_change_from_baseline' : 'transit_stations',
                                'workplaces_percent_change_from_baseline' : 'workplaces',
                                'residential_percent_change_from_baseline' : 'residential'},
                    inplace=True)

#standardize mobility
cases_state.loc[cases_state['retail_recreation'].notna(),'retail_recreation'] += 100 
cases_state.loc[cases_state['grocery_pharmacy'].notna(),'grocery_pharmacy'] += 100 
cases_state.loc[cases_state['parks'].notna(),'parks'] += 100 
cases_state.loc[cases_state['transit_stations'].notna(),'transit_stations'] += 100 
cases_state.loc[cases_state['workplaces'].notna(),'workplaces'] += 100 
cases_state.loc[cases_state['residential'].notna(),'residential'] += 100 

with open(os.path.join(data_dir, 'cases_state.pickle'), 'wb') as output:
    pickle.dump(cases_state, output)