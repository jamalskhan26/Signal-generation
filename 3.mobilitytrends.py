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
import dateutil.parser
### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

#################
# load data
#################

fname=os.path.join(data_dir, 'mobility_google.pickle')
mobility_google = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'mobility_apple.pickle')
mobility_apple = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'case_stats.pickle')
cases = pickle.load( open( fname, "rb" ) )

#################
# merge
#################
# merge
def mergedat(dat,strdat,var):
    temp = strdat[['case_stats', var]].dropna().reset_index(drop=True)
    temp.columns = ['case_stats','location']
    dat = pd.merge(dat,temp,on='location',how='left')
    dat['location'] = dat['case_stats']
    dat = dat.drop(['case_stats'],axis=1).reset_index(drop=True)
    return dat

#countries
## region lists

### consolidate country names
fname = os.path.join(data_dir, 'case_mobility_match.xlsx')
country_list = pd.read_excel(fname,skiprows=range(0))
country_list.columns

country_goo = mobility_google[(mobility_google['geo_level']=='country')].reset_index(drop=True)
country_goo = mergedat(dat=country_goo, strdat=country_list, var='mobility_google')

country_app =  mobility_apple[(mobility_apple['geo_level']=='country')].reset_index(drop=True)
country_app = mergedat(dat=country_app, strdat=country_list, var='mobility_apple')

cases_country = pd.merge(cases[cases['geo_level']=='country'], country_goo, on=['date','location'], how='left', suffixes=('','_g'))
cases_country = pd.merge(cases_country, country_app, on=['date','location'], how='left',suffixes=('','_a'))
cases_country = cases_country[(cases_country['google']==1) | (cases_country['apple']==1)].reset_index(drop=True)

with open(os.path.join(data_dir, 'cases_country.pickle'), 'wb') as output:
    pickle.dump(cases_country, output)

#US states
state_goo = mobility_google[(mobility_google['geo_level']=='city_state') & (mobility_google['country']=='United States')].reset_index(drop=True)
state_app = mobility_apple[(mobility_apple['geo_level']=='state') & (mobility_apple['country']=='United States')].reset_index(drop=True)
cases_state = pd.merge(cases[(cases['geo_level']=='state') & (cases['country']=='United States')], state_goo, on=['date','location'], how='left', suffixes=('','_g'))
cases_state = pd.merge(cases_state, state_app, on=['date','location'], how='left',suffixes=('','_a'))
cases_state = cases_state[(cases_state['google']==1) | (cases_state['apple']==1)].reset_index(drop=True)

with open(os.path.join(data_dir, 'cases_state.pickle'), 'wb') as output:
    pickle.dump(cases_state, output)

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country_state.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'cases_country_state.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country_state.xlsx'),engine='openpyxl')
writer.book=book
cases_country.to_excel(writer,'country')
cases_state.to_excel(writer,'state')
writer.save()
'''
##############################
#counties
##############################

# county population
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility/Other'
fname = os.path.join(data_dir, 'countypopulation.xlsx')
county_pop = pd.read_excel(fname,skiprows=range(0))
county_pop['CTYNAME'] = county_pop['CTYNAME'] + " " + county_pop['STNAME']  
county_pop = county_pop.iloc[:,1:]
county_pop.columns = ['location','county_pop'] 

county_goo = mobility_google[(mobility_google['geo_level']=='county') & (mobility_google['country']=='United States')].reset_index(drop=True)
county_app = mobility_apple[(mobility_apple['geo_level']=='county')].reset_index(drop=True)

## region lists
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'
fname = os.path.join(data_dir, 'county_match.xlsx')
counties = pd.read_excel(fname,skiprows=range(0))
counties.columns

county_pop = mergedat(dat=county_pop,strdat=counties,var='county_pop')
county_goo = mergedat(dat=county_goo,strdat=counties,var='mobility_google')
county_app = mergedat(dat=county_app,strdat=counties,var='mobility_apple')

county_pop = county_pop[county_pop['location'].notna()].drop_duplicates(subset=['location']).reset_index(drop=True)
county_goo = county_goo[county_goo['location'].notna()].drop_duplicates(subset=['date','location']).reset_index(drop=True)
county_app = county_app[county_app['location'].notna()].drop_duplicates(subset=['date','location']).reset_index(drop=True)

cases_county = cases[(cases['geo_level']=='county') & (cases['country']=='United States')].reset_index(drop=True)
cases_county = cases_county[cases_county['location'].isin(counties['case_stats'])].reset_index(drop=True)

cases_county = pd.merge(cases_county, county_goo, on=['date','location'], how='left', suffixes=('','_g'))
cases_county = pd.merge(cases_county, county_app, on=['date','location'], how='left',suffixes=('','_a'))
cases_county = pd.merge(cases_county, county_pop, on=['location'], how='left',suffixes=('','_p'))
cases_county = cases_county[(cases_county['google']==1) | (cases_county['apple']==1)].reset_index(drop=True)

#rename
cases_county.rename(columns = {'retail_and_recreation_percent_change_from_baseline' : 'retail_recreation',
                                'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery_pharmacy',
                                'parks_percent_change_from_baseline' : 'parks',
                                'transit_stations_percent_change_from_baseline' : 'transit_stations',
                                'workplaces_percent_change_from_baseline' : 'workplaces',
                                'residential_percent_change_from_baseline' : 'residential'},
                    inplace=True)

#standardize mobility
cases_county.loc[cases_county['retail_recreation'].notna(),'retail_recreation'] += 100 
cases_county.loc[cases_county['grocery_pharmacy'].notna(),'grocery_pharmacy'] += 100 
cases_county.loc[cases_county['parks'].notna(),'parks'] += 100 
cases_county.loc[cases_county['transit_stations'].notna(),'transit_stations'] += 100 
cases_county.loc[cases_county['workplaces'].notna(),'workplaces'] += 100 
cases_county.loc[cases_county['residential'].notna(),'residential'] += 100 

with open(os.path.join(data_dir, 'cases_county.pickle'), 'wb') as output:
    pickle.dump(cases_county, output)

