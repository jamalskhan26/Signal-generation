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
# mobility data
#################

#google mobility
fname = os.path.join(data_dir, 'Google/Global_Mobility_Report.csv')
mobility_google = pd.read_csv(fname,skiprows=range(0))

mobility_google['geo_level']=''
mobility_google.loc[mobility_google['sub_region_1'].isnull(),'geo_level'] = 'country'
mobility_google.loc[mobility_google['sub_region_1'].notnull(),'geo_level'] = 'city_state'
mobility_google.loc[mobility_google['sub_region_2'].notnull(),'geo_level'] = 'county'

mobility_google.loc[mobility_google['geo_level']=='county','sub_region_2'] = mobility_google.loc[mobility_google['geo_level']=='county', 'sub_region_2'] + " " + mobility_google.loc[mobility_google['geo_level']=='county', 'sub_region_1']

mobility_google.loc[mobility_google['sub_region_1'].isnull(),'sub_region_1']=mobility_google.loc[mobility_google['sub_region_1'].isnull(),'country_region']
mobility_google.loc[mobility_google['sub_region_2'].isnull(),'sub_region_2']=mobility_google.loc[mobility_google['sub_region_2'].isnull(),'sub_region_1']
mobility_google.rename(columns={'sub_region_2':'location','country_region':'country'},inplace=True)
mobility_google.iloc[12345,:].T
mobility_google = mobility_google[['date','location','country','geo_level']+list(mobility_google.columns[-7:-1])]

mobility_google['google']=1
mobility_google['date'] = mobility_google['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

with open(os.path.join(data_dir, 'mobility_google.pickle'), 'wb') as output:
    pickle.dump(mobility_google, output)


#apple mobility
fname = os.path.join(data_dir, 'Apple/applemobilitytrends-2020-05-22.csv')
mobility_apple = pd.read_csv(fname,skiprows=range(0))
mobility_apple.loc[mobility_apple['geo_type']=='county','region'] = mobility_apple.loc[mobility_apple['geo_type']=='county', 'region'] + " " + mobility_apple.loc[mobility_apple['geo_type']=='county', 'sub-region']
mobility_apple = mobility_apple.drop(['alternative_name','sub-region'],axis=1) 
mobility_apple=pd.melt(mobility_apple,id_vars=['geo_type','region','country','transportation_type'],var_name='date', value_name='apple_mobility')
mobility_apple.rename(columns={'region':'location'},inplace=True)

mobility_apple['geo_level']=mobility_apple['geo_type']
mobility_apple.loc[mobility_apple['geo_level']=='city','geo_level'] = 'city'
mobility_apple.loc[mobility_apple['geo_level']=='county','geo_level'] = 'county'
mobility_apple.loc[mobility_apple['geo_level']=='sub-region','geo_level'] = 'state'
mobility_apple.loc[mobility_apple['geo_level']=='country/region','geo_level'] = 'country'
mobility_apple.head(1).T
mobility_apple = mobility_apple[['date','location','country','geo_level','transportation_type','apple_mobility']]

mobility_apple.loc[mobility_apple['country'].isna(),'country'] = ' '
mobility_apple = pd.pivot_table(mobility_apple, index=['date','geo_level','country','location'], columns=['transportation_type'], values=['apple_mobility'], fill_value=np.nan).reset_index()     

mobility_apple['apple']=1
mobility_apple['date'] = mobility_apple['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

mobility_apple.columns = mobility_apple.columns.get_level_values(0)
mobility_apple.columns = ['date', 'geo_level', 'country', 'location', 'driving', 'transit', 'walking', 'apple']

with open(os.path.join(data_dir, 'mobility_apple.pickle'), 'wb') as output:
    pickle.dump(mobility_apple, output)

#################
# case statistics
#################
    
    ## US data -- total/state/county level
fname = os.path.join(data_dir, 'NYTimes/us.csv')
cases_us = pd.read_csv(fname,skiprows=range(0))
fname = os.path.join(data_dir, 'NYTimes/us-states.csv')
cases_us_states = pd.read_csv(fname,skiprows=range(0))
fname = os.path.join(data_dir, 'NYTimes/us-counties.csv')
cases_us_county = pd.read_csv(fname,skiprows=range(0))

cases_us_county['county'] = cases_us_county['county'] + " " + cases_us_county['state']

cases_us['state']='0'
cases_us['county']='0'
cases_us['fips']=0
cases_us_states['county']='1'
cases_us = pd.concat([cases_us_county,cases_us_states,cases_us],axis=0)
cases_us = cases_us.sort_values(by=['date','state','county']).reset_index(drop=True)
cases_us['country']='United States'
cases_us=cases_us[['date','country','state','county','cases','deaths']]

cases_us['geo_level']='county'
cases_us.loc[cases_us['state']=='0','geo_level']='country'
cases_us.loc[cases_us['county']=='1','geo_level']='state'

cases_us.loc[cases_us['geo_level']=='country','county'] = 'United States'
cases_us.loc[cases_us['geo_level']=='state','county'] = cases_us.loc[cases_us['geo_level']=='state','state'] 
cases_us.rename(columns={'county':'location'},inplace=True)
cases_us['recovered'] = np.nan
## NOTE: add this data later
cases_us['Lat'] = np.nan
cases_us['Long'] = np.nan

    ## global 
fname = os.path.join(data_dir, 'JohnsHopkins/time_series_covid19_confirmed_global_narrow.csv')
cases_glo = pd.read_csv(fname).iloc[1:,:]
fname = os.path.join(data_dir, 'JohnsHopkins/time_series_covid19_deaths_global_narrow.csv')
deaths_glo = pd.read_csv(fname).iloc[1:,:]
fname = os.path.join(data_dir, 'JohnsHopkins/time_series_covid19_recovered_global_narrow.csv')
recovered_glo = pd.read_csv(fname).iloc[1:,:]

cases_global = pd.merge(cases_glo[['Date','Country/Region','Province/State','Lat','Long','Value']],deaths_glo[['Date','Province/State','Country/Region','Value']],on=['Date','Country/Region','Province/State'],how='left')
cases_global = pd.merge(cases_global,recovered_glo[['Date','Province/State','Country/Region','Value']],on=['Date','Country/Region','Province/State'],how='left')
cases_global.rename(columns={'Date':'date','Value_x':'cases','Value_y':'deaths','Value':'recovered'},inplace=True)

cases_global['geo_level']='state' 
cases_global.loc[cases_global['Province/State'].isnull(),'geo_level']='country' 
cases_global.loc[cases_global['Province/State'].isnull(),'Province/State']=cases_global.loc[cases_global['Province/State'].isnull(),'Country/Region']
cases_global.rename(columns={'Province/State':'location','Country/Region':'country'},inplace=True)
cases_global['state'] = cases_global['location'] 

cases = pd.concat([cases_global[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']], cases_us[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']]],axis=0)
# fill in US recovered numbers
cases.loc[(cases['country']=='United States') & (cases['geo_level']=='country'),'recovered'] = cases.loc[(cases['country']=='US'),'recovered'].tolist()

cases['date'] = cases['date'].apply(lambda x: dateutil.parser.parse(x))

cases = cases.sort_values(by=['date','country','state','location']).reset_index(drop=True)


##########################################################################################
## UPDATE FOLLOWING STRING MATCHING
##########################################################################################

cases.columns
cases[['Lat', 'Long', 'cases', 'deaths', 'recovered']] = cases[['Lat', 'Long', 'cases', 'deaths', 'recovered']].apply(pd.to_numeric, errors='ignore')

cases.loc[cases['country']=='United States', 'Lat'] = cases.loc[(cases['country']=='US') & (cases['geo_level']=='country'), 'Lat'].unique().astype(float).mean()
cases.loc[cases['country']=='United States', 'Long']  = cases.loc[(cases['country']=='US') & (cases['geo_level']=='country'), 'Long'].unique().astype(float).mean()

cases_aus = cases.loc[cases['country']=='Australia', ['date','cases','deaths','recovered']].reset_index(drop=True)
cases_aus=cases_aus.groupby(['date']).sum().reset_index()
cases_aus['location'] = 'Australia'
cases_aus['country'] = 'Australia'
cases_aus['state'] = ''
cases_aus['geo_level'] = 'country'
cases_aus['Lat'] = cases.loc[cases['country']=='Australia', 'Lat'].unique().astype(float).mean()
cases_aus['Long'] = cases.loc[cases['country']=='Australia', 'Long'].unique().astype(float).mean()

cases_can = cases.loc[cases['country']=='Canada', ['date','cases','deaths','recovered']].reset_index(drop=True)
cases_can=cases_can.groupby(['date']).sum().reset_index()
cases_can['location'] = 'Canada'
cases_can['country'] = 'Canada'
cases_can['state'] = ''
cases_can['geo_level'] = 'country'
cases_can['Lat'] = cases.loc[cases['country']=='Canada', 'Lat'].unique().astype(float).mean()
cases_can['Long'] = cases.loc[cases['country']=='Canada', 'Long'].unique().astype(float).mean()

cases_chi = cases.loc[cases['country']=='China', ['date','cases','deaths','recovered']].reset_index(drop=True)
cases_chi=cases_chi.groupby(['date']).sum().reset_index()
cases_chi['location'] = 'China'
cases_chi['country'] = 'China'
cases_chi['state'] = ''
cases_chi['geo_level'] = 'country'
cases_chi['Lat'] = cases.loc[cases['country']=='China', 'Lat'].unique().astype(float).mean()
cases_chi['Long'] = cases.loc[cases['country']=='China', 'Long'].unique().astype(float).mean()

cases.loc[cases['location']=='Hong Kong', 'geo_level'] = 'country'
cases.loc[cases['location']=='Hong Kong', 'country'] = 'Hong Kong'  

cases.loc[cases['country']=='Taiwan*','country'] = 'Taiwan'
cases.loc[cases['location']=='Taiwan*','location'] = 'Taiwan'
cases.loc[cases['state']=='Taiwan*','state'] = 'Taiwan'

cases = pd.concat([cases[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']],cases_aus[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']],cases_can[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']],cases_chi[['date','location','country','state','geo_level','Lat','Long','cases','deaths','recovered']]],axis=0)
cases = cases.sort_values(by=['date','country','state','location']).reset_index(drop=True)

# save
with open(os.path.join(data_dir, 'case_stats.pickle'), 'wb') as output:
    pickle.dump(cases, output)

writer=pd.ExcelWriter(os.path.join(data_dir,'case_stats.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'case_stats.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'case_stats.xlsx'),engine='openpyxl')
writer.book=book
cases.to_excel(writer,'case_stats')
writer.save()


'''
#################
# facebook symptom map: https://cmu-delphi.github.io/delphi-epidata/api/
#################
# Import
from delphi_epidata import Epidata
# Fetch data
fb = Epidata.covidcast('fb-survey', 'raw_cli', 'day', 'county', [20200401, Epidata.range(20200405, 20200414)], '06001')
print(fb['result'], fb['message'], len(fb['epidata']))




#############Ravenpack#####################
from ravenpackapi import RPApi
api = RPApi(api_key="WuiYytBtMJwvVuYYJzQkXX")
api.common_request_params.update(
        dict(
                proxies={'https': 'http://ZK463GK:XX@webproxy.bankofamerica.com:8080'},
                verify=False
            )
)

ds = api.get_dataset(dataset_id='us30')
ds.available_fields
'''