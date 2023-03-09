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
#region string matching
#################

# match google & apple regions on case data

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
from ftfy import fix_text

def ngrams(string, n=1):
    string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

#tfidf of cases
case_names = cases['location'].sort_values().unique()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tf_idf_matrix = vectorizer.fit_transform(case_names)
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tf_idf_matrix)

###matching query:
def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices

# 1. match: apple on case names
app_names = set(mobility_apple['location'].sort_values().unique()) # set used for increased performance

import time
t1 = time.time()
print('getting nearest n...')
distances, indices = getNearestN(app_names )
t = time.time()-t1
print("COMPLETED IN:", t)

app_names = list(app_names) #need to convert back to a list
print('finding matches...')
matches = []
for i,j in enumerate(indices):
  temp = [round(distances[i][0],2), case_names[j][0],app_names[i]]
  matches.append(temp)

print('Building data frame...')  
app_case_strings = pd.DataFrame(matches, columns=['match_score','matched_name','original_name'])
print('Done')

# 2. match: google on case names
goo_names = set(mobility_google['location'].sort_values().unique()) # set used for increased performance

import time
t1 = time.time()
print('getting nearest n...')
distances, indices = getNearestN(goo_names)
t = time.time()-t1
print("COMPLETED IN:", t)

goo_names = list(goo_names) #need to convert back to a list
print('finding matches...')
matches = []
for i,j in enumerate(indices):
  temp = [round(distances[i][0],2), case_names[j][0],goo_names[i]]
  matches.append(temp)

print('Building data frame...')  
goo_case_strings = pd.DataFrame(matches, columns=['match_score','matched_name','original_name'])
print('Done')


app_case_strings = pd.merge(cases[['location','country','state','geo_level']].drop_duplicates(),app_case_strings,left_on='location',right_on='matched_name',how='left')
goo_case_strings = pd.merge(cases[['location','country','state','geo_level']].drop_duplicates(),goo_case_strings,left_on='location',right_on='matched_name',how='left')

#save
with open(os.path.join(data_dir, 'app_case_strings.pickle'), 'wb') as output:
    pickle.dump(app_case_strings, output)
with open(os.path.join(data_dir, 'goo_case_strings.pickle'), 'wb') as output:
    pickle.dump(goo_case_strings, output)

writer=pd.ExcelWriter(os.path.join(data_dir,'apple_google_stringMatch.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'apple_google_stringMatch.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'apple_google_stringMatch.xlsx'),engine='openpyxl')
writer.book=book
app_case_strings.to_excel(writer,'apple')
goo_case_strings.to_excel(writer,'google')
writer.save()

'''
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

for i in clean_strings.index:
    for j in clean_strings.columns:
        clean_strings.loc[i,j] = fuzz.ratio(i,j)

with open(os.path.join(data_dir, 'clean_region.pickle'), 'wb') as output:
    pickle.dump(clean_strings, output)
'''

''' country-level notes notes:
in case data - sum china, canada, australia states at country level
taiwan - remove *
hong kong > change to country
manual (google) - burma, Cote d'Ivoire, the bahamas
manual (apple) - korea, uk

***UPDATE DATAPREP FILE

'''



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

# contact tracing
fname = os.path.join(data_dir, 'covid-contact-tracing.csv')
contact_tracing = pd.read_csv(fname,skiprows=range(0))
contact_tracing = contact_tracing[['Date','Entity','Contact tracing (OxBSG)']] 
contact_tracing.columns = ['date','location','contact_tracing']
contact_tracing['date'] = contact_tracing['date'].apply(lambda x: dt.datetime.strptime(x, '%b %d, %Y'))

# testing
fname = os.path.join(data_dir, 'full-list-total-tests-for-covid-19.csv')
testing = pd.read_csv(fname,skiprows=range(0))
testing = testing[['Date','Entity','Total tests']] 
testing.columns = ['date','location','total_tests']
testing['date'] = testing['date'].apply(lambda x: dt.datetime.strptime(x, '%b %d, %Y'))

# public gathering rules
fname = os.path.join(data_dir, 'public-gathering-rules-covid.csv')
public_rules = pd.read_csv(fname,skiprows=range(0))
public_rules = public_rules[['Date','Entity','Restrictions on gatherings (OxBSG)']]
public_rules.columns = ['date','location','public_rules']
public_rules['date'] = public_rules['date'].apply(lambda x: dt.datetime.strptime(x, '%b %d, %Y'))

# gdp per cap
fname = os.path.join(data_dir, 'gdppc.xlsx')
gdppc = pd.read_excel(fname,skiprows=range(0))
temp=gdppc.T.iloc[2:,:].fillna(method='ffill').tail(1).reset_index(drop=True).T
gdppc = pd.concat([gdppc['Country Name'],temp],axis=1).dropna().reset_index(drop=True)
gdppc.columns = ['location','gdppc']

# healthcare access
fname = os.path.join(data_dir, 'healthcare-access-and-quality-index.csv')
healthcare = pd.read_csv(fname,skiprows=range(0))
healthcare = healthcare[healthcare['Year']==2015].reset_index(drop=True) 
healthcare = healthcare[['Entity','HAQ Index (IHME (2017))']]
healthcare.columns = ['location','health_access']

# diabetes
fname = os.path.join(data_dir, 'diabetes.xlsx')
diabetes = pd.read_excel(fname,skiprows=range(0))
diabetes.columns = ['location','diabetes'] 

# GCI
fname = os.path.join(data_dir, 'GCI.xlsx')
gci = pd.read_excel(fname,skiprows=range(0))
gci.columns = ['location','gci'] 

# obesity
fname = os.path.join(data_dir, 'obesity.xlsx')
obesity = pd.read_excel(fname,skiprows=range(0))
obesity.columns = ['location','obesity'] 

# median age
fname = os.path.join(data_dir, 'medianage.xlsx')
age = pd.read_excel(fname,skiprows=range(1))
age.columns = ['location','median_age'] 

# population over 65
fname = os.path.join(data_dir, 'PopOver65.xlsx')
pop65 = pd.read_excel(fname,skiprows=range(0))
temp=pop65.T.iloc[2:,:].fillna(method='ffill').tail(1).reset_index(drop=True).T
pop65 = pd.concat([pop65['Country Name'],temp],axis=1).dropna().reset_index(drop=True)
pop65.columns = ['location','pop65'] 

# population level+density
fname = os.path.join(data_dir, 'population.xlsx')
pop = pd.read_excel(fname,sheet_name='country',skiprows=range(0))
pop = pop[['Country', 'per sq mile', 'population', 'Region']]
pop.columns = ['location', 'pop_per_sq_m', 'pop', 'region']

#urban population
fname = os.path.join(data_dir, 'urban.xlsx')
urban = pd.read_excel(fname,skiprows=range(0))
temp=urban.T.iloc[2:,:].fillna(method='ffill').tail(1).reset_index(drop=True).T
urban = pd.concat([urban['Country Name'],temp],axis=1).dropna().reset_index(drop=True)
urban.columns = ['location','urban'] 

#healthcare per capita
fname = os.path.join(data_dir, 'healthcare_pc.xlsx')
healthcare_pc = pd.read_excel(fname,skiprows=range(0))
temp=healthcare_pc.T.iloc[2:,:].fillna(method='ffill').tail(1).reset_index(drop=True).T
healthcare_pc = pd.concat([healthcare_pc['Country Name'],temp],axis=1).dropna().reset_index(drop=True)
healthcare_pc.columns = ['location','healthcare_pc'] 

#oop expenditure
fname = os.path.join(data_dir, 'oop.xlsx')
oop = pd.read_excel(fname,skiprows=range(0))
temp=oop.T.iloc[2:,:].fillna(method='ffill').tail(1).reset_index(drop=True).T
oop = pd.concat([oop['Country Name'],temp],axis=1).dropna().reset_index(drop=True)
oop.columns = ['location','oop'] 


#################
# merge
#################
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

fname=os.path.join(data_dir, 'cases_country.pickle')
cases_country = pickle.load( open( fname, "rb" ) )

_0 = pd.DataFrame(cases_country['location'].unique())
_0.columns = ['cases_country']

_1 = pd.DataFrame(contact_tracing['location'].unique())
_1.columns = ['contact_tracing']

_2 = pd.DataFrame(testing['location'].unique())
_2.columns = ['testing']

_3 = pd.DataFrame(public_rules['location'].unique())
_3.columns = ['public_rules']

_4 = pd.DataFrame(gdppc['location'].unique())
_4.columns = ['gdppc']

_5 = pd.DataFrame(healthcare['location'].unique())
_5.columns = ['healthcare']

_6 = pd.DataFrame(age['location'].unique())
_6.columns = ['age']

_7 = pd.DataFrame(pop65['location'].unique())
_7.columns = ['pop65']

_8 = pd.DataFrame(pop['location'].unique())
_8.columns = ['pop']

_9 = pd.DataFrame(urban['location'].unique())
_9.columns = ['urban']

_10 = pd.DataFrame(healthcare_pc['location'].unique())
_10.columns = ['healthcare_pc']

_11 = pd.DataFrame(oop['location'].unique())
_11.columns = ['oop']

_12 = pd.DataFrame(diabetes['location'].unique())
_12.columns = ['diabetes']

_13 = pd.DataFrame(obesity['location'].unique())
_13.columns = ['obesity']

_14 = pd.DataFrame(gci['location'].unique())
_14.columns = ['gci']

fname=os.path.join(data_dir, 'mobility_google.pickle')
mobility_google = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'mobility_apple.pickle')
mobility_apple = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'case_stats.pickle')
cases = pickle.load( open( fname, "rb" ) )

_15 = pd.DataFrame(mobility_google.loc[(mobility_google['geo_level']=='country'),'location'].unique())
_15.columns = ['mobility_google']

_16 = pd.DataFrame(mobility_apple.loc[(mobility_apple['geo_level']=='country'),'location'].unique())
_16.columns = ['mobility_apple']

_17 = pd.DataFrame(cases.loc[(cases['geo_level']=='country'),'location'].unique())
_17.columns = ['case_stats']

country_list = pd.concat([_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17],axis=1)

writer=pd.ExcelWriter(os.path.join(data_dir,'country_list.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'country_list.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'country_list.xlsx'),engine='openpyxl')
writer.book=book
_0.to_excel(writer,'cases')
country_list.to_excel(writer,'country_list')
writer.save()

#################
# merge data
#################
# country master list
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'
fname = os.path.join(data_dir, 'country_match.xlsx')
countries = pd.read_excel(fname,skiprows=range(0))
countries.columns

fname=os.path.join(data_dir, 'cases_country.pickle')
cases_country = pickle.load( open( fname, "rb" ) )

# merge
def mergedat(dat,var):
    temp = countries[['cases_country', var]].dropna().reset_index(drop=True)
    temp.columns = ['cases_country','location']
    dat = pd.merge(dat,temp,on='location',how='left')
    dat['location'] = dat['cases_country']
    dat = dat.drop(['cases_country'],axis=1).dropna().reset_index(drop=True)
    return dat

age = mergedat(dat=age,var='age')
contact_tracing = mergedat(dat=contact_tracing,var='contact_tracing')
gdppc = mergedat(dat=gdppc,var='gdppc')
healthcare = mergedat(dat=healthcare,var='healthcare')
pop = mergedat(dat=pop,var='pop')
pop65 = mergedat(dat=pop65,var='pop65')
public_rules = mergedat(dat=public_rules,var='public_rules')
testing = mergedat(dat=testing,var='testing')
urban = mergedat(dat=urban,var='urban')
healthcare_pc = mergedat(dat=healthcare_pc,var='healthcare_pc')
oop = mergedat(dat=oop,var='oop')
diabetes = mergedat(dat=diabetes,var='diabetes')
obesity = mergedat(dat=obesity,var='obesity')
gci = mergedat(dat=gci,var='gci')

def mergevars(dat1,dat2,var):
    return pd.merge(dat1,dat2,on=var,how='left')

cases_country = mergevars(cases_country, contact_tracing, ['date','location'])
cases_country = mergevars(cases_country, testing, ['date','location'])
cases_country = mergevars(cases_country, public_rules, ['date','location'])
cases_country = mergevars(cases_country, age, ['location'])
cases_country = mergevars(cases_country, gdppc, ['location'])
cases_country = mergevars(cases_country, pop, ['location'])
cases_country = mergevars(cases_country, pop65, ['location'])
cases_country = mergevars(cases_country, urban, ['location'])
cases_country = mergevars(cases_country, healthcare, ['location'])
cases_country = mergevars(cases_country, healthcare_pc, ['location'])
cases_country = mergevars(cases_country, oop, ['location'])
cases_country = mergevars(cases_country, diabetes, ['location'])
cases_country = mergevars(cases_country, obesity, ['location'])
cases_country = mergevars(cases_country, gci, ['location'])

#fill missing
cases_country.loc[cases_country['total_tests'].isna(),'total_tests'] = 0.
cases_country.loc[cases_country['contact_tracing'].isna(),'contact_tracing'] = 1.0
cases_country.loc[cases_country['public_rules'].isna(),'public_rules'] = 1.0
cases_country.loc[cases_country['region'].isna(),'region'] = 'Asia'
#rename
cases_country.rename(columns = {'retail_and_recreation_percent_change_from_baseline' : 'retail_recreation',
                                'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery_pharmacy',
                                'parks_percent_change_from_baseline' : 'parks',
                                'transit_stations_percent_change_from_baseline' : 'transit_stations',
                                'workplaces_percent_change_from_baseline' : 'workplaces',
                                'residential_percent_change_from_baseline' : 'residential'},
                    inplace=True)

#standardize mobility
cases_country.loc[cases_country['retail_recreation'].notna(),'retail_recreation'] += 100 
cases_country.loc[cases_country['grocery_pharmacy'].notna(),'grocery_pharmacy'] += 100 
cases_country.loc[cases_country['parks'].notna(),'parks'] += 100 
cases_country.loc[cases_country['transit_stations'].notna(),'transit_stations'] += 100 
cases_country.loc[cases_country['workplaces'].notna(),'workplaces'] += 100 
cases_country.loc[cases_country['residential'].notna(),'residential'] += 100 

with open(os.path.join(data_dir, 'cases_country.pickle'), 'wb') as output:
    pickle.dump(cases_country, output)

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country_state.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'cases_country_state.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country_state.xlsx'),engine='openpyxl')
writer.book=book
cases_country.to_excel(writer,'country')
writer.save()
'''

temp = cases_country[(cases_country['country']=='United States') & (cases_country['geo_level']=='country')].reset_index()

writer=pd.ExcelWriter(os.path.join(data_dir,'us_aggregate_ts.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'us_aggregate_ts.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'us_aggregate_ts.xlsx'),engine='openpyxl')
writer.book=book
temp.to_excel(writer,'us')
writer.save()



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


### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

#################
# import data
#################

fname=os.path.join(data_dir, 'cases_country.pickle')
cases_country = pickle.load( open( fname, "rb" ) )

mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving','transit','walking']
control_vars = ['median_age','gdppc','pop_per_sq_m','pop','pop65','urban','health_access','healthcare_pc','oop','diabetes','obesity','gci','contact_tracing','total_tests','public_rules','Lat','Long']
lagged_dep = ['case_growth_1w']

stdev = cases_country[['location','median_age','gdppc','pop_per_sq_m','pop','pop65','urban','health_access','healthcare_pc','oop','diabetes','obesity','gci']].drop_duplicates()
stdev = stdev.iloc[:,1:].std() 

writer=pd.ExcelWriter(os.path.join(data_dir, 'stdev_controls.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'stdev_controls.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'stdev_controls.xlsx'),engine='openpyxl')
writer.book=book    
stdev.to_excel(writer,'stdev_controls')
writer.save()


#################
# data prep
#################

#transformations
cases_country['case_rate'] = cases_country['cases'] / cases_country['pop'] * 1000 
# death / recovery rates as of cases two weeks ago
cases_country['cases_2w'] =  cases_country[['location','cases']].groupby('location').shift(13)
cases_country['death_rate'] = cases_country['deaths'] / cases_country['cases']
cases_country['recovery_rate'] = cases_country['recovered'] / cases_country['cases']
cases_country['cases_1w'] =  cases_country[['location','cases']].groupby('location').shift(8)
cases_country['case_growth_1w'] = (cases_country['cases'] / cases_country['cases_1w'])-1
cases_country['case_growth_1w'] = cases_country[['location','case_growth_1w']].groupby('location').shift(1)
cases_country['case_growth_1w'] = cases_country['case_growth_1w'].replace(np.inf,np.nan).replace(-np.inf,np.nan)

country_data_des = cases_country.describe()
country_data_des['countries'] = str(cases_country.location.unique())
country_data_des['N'] = len(cases_country.location.unique())
country_data_des['T'] = len(cases_country.date.unique())
country_data_des = country_data_des.T

writer=pd.ExcelWriter(os.path.join(data_dir, 'countryleveldescription.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'countryleveldescription.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'countryleveldescription.xlsx'),engine='openpyxl')
writer.book=book    
country_data_des.to_excel(writer,'des')
writer.save()


top10_cases = ['United States','United Kingdom','Germany'	,'France'	,'Italy','Spain',	'Brazil',	'Russia','India','Turkey','Mexico']
top10_cases_em = ['Brazil',	'Russia',	'India',	'Turkey',	'Iran',	'Peru',	'Saudi Arabia',	'Chile',	'Mexico',	'Pakistan']
bestresponse_cases = ['Korea, South', 'Singapore',	'Hong Kong','Australia','New Zealand','Taiwan','Finland','Denmark','Belgium','Sweden']

top10_dat = cases_country[cases_country['location'].isin(top10_cases)].reset_index(drop=True)
top10_em_dat = cases_country[cases_country['location'].isin(top10_cases_em)].reset_index(drop=True)
best_dat = cases_country[cases_country['location'].isin(bestresponse_cases)].reset_index(drop=True)

writer=pd.ExcelWriter(os.path.join(data_dir, 'country_mob_trends_top10.xlsx'))
writer.save() 
for mob in mobility_vars_g+mobility_vars_a:
    temp = pd.pivot_table(top10_dat[['date','location']+[mob]], index=['date'], columns=['location'], values=[mob], fill_value=np.nan).dropna(how='all')
    temp.iloc[7:,:] = temp.rolling(7).mean().iloc[7:,:]
    book= load_workbook(os.path.join(data_dir, 'country_mob_trends_top10.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'country_mob_trends_top10.xlsx'),engine='openpyxl')
    writer.book=book    
    temp.to_excel(writer,mob)
    writer.save()

writer=pd.ExcelWriter(os.path.join(data_dir, 'country_mob_trends_best10.xlsx'))
writer.save() 
for mob in mobility_vars_g+mobility_vars_a:
    temp = pd.pivot_table(best_dat[['date','location']+[mob]], index=['date'], columns=['location'], values=[mob], fill_value=np.nan).dropna(how='all')
    temp.iloc[7:,:] = temp.rolling(7).mean().iloc[7:,:]
    book= load_workbook(os.path.join(data_dir, 'country_mob_trends_best10.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'country_mob_trends_best10.xlsx'),engine='openpyxl')
    writer.book=book    
    temp.to_excel(writer,mob)
    writer.save()


#change in case rates, pct change in mobility
for c in cases_country['location'].unique():
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'cases_d'] = (cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'cases']).diff().values
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'cases_ld'] = np.log(cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'cases']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'case_rate_ld'] = np.log(cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'case_rate']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'death_rate_d'] = cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'death_rate'].diff().values
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'death_rate_ld'] = np.log(cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'death_rate']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'recovery_rate_d'] = cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),'recovery_rate'].diff().values
#    cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'), mobility_vars_g+mobility_vars_a] = (cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),mobility_vars_g+mobility_vars_a]).div((cases_country.loc[(cases_country['location']==c) & (cases_country['geo_level']=='country'),mobility_vars_g+mobility_vars_a]).shift(1).values)

# run VIF test
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = cases_country[['median_age','gdppc','pop_per_sq_m','pop','pop65','urban','health_access','healthcare_pc','oop','diabetes','obesity','gci','contact_tracing','total_tests','public_rules','Lat','Long']].replace(np.inf,np.nan).replace(-np.inf,np.nan)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.dropna().values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

#remove median age, pop, gdppc
control_vars = ['const','pop_per_sq_m','pop65','urban','health_access','healthcare_pc','oop','diabetes','obesity','gci','contact_tracing','total_tests','public_rules','Lat','Long']

#################
## regressions
#################

import sys

# shift mobility vars 
for i in range(1, 10, 1):
    lagged = cases_country[['location'] + mobility_vars_g+mobility_vars_a].groupby('location').shift(i)
    lagged.columns = [s + ('_' + str(i)) for s in list(lagged.columns)]
    colnames = list(cases_country.columns) + list(lagged.columns)
    cases_country = pd.concat([cases_country,lagged],axis=1, ignore_index=True)
    cases_country.columns = colnames

cases_country_testing = cases_country[cases_country['total_tests']!=0].reset_index(drop=True)

# set index for panel reg
cases_country['countrycode'] = pd.Categorical(cases_country.location).codes
cases_country = cases_country.set_index(['date', 'countrycode'])
cases_country['const']  = 1

cases_country_testing['countrycode'] = pd.Categorical(cases_country_testing.location).codes
cases_country_testing = cases_country_testing.set_index(['date', 'countrycode'])
cases_country_testing['const']  = 1

'''
#to use later for testing
temp = cases_country.copy()
temp.index = temp['date']
split_index = round(len(temp.index.unique())*0.8)
split_date = temp.index.unique()[split_index]
df_train = temp.loc[temp.index <= split_date].copy()
df_test = temp.loc[temp.index > split_date].copy()
'''

'''
# determine optimal lag length
aic_df = {}
r2_fe = {}
r2_re = {}
lag_models_fe = {}
lag_models_re = {}
#Current minimum AIC score
min_aic = sys.float_info.max
#Current max R2
max_r2 = sys.float_info.min
#Lag for the best model seen so far
best_lag_aic = ''
best_lag_r2 = ''
best_mod_r2 = ''
#OLSResults objects for the best model seen so far
best_olsr_model_results = None

depvar_ = 'cases_ld'

#Run through each lag var   
for lag_num in range(0, 10):
    if lag_num == 0:
        colnames = mobility_vars_g+mobility_vars_a
    else:
        colnames = [s + ('_' + str(lag_num)) for s in mobility_vars_g+mobility_vars_a]
    
    print('Building model for lag: ' + str(lag_num))
    
    #Build and fit the OLSR model 
    olsr_results = sm.OLS(cases_country[depvar_], cases_country[colnames + control_vars + lagged_dep], missing = 'drop').fit()
    #Store away the model's AIC score and R2 
    aic_df[lag_num] = olsr_results.aic
    print('AIC='+str(aic_df[lag_num]))

   #If the model's AIC score is less than the current minimum score, update the current minimum AIC score and the current best model
    if olsr_results.aic < min_aic:
        min_aic = olsr_results.aic
        best_lag_aic = str(lag_num)

##Build and fit  fixed and random effects models 
    mod = PanelOLS(cases_country[depvar_], cases_country[colnames + control_vars + lagged_dep]).fit()
    r2_fe[lag_num] = mod.rsquared_overall
    lag_models_fe[lag_num] = mod

   #If the model's R2 is greater than the current max, update the current max and the current best model
    if mod.rsquared_overall > max_r2:
        max_r2 = mod.rsquared_overall
        best_lag_r2 = str(lag_num)
        best_mod_r2 = 'fe'

    mod = RandomEffects(cases_country[depvar_], cases_country[colnames + control_vars + lagged_dep]).fit()
    r2_re[lag_num] = mod.rsquared_overall    
    lag_models_re[lag_num] = mod

   #If the model's R2 is greater than the current max, update the current max and the current best model
    if mod.rsquared_overall > max_r2:
        max_r2 = mod.rsquared_overall
        best_lag_r2 = str(lag_num)
        best_mod_r2 = 're'
'''

#######################################################################

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'cases_country.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country.xlsx'),engine='openpyxl')
writer.book=book
cases_country.to_excel(writer,'country')
writer.save()

writer=pd.ExcelWriter(os.path.join(data_dir,'correl.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'correl.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'correl.xlsx'),engine='openpyxl')
writer.book=book
cases_country[mobility_vars_g+mobility_vars_a+control_vars].corr().to_excel(writer,'exog_corr')
writer.save()
'''

############################
# iterative models for mobility
############################

dep_vars = ['case_rate_ld','death_rate_ld']

mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving','transit','walking']
mobility_vars_g_7 = [s + ('_' + str(7)) for s in mobility_vars_g]
mobility_vars_a_7 = [s + ('_' + str(7)) for s in mobility_vars_a]

def run_reg(dat,dep_var,mob_g,mob_a):
    reg_results = {}
    for mob_var in (mob_g+mob_a):
        mod = PanelOLS(dat[dep_var], dat[control_vars + lagged_dep +[mob_var]]).fit()
        temp = mod.summary.tables[1].as_html()
        reg_results[mob_var] = pd.read_html(temp, header=0, index_col=0)[0]            
        reg_results[mob_var].loc['R2_within','Parameter'] =  mod.rsquared_within
        reg_results[mob_var].loc['R2_between','Parameter'] = mod.rsquared_between
        reg_results[mob_var].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(dat[dep_var], dat[control_vars + lagged_dep +mob_g+mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['all'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['all'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['all'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['all'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(dat[dep_var], dat[control_vars + lagged_dep +mob_g]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['google'] = pd.read_html(temp, header=0, index_col=0)[0]
    reg_results['google'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['google'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['google'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(dat[dep_var], dat[control_vars + lagged_dep +mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['apple'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['apple'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['apple'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['apple'].loc['R2_overall','Parameter'] = mod.rsquared_overall
            
    return reg_results

reg_cases =    run_reg(dat = cases_country, dep_var = dep_vars[0], mob_g = mobility_vars_g, mob_a = mobility_vars_a)
reg_cases_6 =  run_reg(dat = cases_country, dep_var = dep_vars[0], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)

writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases.xlsx'))
writer.save() 
for p in reg_cases.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases[p].to_excel(writer,p)
    writer.save()
for p in reg_cases_6.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases_6[p].to_excel(writer,p)
    writer.save()

reg_deaths =  run_reg(dat = cases_country, dep_var = dep_vars[1], mob_g = mobility_vars_g, mob_a = mobility_vars_a)
reg_deaths_6 =  run_reg(dat = cases_country, dep_var = dep_vars[1], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)

writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_deaths.xlsx'))
writer.save() 
for p in reg_deaths.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_deaths.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_deaths.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_deaths[p].to_excel(writer,p)
    writer.save()
for p in reg_deaths_6.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_deaths.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_deaths.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_deaths_6[p].to_excel(writer,p)
    writer.save()


####################################################################################

def run_reg_re(dat,dep_var,mob_g,mob_a):
    reg_results = {}
    for mob_var in (mob_g+mob_a):
        mod = RandomEffects(dat[dep_var], dat[control_vars + lagged_dep +[mob_var]]).fit()
        temp = mod.summary.tables[1].as_html()
        reg_results[mob_var] = pd.read_html(temp, header=0, index_col=0)[0]            
        reg_results[mob_var].loc['R2_within','Parameter'] =  mod.rsquared_within
        reg_results[mob_var].loc['R2_between','Parameter'] = mod.rsquared_between
        reg_results[mob_var].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = RandomEffects(dat[dep_var], dat[control_vars + lagged_dep +mob_g+mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['all'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['all'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['all'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['all'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = RandomEffects(dat[dep_var], dat[control_vars + lagged_dep +mob_g]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['google'] = pd.read_html(temp, header=0, index_col=0)[0]
    reg_results['google'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['google'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['google'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = RandomEffects(dat[dep_var], dat[control_vars + lagged_dep +mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['apple'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['apple'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['apple'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['apple'].loc['R2_overall','Parameter'] = mod.rsquared_overall
            
    return reg_results

reg_cases =    run_reg_re(dat = cases_country, dep_var = dep_vars[0], mob_g = mobility_vars_g, mob_a = mobility_vars_a)
reg_cases_6 =  run_reg_re(dat = cases_country, dep_var = dep_vars[0], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)

writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_re.xlsx'))
writer.save() 
for p in reg_cases.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases_re.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_re.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases[p].to_excel(writer,p)
    writer.save()
for p in reg_cases_6.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases_re.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_re.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases_6[p].to_excel(writer,p)
    writer.save()

####################################################################################

top10_cases = ['United States','United Kingdom','Germany'	,'France'	,'Italy','Spain',	'Brazil',	'Russia','India','Turkey']
top10_cases_em = ['Brazil',	'Russia',	'India',	'Turkey',	'Iran',	'Peru',	'Saudi Arabia',	'Chile',	'Mexico',	'Pakistan']
bestresponse_cases = ['Korea, South', 'Singapore',	'Hong Kong','Australia','New Zealand','Taiwan','Finland','Denmark','Belgium','Sweden']

top10_dat = cases_country[cases_country['location'].isin(top10_cases)].reset_index()
top10_em_dat = cases_country[cases_country['location'].isin(top10_cases_em)].reset_index()
best_dat = cases_country[cases_country['location'].isin(bestresponse_cases)].reset_index()

# set index for panel reg
top10_dat['countrycode'] = pd.Categorical(top10_dat.location).codes
top10_dat = top10_dat.set_index(['date', 'countrycode'])
top10_dat['const']  = 1

# set index for panel reg
best_dat['countrycode'] = pd.Categorical(best_dat.location).codes
best_dat = best_dat.set_index(['date', 'countrycode'])
best_dat['const']  = 1

top10_dat['location'].unique()
best_dat['location'].unique()

reg_cases_top10 =  run_reg(dat = top10_dat, dep_var = dep_vars[0], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)
reg_cases_best =  run_reg(dat = best_dat, dep_var = dep_vars[0], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)

####################################################################################

reg_cases_testing = run_reg(dat = cases_country_testing, dep_var = dep_vars[0], mob_g = mobility_vars_g, mob_a = mobility_vars_a)

writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_testing.xlsx'))
writer.save() 
for p in reg_cases_testing .keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases_testing.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_testing.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases_testing[p].to_excel(writer,p)
    writer.save()


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

### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

#################
# import data
#################

fname=os.path.join(data_dir, 'cases_state.pickle')
cases_state = pickle.load( open( fname, "rb" ) )

state_data_des = cases_state.describe()
state_data_des['N'] = len(cases_state.location.unique())
state_data_des['T'] = len(cases_state.date.unique())
state_data_des = state_data_des.T

writer=pd.ExcelWriter(os.path.join(data_dir, 'stateleveldescription.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'stateleveldescription.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'stateleveldescription.xlsx'),engine='openpyxl')
writer.book=book    
state_data_des.to_excel(writer,'des')
writer.save()


mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving']
control_vars = ['over65','median_income','median_age','pop_density','pop','household_size','tests']
lagged_dep = ['case_growth_1w']
#################
# data prep
#################

#transformations
cases_state['case_rate'] = cases_state['cases'] / cases_state['pop'] * 1000 
# death / recovery rates as of cases two weeks ago
cases_state['cases_2w'] =  cases_state[['location','cases']].groupby('location').shift(13)
cases_state['death_rate'] = cases_state['deaths'] / cases_state['cases_2w']
cases_state['cases_1w'] =  cases_state[['location','cases']].groupby('location').shift(8)
cases_state['case_growth_1w'] = (cases_state['cases'] / cases_state['cases_1w'])-1
cases_state['case_growth_1w'] = cases_state[['location','case_growth_1w']].groupby('location').shift(1)

#change in case rates, pct change in mobility
for c in cases_state['location'].unique():
    cases_state.loc[(cases_state['location']==c),'cases_d'] = (cases_state.loc[(cases_state['location']==c),'cases']).diff().values
    cases_state.loc[(cases_state['location']==c),'cases_ld'] = np.log(cases_state.loc[(cases_state['location']==c),'cases']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_state.loc[(cases_state['location']==c),'case_rate_ld'] = np.log(cases_state.loc[(cases_state['location']==c),'case_rate']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_state.loc[(cases_state['location']==c),'death_rate_d'] = cases_state.loc[(cases_state['location']==c),'death_rate'].diff().values
    cases_state.loc[(cases_state['location']==c),'death_rate_ld'] = np.log(cases_state.loc[(cases_state['location']==c),'death_rate']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
#    cases_state.loc[(cases_state['location']==c), mobility_vars_g+mobility_vars_a] = np.log(cases_state.loc[(cases_state['location']==c),mobility_vars_g+mobility_vars_a]).diff().values

# run VIF test
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = cases_state[control_vars].replace(np.inf,np.nan).replace(-np.inf,np.nan)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.dropna().values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

#remove median age, pop, gdppc
control_vars = ['const','over65','median_income','median_age','pop_density','household_size','tests']

#################
## regressions
#################

import sys

# shift mobility vars 
for i in range(1, 15, 1):
    lagged = cases_state[['location'] + mobility_vars_g+mobility_vars_a].groupby('location').shift(i)
    lagged.columns = [s + ('_' + str(i)) for s in list(lagged.columns)]
    colnames = list(cases_state.columns) + list(lagged.columns)
    cases_state = pd.concat([cases_state,lagged],axis=1, ignore_index=True)
    cases_state.columns = colnames

# set index for panel reg
cases_state['statecode'] = pd.Categorical(cases_state.location).codes
cases_state = cases_state.set_index(['date', 'statecode'])
cases_state['const']  = 1

'''
#to use later for testing
temp = cases_country.copy()
temp.index = temp['date']
split_index = round(len(temp.index.unique())*0.8)
split_date = temp.index.unique()[split_index]
df_train = temp.loc[temp.index <= split_date].copy()
df_test = temp.loc[temp.index > split_date].copy()
'''
'''
# determine optimal lag length
aic_df = {}
r2_fe = {}
r2_re = {}
lag_models_fe = {}
lag_models_re = {}
#Current minimum AIC score
min_aic = sys.float_info.max
#Current max R2
max_r2 = sys.float_info.min
#Lag for the best model seen so far
best_lag_aic = ''
best_lag_r2 = ''
best_mod_r2 = ''
#OLSResults objects for the best model seen so far
best_olsr_model_results = None

depvar_ = 'cases_ld'

#Run through each lag var   
for lag_num in range(0, 15):
    if lag_num == 0:
        colnames = mobility_vars_g+mobility_vars_a
    else:
        colnames = [s + ('_' + str(lag_num)) for s in mobility_vars_g+mobility_vars_a]
    
    print('Building model for lag: ' + str(lag_num))
    
    #Build and fit the OLSR model 
    olsr_results = sm.OLS(cases_state[depvar_], cases_state[colnames + control_vars + lagged_dep], missing = 'drop').fit()
    #Store away the model's AIC score and R2 
    aic_df[lag_num] = olsr_results.aic
    print('AIC='+str(aic_df[lag_num]))

   #If the model's AIC score is less than the current minimum score, update the current minimum AIC score and the current best model
    if olsr_results.aic < min_aic:
        min_aic = olsr_results.aic
        best_lag_aic = str(lag_num)

##Build and fit  fixed and random effects models 
    mod = PanelOLS(cases_state[depvar_], cases_state[colnames + control_vars + lagged_dep]).fit()
    r2_fe[lag_num] = mod.rsquared_overall
    lag_models_fe[lag_num] = mod

   #If the model's R2 is greater than the current max, update the current max and the current best model
    if mod.rsquared_overall > max_r2:
        max_r2 = mod.rsquared_overall
        best_lag_r2 = str(lag_num)
        best_mod_r2 = 'fe'

    mod = RandomEffects(cases_state[depvar_], cases_state[colnames + control_vars + lagged_dep]).fit()
    r2_re[lag_num] = mod.rsquared_overall    
    lag_models_re[lag_num] = mod

   #If the model's R2 is greater than the current max, update the current max and the current best model
    if mod.rsquared_overall > max_r2:
        max_r2 = mod.rsquared_overall
        best_lag_r2 = str(lag_num)
        best_mod_r2 = 're'
'''

#######################################################################

'''
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'cases_country.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'cases_country.xlsx'),engine='openpyxl')
writer.book=book
cases_country.to_excel(writer,'country')
writer.save()

writer=pd.ExcelWriter(os.path.join(data_dir,'correl.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir,'correl.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir,'correl.xlsx'),engine='openpyxl')
writer.book=book
cases_country[mobility_vars_g+mobility_vars_a+control_vars].corr().to_excel(writer,'exog_corr')
writer.save()
'''

############################
# iterative models for mobility
############################

dep_vars = ['case_rate_ld','death_rate_ld']

mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving']
mobility_vars_g_7 = [s + ('_' + str(7)) for s in mobility_vars_g]
mobility_vars_a_7 = [s + ('_' + str(7)) for s in mobility_vars_a]

def run_reg(dep_var,mob_g,mob_a):
    reg_results = {}
    for mob_var in (mob_g+mob_a):
        mod = PanelOLS(cases_state[dep_var], cases_state[lagged_dep+control_vars+[mob_var]]).fit()
        temp = mod.summary.tables[1].as_html()
        reg_results[mob_var] = pd.read_html(temp, header=0, index_col=0)[0]           
        reg_results[mob_var].loc['R2_within','Parameter'] =  mod.rsquared_within
        reg_results[mob_var].loc['R2_between','Parameter'] = mod.rsquared_between
        reg_results[mob_var].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(cases_state[dep_var], cases_state[lagged_dep+control_vars+mob_g+mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['all'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['all'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['all'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['all'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(cases_state[dep_var], cases_state[lagged_dep+control_vars+mob_g]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['google'] = pd.read_html(temp, header=0, index_col=0)[0]
    reg_results['google'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['google'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['google'].loc['R2_overall','Parameter'] = mod.rsquared_overall

    mod = PanelOLS(cases_state[dep_var], cases_state[lagged_dep+control_vars+mob_a]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results['apple'] = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results['apple'].loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results['apple'].loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results['apple'].loc['R2_overall','Parameter'] = mod.rsquared_overall
            
    return reg_results

reg_cases =      run_reg(dep_var = dep_vars[0], mob_g = mobility_vars_g, mob_a = mobility_vars_a)
reg_cases_6 =  run_reg(dep_var = dep_vars[0], mob_g = mobility_vars_g_7, mob_a = mobility_vars_a_7)

mobility_vars_a_g = ['transit_stations','driving','residential']
mobility_vars_a_g_7 = [s + ('_' + str(7)) for s in mobility_vars_a_g]

def run_reg_custom(dep_var,mob):
    mod = PanelOLS(cases_state[dep_var], cases_state[lagged_dep+control_vars+mob]).fit()
    temp = mod.summary.tables[1].as_html()
    reg_results = pd.read_html(temp, header=0, index_col=0)[0]            
    reg_results.loc['R2_within','Parameter'] = mod.rsquared_within
    reg_results.loc['R2_between','Parameter'] = mod.rsquared_between
    reg_results.loc['R2_overall','Parameter'] = mod.rsquared_overall
    return reg_results

reg_cases['driving_transit'] = run_reg_custom(dep_vars[0],mobility_vars_a_g)
reg_cases['driving_transit1'] = run_reg_custom(dep_vars[0],mobility_vars_a_g_7)

writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_states.xlsx'))
writer.save() 
for p in reg_cases.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases_states.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_states.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases[p].to_excel(writer,p)
    writer.save()
for p in reg_cases_6.keys(): 
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'reg_cases_states.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'reg_cases_states.xlsx'),engine='openpyxl')
    writer.book=book    
    reg_cases_6[p].to_excel(writer,p)
    writer.save()

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
import statsmodels.api as sm
import datetime as dt
from datetime import timedelta  
from eqd import bbg
os.chdir('//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal')
import DataPrepFunctions as fns
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels import RandomEffects

### set paths 
### !!UPDATE!! before running
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

#################
# import data
#################

fname=os.path.join(data_dir, 'cases_county.pickle')
cases_county = pickle.load( open( fname, "rb" ) )

cases_county_des = cases_county.describe()
cases_county_des['N'] = len(cases_county.location.unique())
cases_county_des['T'] = len(cases_county.date.unique())
cases_county_des = cases_county_des.T

writer=pd.ExcelWriter(os.path.join(data_dir, 'countyleveldescription.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'countyleveldescription.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'countyleveldescription.xlsx'),engine='openpyxl')
writer.book=book    
cases_county_des.to_excel(writer,'des')
writer.save()


mobility_vars_g = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential']
mobility_vars_a = ['driving']

#transformations

cases_county['case_rate'] = cases_county['cases'] / cases_county['county_pop'] * 1000 

# death / recovery rates / case growth
cases_county['cases_2w'] =  cases_county[['location','cases']].groupby('location').shift(14)
cases_county['death_rate'] = cases_county['deaths'] / cases_county['cases']
cases_county['recovery_rate'] = cases_county['recovered'] / cases_county['cases']
cases_county['cases_1w'] =  cases_county[['location','cases']].groupby('location').shift(7)
cases_county['case_growth_1w'] = (np.log(cases_county['cases']) - np.log(cases_county['cases_1w'])).replace(np.inf,np.nan).replace(-np.inf,np.nan)
cases_county['case_growth_1w'] = cases_county[['location','case_growth_1w']].groupby('location').shift(1)

county_data_des = cases_county.describe()
county_data_des['countries'] = str(cases_county.location.unique())
county_data_des['N'] = len(cases_county.location.unique())
county_data_des['T'] = len(cases_county.date.unique())
county_data_des = county_data_des.T

writer=pd.ExcelWriter(os.path.join(data_dir, 'countyleveldescription.xlsx'))
writer.save() 
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'countyleveldescription.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'countyleveldescription.xlsx'),engine='openpyxl')
writer.book=book    
county_data_des.to_excel(writer,'des')
writer.save()


for c in cases_county['location'].unique():
    cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'cases_d'] = (cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'cases']).diff().values
    cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'case_rate_ld'] = np.log(cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'case_rate']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values
    cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'deaths_d'] = (cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'deaths']).diff().values
    cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'death_rate_ld'] = np.log(cases_county.loc[(cases_county['location']==c) & (cases_county['geo_level']=='county'),'deaths']).replace(np.inf,np.nan).replace(-np.inf,np.nan).diff().values


with open(os.path.join(data_dir, 'cases_county.pickle'), 'wb') as output:
    pickle.dump(cases_county, output)

#dim 1
county_summary = cases_county.groupby('location')[['location','case_rate']].tail(1).reset_index(drop=True)

#dim 2
temp = cases_county.groupby('location')[['location','case_rate_ld']].tail(30).reset_index(drop=True)
#temp = temp.groupby('location')[['location','case_rate_ld']].mean().reset_index()
temp = temp.groupby('location')[['location','case_rate_ld']].apply(lambda x: x.ewm(alpha=0.1, adjust=False).mean()).reset_index(drop=True)
temp = temp.groupby('location')[['location','case_rate_ld']].tail(1).reset_index(drop=True)

county_summary = pd.merge(county_summary,temp,on='location',how='left')

#dim 3
temp = cases_county.groupby('location')[['location','death_rate']].tail(1).reset_index(drop=True)
county_summary = pd.merge(county_summary,temp,on='location',how='left')

#dim 4
temp = cases_county.groupby('location')[['location','death_rate_ld']].tail(30).reset_index(drop=True)
#temp = temp.groupby('location')[['location','death_rate_ld']].mean().reset_index()
temp = temp.groupby('location')[['location','death_rate_ld']].apply(lambda x: x.ewm(alpha=0.1, adjust=False).mean()).reset_index(drop=True)
temp = temp.groupby('location')[['location','death_rate_ld']].tail(1).reset_index(drop=True)

county_summary = pd.merge(county_summary,temp,on='location',how='left')


#import FIPS
data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'
fname = os.path.join(data_dir, 'county_match.xlsx')
fips = pd.read_excel(fname,skiprows=range(0))
fips.columns
fips = fips[['case_stats','FIPS']]
fips.columns = ['location','FIPS']
fips['FIPS'] = fips['FIPS'].map(lambda x: str(x).lstrip("'"))

#county_summary = county_summary.drop(['FIPS'],axis=1) 
county_summary = pd.merge(county_summary, fips, on ='location', how = 'left')    
    
with open(os.path.join(data_dir, 'county_summary.pickle'), 'wb') as output:
    pickle.dump(county_summary, output)

##############DRIVING

fname=os.path.join(data_dir, 'county_summary.pickle')
county_summary = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'cases_county.pickle')
cases_county = pickle.load( open( fname, "rb" ) )


driving_county = cases_county[['date','location','geo_level','apple','driving']]
driving_county = driving_county[driving_county['apple']==1].reset_index(drop=True)

driving_county_1 = driving_county[(driving_county['date']>='2020-03-01') & (driving_county['date']<='2020-04-10')].reset_index(drop=True)
driving_county_2 = driving_county[(driving_county['date']>'2020-04-10')].reset_index(drop=True)

driving_county_avg_1 = pd.DataFrame()
for c in driving_county_1['location'].unique():
    driving_county_avg_1.loc[c,'average_driving'] = (driving_county_1.loc[(driving_county_1['location']==c) & (driving_county_1['geo_level']=='county'),'driving']).mean()
driving_county_avg_1 = driving_county_avg_1.reset_index()
driving_county_avg_1.columns = ['location','driving'] 
driving_county_avg_1 = pd.merge(driving_county_avg_1,county_summary[['location','FIPS']],on='location',how='left')
with open(os.path.join(data_dir, 'driving_county_avg_1.pickle'), 'wb') as output:
    pickle.dump(driving_county_avg_1, output)

driving_county_avg_2 = pd.DataFrame()
for c in driving_county_2['location'].unique():
    driving_county_avg_2.loc[c,'average_driving'] = (driving_county_2.loc[(driving_county_1['location']==c) & (driving_county_2['geo_level']=='county'),'driving']).mean()
driving_county_avg_2 = driving_county_avg_2.reset_index()
driving_county_avg_2.columns = ['location','driving'] 
driving_county_avg_2 = pd.merge(driving_county_avg_2,county_summary[['location','FIPS']],on='location',how='left')
with open(os.path.join(data_dir, 'driving_county_avg_2.pickle'), 'wb') as output:
    pickle.dump(driving_county_avg_2, output)

#######################################################

    
'''
#K-means

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
'''

import matplotlib.pyplot as plt

clusterdata = county_summary.dropna().reset_index(drop=True) 
#remove outliers
from scipy import stats
clusterdata['z_score_cases']=stats.zscore(clusterdata['case_rate_ld'])
clusterdata = clusterdata.loc[clusterdata['z_score_cases'].abs()<=3].reset_index(drop=True)
clusterdata['z_score_deaths']=stats.zscore(clusterdata['death_rate_ld'])
clusterdata = clusterdata.loc[clusterdata['z_score_deaths'].abs()<=3].reset_index(drop=True)

X = clusterdata[['case_rate_ld','death_rate_ld']]
y = clusterdata['location']


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=9, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
plt.scatter(X.iloc[:,1],X.iloc[:,0], c=cluster.labels_, cmap='rainbow')

len(cluster.labels_)
cluster_labels = pd.concat([clusterdata,pd.DataFrame(cluster.labels_)],axis=1)
cluster_labels.rename(columns = {0:'cluster'},inplace=True)

#temp = cluster_labels[cluster_labels[0]==0]['location'].reset_index(drop=True)
#temp['cluster'] = '0' 

#cases_county = cases_county.drop(['cluster'],axis=1)
cases_county = pd.merge(cases_county,cluster_labels[['location','cluster']],on='location',how='left')

with open(os.path.join(data_dir, 'cases_county.pickle'), 'wb') as output:
    pickle.dump(cases_county, output)

with open(os.path.join(data_dir, 'cluster_labels.pickle'), 'wb') as output:
    pickle.dump(cluster_labels, output)


'''
from sklearn.cluster import KMeans
plt.scatter(X['death_rate_ld'],X['case_rate_ld'])

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''
'''
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=1000, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

plt.scatter(clusterdata['death_rate_ld'],clusterdata['case_rate_ld'])
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c='red')
plt.show()

from collections import Counter, defaultdict
print(Counter(kmeans.labels_))

clusters_indices = defaultdict(list)
for index, c  in enumerate(kmeans.labels_):
    clusters_indices[c].append(index)
    
print(clusters_indices[0])    


cluster1 = clusterdata.iloc[clusters_indices[0],:].reset_index(drop=True)
'''

###########################RF FEATURE SELECTION#################################### 

from sklearn.ensemble import RandomForestRegressor

mob_features_case = pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g)
mob_features_death= pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g) 

for c in cases_county['location'].unique():
    ## CHANGE IN CASE RATE
    temp = cases_county.loc[(cases_county['location']==c),['case_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:]
    y = temp.iloc[:,0]
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 100,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_case.loc[c,:] = rff.feature_importances_
    except:
        continue    
    ## CHANGE IN DEATH RATE
    temp = cases_county.loc[(cases_county['location']==c),['death_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:]
    y = temp.iloc[:,0]
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 100,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_death.loc[c,:] = rff.feature_importances_
    except:
        continue    

with open(os.path.join(data_dir, 'mob_features_case.pickle'), 'wb') as output:
    pickle.dump(mob_features_case, output)
with open(os.path.join(data_dir, 'mob_features_death.pickle'), 'wb') as output:
    pickle.dump(mob_features_death, output)
    
#######################################

from sklearn.ensemble import RandomForestRegressor
mob_features_case_lag = pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g)
mob_features_death_lag= pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g) 

for c in cases_county['location'].unique():
    ## CHANGE IN CASE RATE
    temp = cases_county.loc[(cases_county['location']==c),['case_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(7).iloc[7:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[7:].reset_index(drop=True)
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 1000,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_case_lag.loc[c,:] = rff.feature_importances_
    except:
        continue    
    ## CHANGE IN DEATH RATE
    temp = cases_county.loc[(cases_county['location']==c),['death_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(7).iloc[7:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[7:].reset_index(drop=True)
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 1000,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_death_lag.loc[c,:] = rff.feature_importances_
    except:
        continue    

with open(os.path.join(data_dir, 'mob_features_case_lag.pickle'), 'wb') as output:
    pickle.dump(mob_features_case_lag, output)
with open(os.path.join(data_dir, 'mob_features_death_lag.pickle'), 'wb') as output:
    pickle.dump(mob_features_death_lag, output)    

#######################################

mob_features_case_lag_2 = pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g)
mob_features_death_lag_2= pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g) 

for c in cases_county['location'].unique():
    ## CHANGE IN CASE RATE
    temp = cases_county.loc[(cases_county['location']==c),['case_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(14).iloc[14:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[14:].reset_index(drop=True)
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 100,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_case_lag_2.loc[c,:] = rff.feature_importances_
    except:
        continue    
    ## CHANGE IN DEATH RATE
    temp = cases_county.loc[(cases_county['location']==c),['death_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(14).iloc[14:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[14:].reset_index(drop=True)
    # Create a random forest classifier
    rff = RandomForestRegressor(n_estimators = 100,random_state = 0, n_jobs = -1)   
    # Train the classifier
    try:
        rff.fit(X, y)
        # get feature importance
        mob_features_death_lag_2.loc[c,:] = rff.feature_importances_
    except:
        continue    

with open(os.path.join(data_dir, 'mob_features_case_lag_2.pickle'), 'wb') as output:
    pickle.dump(mob_features_case_lag_2, output)
with open(os.path.join(data_dir, 'mob_features_death_lag_2.pickle'), 'wb') as output:
    pickle.dump(mob_features_death_lag_2, output)    
    

########################### RIDGE #################################### 

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = {'alpha' : [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

l_mob_features_case_lag = pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g)
l_mob_features_death_lag= pd.DataFrame(index=cases_county['location'].unique(),columns=mobility_vars_g) 

for c in cases_county['location'].unique():
    ## CHANGE IN CASE RATE
    temp = cases_county.loc[(cases_county['location']==c),['case_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(7).iloc[7:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[7:].reset_index(drop=True)
    # Create a ridge classifier
    ridge_reg = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)   
    # Train the classifier
    try:
        ridge_reg.fit(X, y)
        # get feature importance
        l_mob_features_case_lag.loc[c,:] = ridge_reg.best_estimator_.coef_
    except:
        continue    
    ## CHANGE IN DEATH RATE
    temp = cases_county.loc[(cases_county['location']==c),['death_rate_ld']+mobility_vars_g].dropna().reset_index(drop=True)
    X = temp.iloc[:,1:].shift(7).iloc[7:,:].reset_index(drop=True)
    y = temp.iloc[:,0].iloc[7:].reset_index(drop=True)

    ridge_reg = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)   
    # Train the classifier
    try:
        ridge_reg.fit(X, y)
        # get feature importance
        l_mob_features_death_lag.loc[c,:] = ridge_reg.best_estimator_.coef_
    except:
        continue    

with open(os.path.join(data_dir, 'l_mob_features_case_lag.pickle'), 'wb') as output:
    pickle.dump(l_mob_features_case_lag, output)
with open(os.path.join(data_dir, 'l_mob_features_death_lag.pickle'), 'wb') as output:
    pickle.dump(l_mob_features_death_lag, output)    


writer=pd.ExcelWriter(os.path.join(data_dir,'mob_features_ranking_ridge.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'mob_features_ranking_ridge.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'mob_features_ranking_ridge.xlsx'),engine='openpyxl')
writer.book=book
l_mob_features_case_lag.to_excel(writer,'cases')
l_mob_features_death_lag.to_excel(writer,'deaths')
writer.save()
   

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

data_dir = '//rschfiler2.baml.com/UK-NYCOMMODITIES/Jamal/CovidAnalysis/Mobility'

fname=os.path.join(data_dir, 'county_summary.pickle')
county_summary = pickle.load( open( fname, "rb" ) )

fname=os.path.join(data_dir, 'cluster_labels.pickle')
cluster_labels = pickle.load( open( fname, "rb" ) )

clusterdata = cluster_labels[['cluster','case_rate_ld','death_rate_ld']]

writer=pd.ExcelWriter(os.path.join(data_dir,'county_cluster_data.xlsx'))
writer.save()
for c in clusterdata['cluster'].unique():
    from openpyxl import load_workbook
    book= load_workbook(os.path.join(data_dir, 'county_cluster_data.xlsx'))
    writer=pd.ExcelWriter(os.path.join(data_dir, 'county_cluster_data.xlsx'),engine='openpyxl')
    writer.book=book
    clusterdata[clusterdata['cluster']==c].reset_index(drop=True).to_excel(writer,str(c))
    writer.save()
    
fname=os.path.join(data_dir, 'cases_county.pickle')
cases_county = pickle.load( open( fname, "rb" ) )

mobility_vars = ['retail_recreation','grocery_pharmacy','parks','transit_stations','workplaces','residential','driving']

temp = (cases_county[['location']+mobility_vars].groupby('location').mean() - 100).reset_index()
cluster_mobility_avg = pd.merge(temp,cluster_labels[['location','cluster']],on='location',how='left').drop(['location'],axis=1)
cluster_mobility_avg = cluster_mobility_avg.groupby('cluster').mean()
cluster_mobility_avg = cluster_mobility_avg

temp = (cases_county[['location']+mobility_vars].groupby('location').quantile(0.9) - 100).reset_index()
cluster_mobility_90q = pd.merge(temp,cluster_labels[['location','cluster']],on='location',how='left').drop(['location'],axis=1)
cluster_mobility_90q = cluster_mobility_90q.groupby('cluster').mean()
cluster_mobility_90q = cluster_mobility_90q

temp = (cases_county[['location']+mobility_vars].groupby('location').quantile(0.5) - 100).reset_index()
cluster_mobility_50q = pd.merge(temp,cluster_labels[['location','cluster']],on='location',how='left').drop(['location'],axis=1)
cluster_mobility_50q = cluster_mobility_50q.groupby('cluster').mean()
cluster_mobility_50q = cluster_mobility_50q

writer=pd.ExcelWriter(os.path.join(data_dir,'cluster_mobility_avg.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'cluster_mobility_avg.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'cluster_mobility_avg.xlsx'),engine='openpyxl')
writer.book=book
cluster_mobility_avg.to_excel(writer,'cluster_mobility_avg')
cluster_mobility_90q.to_excel(writer,'cluster_mobility_90q')
cluster_mobility_50q.to_excel(writer,'cluster_mobility_50q')
writer.save()
       
############################################################################################################

## map data
from bs4 import BeautifulSoup

fname=os.path.join(data_dir, 'driving_county_avg_1.pickle')
driving_county_avg_1 = pickle.load( open( fname, "rb" ) )
driving_county_dict_1 = {}
for c in driving_county_avg_1['FIPS'].unique():
    try:
        driving_county_dict_1[c] = float(driving_county_avg_1.loc[driving_county_avg_1['FIPS'] == c, 'driving'])
    except:
        continue

fname=os.path.join(data_dir, 'driving_county_avg_2.pickle')
driving_county_avg_2 = pickle.load( open( fname, "rb" ) )
driving_county_dict_2 = {}
for c in driving_county_avg_2['FIPS'].unique():
    try:
        driving_county_dict_2[c] = float(driving_county_avg_2.loc[driving_county_avg_2['FIPS'] == c, 'driving'])
    except:
        continue


############################################################################################################

# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_cases =  ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = driving_county_dict_1[p['id']]
        except:
            continue
             
        if rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.95):
            color_class = 8
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.90):
            color_class = 7
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.75):
            color_class = 6
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.6):
            color_class = 5
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.5):
            color_class = 4
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.25):
            color_class = 3
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.1):
            color_class = 2
        elif rate > np.quantile(driving_county_avg_1['driving'].dropna(),0.05):
            color_class = 1
        else:
            color_class = 0
        
        color = colors_cases[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_driving_1.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
    
# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_cases =  ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = driving_county_dict_2[p['id']]
        except:
            continue
             
        if rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.95):
            color_class = 8
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.90):
            color_class = 7
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.75):
            color_class = 6
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.6):
            color_class = 5
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.5):
            color_class = 4
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.25):
            color_class = 3
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.1):
            color_class = 2
        elif rate > np.quantile(driving_county_avg_2['driving'].dropna(),0.05):
            color_class = 1
        else:
            color_class = 0
        
        color = colors_cases[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_driving_2.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
    
    
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################
        
import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_driving_1.svg')
SVG(fname)

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_driving_2.svg')
SVG(fname)

############################################################################################################

############################################################################################################

## map data

from bs4 import BeautifulSoup

map_dict_cases = {}
map_dict_deaths = {}
for c in county_summary['FIPS'].unique():
    try:
        map_dict_cases[c] = float(county_summary.loc[county_summary['FIPS'] == c, 'case_rate'])
    except:
        continue
    try:
        map_dict_deaths[c] = float(county_summary.loc[county_summary['FIPS'] == c, 'death_rate'])
    except:
        continue

############################################################################################################

# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_cases =  ['#fcfbfd', '#efedf5', '#dadaeb', '#dadaeb', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = map_dict_cases[p['id']]
        except:
            continue
             
        if rate > np.quantile(county_summary['case_rate'].dropna(),0.95):
            color_class = 8
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.90):
            color_class = 7
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.75):
            color_class = 6
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.6):
            color_class = 5
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.4):
            color_class = 4
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.25):
            color_class = 3
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.1):
            color_class = 2
        elif rate > np.quantile(county_summary['case_rate'].dropna(),0.01):
            color_class = 1
        else:
            color_class = 0
        
        color = colors_cases[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_cases.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
    
    
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################
        
import svgutils.compose as sc
from IPython.display import SVG # /!\ note the 'SVG' function also in svgutils.compose
import numpy as np

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_cases.svg')
SVG(fname)

############################################################################################################

# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_deaths = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = map_dict_deaths[p['id']]
        except:
            continue
             
        if rate > np.quantile(county_summary['death_rate'].dropna(),0.99):
            color_class = 8
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.95):
            color_class = 7
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.9):
            color_class = 6
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.8):
            color_class = 5
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.7):
            color_class = 4
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.6):
            color_class = 3
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.5):
            color_class = 2
        elif rate > np.quantile(county_summary['death_rate'].dropna(),0.4):
            color_class = 1
        else:
            color_class = 0
        
        color = colors_deaths[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_deaths.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
    
    
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_deaths.svg')
SVG(fname)


###############################################################

fname=os.path.join(data_dir, 'cluster_labels.pickle')
cluster_labels = pickle.load( open( fname, "rb" ) )

cluster_labels = pd.merge(cluster_labels, county_summary[['location','FIPS']],on='location',how='left')

cluster_labels_dict = {}
for c in cluster_labels['FIPS'].unique():
    try:
        cluster_labels_dict[c] = int(cluster_labels.loc[cluster_labels['FIPS'] == c, 'cluster'])
    except:
        continue


# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_clusters = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = cluster_labels_dict[p['id']]
        except:
            continue
             
        color_class = rate
        
        color = colors_clusters[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_clusters.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
        
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_clusters.svg')
SVG(fname)

###############################################################

fname=os.path.join(data_dir, 'mob_features_case_lag.pickle')
mob_features_case = pickle.load( open( fname, "rb" ) )
mob_features_case = mob_features_case.fillna(0)

top_features_case = mob_features_case.eq(mob_features_case.max(1), axis=0).dot(mob_features_case.columns).reset_index()
top_features_case.columns = ['location','type'] 
top_features_case['topmob'] = int(0)

top_features_case = pd.merge(top_features_case, county_summary[['location','FIPS']],on='location',how='left')

top_features_case.loc[top_features_case['type']=='retail_recreation','topmob'] = 1
top_features_case.loc[top_features_case['type']=='grocery_pharmacy','topmob'] = 2
top_features_case.loc[top_features_case['type']=='parks','topmob'] = 3
top_features_case.loc[top_features_case['type']=='transit_stations','topmob'] = 4
top_features_case.loc[top_features_case['type']=='workplaces','topmob'] = 5
top_features_case.loc[top_features_case['type']=='residential','topmob'] = 6

top_features_case_dict  = {}
for c in top_features_case['FIPS'].unique():
    try:
        top_features_case_dict[c] = top_features_case.loc[top_features_case['FIPS'] == c, 'topmob'].values[0]
    except:
        continue

# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_topmob = ['#d0d0d0', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = int(top_features_case_dict[p['id']])
        except:
            continue
                 
        color_class = rate
        
        color = colors_topmob[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_mob_cases.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
        
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_mob_cases.svg')
SVG(fname)

###############################################################

fname=os.path.join(data_dir, 'mob_features_death_lag.pickle')
mob_features_death = pickle.load( open( fname, "rb" ) )
mob_features_death = mob_features_death.fillna(0)

top_features_death = mob_features_death.eq(mob_features_death.max(1), axis=0).dot(mob_features_death.columns).reset_index()
top_features_death.columns = ['location','type'] 
top_features_death['topmob'] = int(0)

top_features_death = pd.merge(top_features_death, county_summary[['location','FIPS']],on='location',how='left')

top_features_death.loc[top_features_death['type']=='retail_recreation','topmob'] = 1
top_features_death.loc[top_features_death['type']=='grocery_pharmacy','topmob'] = 2
top_features_death.loc[top_features_death['type']=='parks','topmob'] = 3
top_features_death.loc[top_features_death['type']=='transit_stations','topmob'] = 4
top_features_death.loc[top_features_death['type']=='workplaces','topmob'] = 5
top_features_death.loc[top_features_death['type']=='residential','topmob'] = 6

top_features_death_dict  = {}
for c in top_features_death['FIPS'].unique():
    try:
        top_features_death_dict[c] = top_features_death.loc[top_features_death['FIPS'] == c, 'topmob'].values[0]
    except:
        continue

# Load the SVG map
fname=os.path.join(data_dir, 'counties.svg')
svg = open(fname, 'r').read()

# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])

# Find counties
paths = soup.findAll('path')

# map colors
colors_topmob = ['#d0d0d0', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
                
# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'

    
# Color the counties based on case rate 
for p in paths:
     
    if p['id'] not in ["State_Lines", "separator"]:
        # pass
        try:
            rate = int(top_features_death_dict[p['id']])
        except:
            continue
                 
        color_class = rate
        
        color = colors_topmob[color_class]
        p['style'] = path_style + color
    

# Output map
fname = os.path.join(data_dir, 'countymap_mob_deaths.svg')      
with open(fname, 'w') as f:
    print(soup.prettify(),file=f)
        
### CHANGE STROKE IN LAST TWO LINES TO #FFFFFF    

###############################################################

# print image then save as png -- word to excel and RSCH copy into word with title
fname=os.path.join(data_dir, 'countymap_mob_deaths.svg')
SVG(fname)

###############################################################

fname=os.path.join(data_dir, 'mob_features_case_lag.pickle')
mob_features_case = pickle.load( open( fname, "rb" ) )
mob_features_case = mob_features_case.fillna(0)
mob_features_case = mob_features_case.reset_index() 
mob_features_case = mob_features_case.rename(columns = {'index':'location'})

fname=os.path.join(data_dir, 'cluster_labels.pickle')
cluster_labels = pickle.load( open( fname, "rb" ) )

mob_features_ranking_case = pd.merge(mob_features_case,cluster_labels[['location','cluster']],on='location',how='left')

writer=pd.ExcelWriter(os.path.join(data_dir,'mob_features_ranking_case.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'mob_features_ranking_case.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'mob_features_ranking_case.xlsx'),engine='openpyxl')
writer.book=book
mob_features_ranking_case.to_excel(writer,'mob_features_ranking_case')
writer.save()



fname=os.path.join(data_dir, 'mob_features_death_lag.pickle')
mob_features_death = pickle.load( open( fname, "rb" ) )
mob_features_death = mob_features_death.fillna(0)
mob_features_death = mob_features_death.reset_index() 
mob_features_death = mob_features_death.rename(columns = {'index':'location'})

fname=os.path.join(data_dir, 'cluster_labels.pickle')
cluster_labels = pickle.load( open( fname, "rb" ) )

mob_features_ranking_death = pd.merge(mob_features_death,cluster_labels[['location','cluster']],on='location',how='left')
mob_features_ranking_death = mob_features_ranking_death.dropna().reset_index(drop=True)
temp = mob_features_ranking_death.iloc[:,1:-1]
temp['sum'] = temp.iloc[:,:6].sum(1) 
temp['mean']= temp.iloc[:,:6].mean(1) 
temp['mean'].mean()

temp[temp['sum'] != 0]

writer=pd.ExcelWriter(os.path.join(data_dir,'mob_features_ranking_death.xlsx'))
writer.save()
from openpyxl import load_workbook
book= load_workbook(os.path.join(data_dir, 'mob_features_ranking_death.xlsx'))
writer=pd.ExcelWriter(os.path.join(data_dir, 'mob_features_ranking_death.xlsx'),engine='openpyxl')
writer.book=book
mob_features_ranking_death.to_excel(writer,'mob_features_ranking_death')
writer.save()

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

    