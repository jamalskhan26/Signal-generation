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



