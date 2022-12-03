# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:11:48 2022

@author: Roel
"""
# Imports
import numpy as np
import json
import re
from sklearn.metrics import jaccard_score


### Loading the data file
file = open('TVs-all-merged.json')
data = json.load(file)

# Load the data to a list TODO: Check if this can be done in a dictionary, this could be faster?
tv_uncleaned = []
for i in data.keys():
    for j in range(len(data[i])):
        tv_uncleaned.append(data[i][j]['title'])
        


### Data cleaning
tv_titles = tv_uncleaned # Keep uncleaned stored for comparison
# Replace all capital letters with non-capital
tv_titles = [i.lower() for i in tv_titles]

# Replace variants of inch with 'inch'
tv_titles = [i.replace("\"", "inch") for i in tv_titles]
tv_titles = [i.replace("\' \'", "inch") for i in tv_titles]
tv_titles = [i.replace("inches", "inch") for i in tv_titles]
tv_titles = [i.replace("-inch", "inch") for i in tv_titles]
tv_titles = [i.replace(" inch", "inch") for i in tv_titles]

#  Replace variants of hertz with 'z'
tv_titles = [i.replace("-hz", "hz") for i in tv_titles]
tv_titles = [i.replace(" hz", "hz") for i in tv_titles]
tv_titles = [i.replace("hertz", "hz") for i in tv_titles]
tv_titles = [i.replace(" hertz", "hz") for i in tv_titles]



### Obtaining binary vectors
# Creating the model word set
mw_set = set()
for i in tv_titles:
    modelword = re.search('[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*', i)
    modelword = modelword.group()  # TODO: Rick had dit erachter, moet dat blijven?.replace(" ", "")
    mw_set.add(modelword)


# Create signature matrix 
# TODO: Is dit de signature matrix of is het iets anders? 
B = np.zeros((len(mw_set),len(tv_titles)))
for tv in tv_titles:
    for mw in mw_set:
        if mw in tv:
            B[mw,tv] = 1
        else:
            B[mw,tv] = 0




# Uitleg van die grote RegEx string
# [a-zA-Z0-9]* #zero or more alphanumericals
# (
#  ([0-9]+[^0-9^,^ ]+ # Ten minste 1 numerica gevolgd met ten minste 1 [niet nummer, niet comma, niet spatie] 
#   )| # Óf
#   ([^0-9^,^ ]+[0-9]+) # ten minste 1 [niet nummer, niet comma, niet spatie], gevolgd door ten minste 1 numerical
# )
# [a-zA-Z0-9]* #zero or more alphanumericals





# Building signature matrix

# Initialize LSH



# Candidate pair selection by LSH




























