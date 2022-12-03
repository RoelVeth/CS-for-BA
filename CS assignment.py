# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:11:48 2022

@author: Roel
"""
# Imports
import numpy as np
import json
import re
from sklean.metrics import jaccard_score


# Loading the data file
file = open('TVs-all-merged.json')
data = json.load(file)


# Load the data to a list TODO: Check if this can be done in a dictionary, this could be faster?
tv = [] # List to fill
for i in data.keys():
    for j in range(len(data[i])):
        tv.append(data[i][j]['title'])


# Data cleaning
# Code carlos
wordsWithStrignAndChar = set()
for i in tv:
    tmp = re.findall('[a-zA-Z0-9]*(([0-9]+[^0-9^,^ ]+)|([^0-9^,^ ]+[0-9]+))[a-zA-Z0-9]*',i)



# Initialize LSH



# Candidate pair selection by LSH




























