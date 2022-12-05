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


### Parameters
n = 100 # Number of minhashes/rows in signature matrix
# t = # Threshold
LSH_number_of_bands = 20# Number of bands in LSH
LSH_number_of_rows = 5# Number of rows in each band in LSH

 
### Function
# Found on 4-12-2022 at: https://datagy.io/python-prime-numbers/
def is_prime(number):
    if number > 1:
        for num in range(2, int(number**0.5) + 1):
            if number % num == 0:
                return False
        return True
    return False

def find_next_prime(number):
    while True:
        if is_prime(number):
            return number
        elif (number%2 == 0): # Check if p is an even number, if yes, make uneven
            number += 1
        else:
            number += 2




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

# Remove symbols make tv titles more unclear
tv_titles = [i.replace("(", "") for i in tv_titles]
tv_titles = [i.replace(")", "") for i in tv_titles]
tv_titles = [i.replace("[", "") for i in tv_titles]
tv_titles = [i.replace("]", "") for i in tv_titles]
tv_titles = [i.replace("{", "") for i in tv_titles]
tv_titles = [i.replace("}", "") for i in tv_titles]
tv_titles = [i.replace(":", "") for i in tv_titles]
tv_titles = [i.replace("/", "") for i in tv_titles]

# Replace variants of inch with 'inch'
tv_titles = [i.replace("\"", "inch") for i in tv_titles]
tv_titles = [i.replace("\' \'", "inch") for i in tv_titles]
tv_titles = [i.replace("''", "inch") for i in tv_titles]
tv_titles = [i.replace("'", "inch") for i in tv_titles]
tv_titles = [i.replace(" in", "inch") for i in tv_titles]
tv_titles = [i.replace("inches", "inch") for i in tv_titles]
tv_titles = [i.replace("-inch", "inch") for i in tv_titles]
tv_titles = [i.replace(" inch", "inch") for i in tv_titles]
tv_titles = [i.replace("”", "inch") for i in tv_titles]

#  Replace variants of hertz with 'z'
tv_titles = [i.replace("-hz", "hz") for i in tv_titles]
tv_titles = [i.replace(" hz", "hz") for i in tv_titles]
tv_titles = [i.replace("hertz", "hz") for i in tv_titles]
tv_titles = [i.replace(" hertz", "hz") for i in tv_titles]



### Creating the model word set
mw_set = set()
for i in tv_titles:
    modelwords = re.finditer('[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*', i)
    for k in modelwords:
        mw_set.add(k.group())
mw_list = list(mw_set) # Change set to list to allow for indices


### Create the characteristic matrix
B = np.zeros((len(mw_list),len(tv_titles))) # Empty characteristic matrix
for tv in tv_titles:
    for mw in mw_list:
        if mw in tv:
            B[mw_list.index(mw),tv_titles.index(tv)] = 1 # No else statement needed as we start with a zero matrix



### Creating the signature matrix
S = np.inf * np.ones((n,len(tv_titles))) # Initialize signature matrix with each element +inf

# Parameters for the hash functions
a = np.random.randint(len(mw_set), 10*len(mw_set), size = n) # Choose len(mw_set) as upper bound to find good numbers
b = np.random.randint(len(mw_set), 10*len(mw_set), size = n)
p = find_next_prime(len(mw_set)) # The mod value should be greater than the number of elements

# Filling the signature matrix
# TODO: kijken of dit wel écht goed gaat, nu erg veel columns die alleen maar inf zijn (hogere n waarde werkt beter)
h = np.zeros((n,1)) # Initiate the hash values array
for r in range(len(B)): # For each row r
    for k in range(n):
        h[k] = (a[k] + b[k] * r) % p # TODO: Als je hier de indices veranderrd naar iets anders dan 'n' dan lijkt het stuk te gaan
    for c in range(len(B[0])): # For each column c
        if B[r,c] == 1:
            for i in range(n):
                if h[i] < S[i,c]:
                    S[i,c] = h[i]


### LSH
# Parameters
nob = LSH_number_of_bands # number of bands
rpb = LSH_number_of_rows # rows per band
if nob*rpb != len(S):
    print("Number of bands times number of rows is not equal to length of Signature matrix! \nChoose different values so that b*r=n! ")


# Divide signature matrix in bands
LSH_matrix = np.zeros((S.shape[0]//nob,S.shape[1],nob))
for b in range(nob):
    for i in range(rpb):
        LSH_matrix[i,:,b] = S[b*rpb+i,:]


# Hash into buckets
candidate_pairs = set()
buckets_number = find_next_prime(2*len(tv_titles)) # Ensure that nr buckets > nr titles
bucket_values = np.zeros((buckets_number,len(tv_titles)))
alpha = np.random.randint(0, 10*len(mw_set)) # Choose len(mw_set) as upper bound to find good numbers
beta = np.random.randint(1, 10*len(mw_set))
for b in range(nob):
    for i in range(len(S[0])):
        # Hash function on LSH_matrix[:,i,b]
        x = LSH_matrix[:,i,b].sum(axis=0)
        if x != np.inf:
            y = (alpha + beta*int(x)) % p
            bucket_values[y,i] += 1
        
# Candidate pair selection by LSH






















