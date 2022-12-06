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
import math

# TODO: Check to also take features into account, maybe concatenate them after title
# TODO: Check to implement another similarity measure than MSM
# 



### Parameters
Rows_of_signature_matrix= 650 # Number of minhashes/rows in signature matrix
# t = # Threshold
LSH_number_of_bands = 13# Number of bands in LSH
LSH_number_of_rows = 50# Number of rows in each band in LSH

 
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
        
        # TODO: Try to append the features to 'title', so the features are also taken into account
        # TODO: In ieder geval brand en webshop, die zijn heel belangrijk
        
        
        


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
n = Rows_of_signature_matrix
# S = np.inf * np.ones((n,len(tv_titles))) # Initialize signature matrix with each element +inf
S = 99999 * np.ones((n,len(tv_titles))) # Initialize signature matrix with each element +inf

# Parameters for the hash functions
a = np.random.randint(0, 100*len(mw_set), size = n) # Choose len(mw_set) as upper bound to find good numbers
b = np.random.randint(1, 100*len(mw_set), size = n)
p = find_next_prime(len(mw_set)) # The mod value should be greater than the number of elements

# Filling the signature matrix
h = np.zeros((n,1)) # Initiate the hash values array
for r in range(len(B)): # For each row r
    for k in range(n):
        h[k] = (a[k] + b[k] * r) % p # TODO: Er zijn nog steeds columns inf, uitzoeken hoe dat komt en het fixen
        
    for c in range(len(B[0])): # For each column c
        if B[r,c] == 1:
            for i in range(n):
                if h[i] < S[i,c]:
                    S[i,c] = h[i]
                    
# Test if all values are < inf
print("The highest value element in the signature matrix is:", np.max(S))
if math.isinf(np.max(S)):
    for i in range(len(tv_titles)):
        if math.isinf(S[0,i]):
            print("Column ",i,' is inf!')



### LSH to find candidate pairs
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
bucket_number = find_next_prime(50*len(tv_titles)) # Create a large number of buckets, so accidental candidate pairs are unlikely
buckets = [[]] #np.zeros((bucket_number))
alpha = np.random.randint(0, 10*len(mw_set), size = nob) 
beta = np.random.randint(1, 10*len(mw_set), size = nob)

buckets = [] # Create a list of lists (a list of buckets)
for i in range(bucket_number):
    buckets.append([])                   

candidate_pairs_set = set()
for b in range(nob):
    candidate_pairs_new = set() # For each band, start with an empty set of candidate pairs
    for i in range(len(S[0])):
          # value_to_hash = ''.join(map(str, LSH_matrix[:,i,b]))
          value_to_hash = int(''.join(map(str, LSH_matrix[:,i,b].astype(int))))
          bucket = (alpha[b] + beta[b] * value_to_hash) % bucket_number # Use a different has function for each band
          buckets[bucket].append(i) # Add the index of the current i to the bucket
          
    candidate_pairs_new = [x for x in buckets if len(x)>1]# Remove buckets with less than 2 indices
    candidate_pairs_set.update(set(tuple(x) for x in candidate_pairs_new)) # add new candidate pairs to the set




### Comparing the candidate pairs
candidate_pairs = list(candidate_pairs_set) # Change the set to a list to iterate over the list.
# print(candidate_pairs[i][j]), prints tuple value j of candidate pair i

# Define the threshold value. If Jaccard score is over the threshold, consider the pair as a duplicate.
threshold = (1/nob)**(1/rpb)
print("The threshold value is: ",threshold)

# Hier MSM maken

            

















































