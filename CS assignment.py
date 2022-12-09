# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:11:48 2022

@author: Roel Veth (622593)
Computer Science for Business Analytics
Individual assignment
"""
#%% Imports
import numpy as np
import json
import re
from sklearn.metrics import jaccard_score
import math



#%% Parameters
Rows_of_signature_matrix= 650 # Number of minhashes/rows in signature matrix
#Threshold_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80] # The set of Threshold values used
Threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]#, 0.6] # The sigle threshold value which was best from bootstrapping
# Modify the LSH number of bands/rows manually beteen simulations.
LSH_number_of_bands = 10 # Number of bands in LSH
LSH_number_of_rows = 65 # Number of rows in each band in LSH
number_of_simulations = len(Threshold_values)
LSH_threshold = (1/LSH_number_of_bands)**(1/LSH_number_of_rows)
print('(Bands, Rows) = (',LSH_number_of_bands,",", LSH_number_of_rows,")")
print('LSH Threshold ~=', LSH_threshold)
print('---------------------------------------------------------------')

Number_of_bootstraps = 5
Bootstrapping = True # Boolean to turn on/off taking bootstraps of the data

# Initiating some performance measures to place results in
# For the training data
TP = np.zeros((Number_of_bootstraps,number_of_simulations)) # True positives
FP = np.zeros((Number_of_bootstraps,number_of_simulations)) # False positives
FN = np.zeros((Number_of_bootstraps,number_of_simulations)) # False negatives
Precision = np.zeros((Number_of_bootstraps,number_of_simulations))
Recall = np.zeros((Number_of_bootstraps,number_of_simulations))
F1 = np.zeros((Number_of_bootstraps,number_of_simulations))
number_duplicates_found = np.zeros((Number_of_bootstraps,number_of_simulations))
number_comparisons_made = np.zeros((Number_of_bootstraps,number_of_simulations))
PQ = np.zeros((Number_of_bootstraps,number_of_simulations))
PC = np.zeros((Number_of_bootstraps,number_of_simulations))
F1star = np.zeros((Number_of_bootstraps,number_of_simulations))
FOC = np.zeros((Number_of_bootstraps,number_of_simulations))
F1_best = np.zeros(Number_of_bootstraps)

# For the test data
TP_test = np.zeros((Number_of_bootstraps)) # True positives
FP_test = np.zeros((Number_of_bootstraps)) # False positives
FN_test = np.zeros((Number_of_bootstraps)) # False negatives
Precision_test = np.zeros((Number_of_bootstraps))
Recall_test = np.zeros((Number_of_bootstraps))
F1_test = np.zeros((Number_of_bootstraps))
number_duplicates_found_test = np.zeros((Number_of_bootstraps))
number_comparisons_made_test = np.zeros((Number_of_bootstraps))
PQ_test = np.zeros((Number_of_bootstraps))
PC_test = np.zeros((Number_of_bootstraps))
F1star_test = np.zeros((Number_of_bootstraps))
FOC_test = np.zeros((Number_of_bootstraps))





#%% Functions
# A function to test whether a number is a prime number
# Found on 4-12-2022 at: https://datagy.io/python-prime-numbers/
def is_prime(number):
    if number > 1:
        for num in range(2, int(number**0.5) + 1):
            if number % num == 0:
                return False
        return True
    return False

# A function to find the next prime number
def find_next_prime(number):
    while True:
        if is_prime(number):
            return number
        elif (number%2 == 0): # Check if p is an even number, if yes, make uneven
            number += 1
        else:
            number += 2

# Calculates the harmonic mean between two floats.
def harmonic_mean(x,y):
    if x==0 or y==0:
        return 0
    else:
        return 2*x*y/(x+y)



#%% Loading the data file
file = open('TVs-all-merged.json')
data = json.load(file)

# Load the data to a list TODO: Check if this can be done in a dictionary, this could be faster?
tv_uncleaned = [[],[],[],[]]
for i in data.keys():
    for j in range(len(data[i])):
        tv_uncleaned[0].append(data[i][j]['modelID'])
        tv_uncleaned[1].append(data[i][j]['title'])
        tv_uncleaned[2].append(data[i][j]['shop'])
        try:
            tv_uncleaned[3].append(data[i][j]['featuresMap']['Brand'])
        except:
            tv_uncleaned[3].append('-')

# Get the total number of duplicates (not uesd)
total_duplicates_number = 0
for i in data.keys():
    if len(data[i]) > 1:
        total_duplicates_number += math.comb(len(data[i]),2)
        
        
        
#%% Data cleaning
tv_all = tv_uncleaned # Keep uncleaned stored for comparison

# Indices of tv_titles are:
modelID = 0
title = 1
shop = 2
brand = 3

# Make sets before cleaning to check what cleaning should be done
modelID_set = set()
shop_set = set()
brand_set = set()
for i in range(len(tv_all[1])):
    modelID_set.add(tv_all[modelID][i])
    shop_set.add(tv_all[shop][i])
    brand_set.add(tv_all[brand][i])


# Replace all capital letters with non-capital
for k in range(1, len(tv_all)): # Skip this for modelID, as lowercasing those removes some ID's
    tv_all[k] = [i.lower() for i in tv_all[k]]

# Remove symbols make tv titles more unclear
tv_all[title] = [i.replace("(", "") for i in tv_all[title]]
tv_all[title] = [i.replace(")", "") for i in tv_all[title]]
tv_all[title] = [i.replace("[", "") for i in tv_all[title]]
tv_all[title] = [i.replace("]", "") for i in tv_all[title]]
tv_all[title] = [i.replace("{", "") for i in tv_all[title]]
tv_all[title] = [i.replace("}", "") for i in tv_all[title]]
tv_all[title] = [i.replace(":", "") for i in tv_all[title]]
tv_all[title] = [i.replace("/", "") for i in tv_all[title]]

# Replace variants of inch with 'inch'
tv_all[title] = [i.replace("\"", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("\' \'", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("''", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("'", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace(" in", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("inches", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("-inch", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace(" inch", "inch") for i in tv_all[title]]
tv_all[title] = [i.replace("”", "inch") for i in tv_all[title]]

#  Replace variants of hertz with 'z'
tv_all[title] = [i.replace("-hz", "hz") for i in tv_all[title]]
tv_all[title] = [i.replace(" hz", "hz") for i in tv_all[title]]
tv_all[title] = [i.replace("hertz", "hz") for i in tv_all[title]]
tv_all[title] = [i.replace(" hertz", "hz") for i in tv_all[title]]


# Remove some parts of the brands
# 'tv' and spaces
tv_all[brand] = [i.replace("tv", "") for i in tv_all[brand]]
tv_all[brand] = [i.replace(" ", "") for i in tv_all[brand]]




#%% Creating the model word set
mw_set = set()
for i in tv_all[title]:
    modelwords = re.finditer('[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*', i)
    for k in modelwords:
        mw_set.add(k.group())
mw_list = list(mw_set) # Change set to list to allow for indices



#%% Create the characteristic matrix
B = np.zeros((len(mw_list),len(tv_all[title]))) # Empty characteristic matrix
for tv in range(len(tv_all[title])): # go through the indices instead of elements, as some elements have the same name.
    for mw in mw_list:
        if mw in tv_all[title][tv]:
            B[mw_list.index(mw), tv] = 1



#%% Creating the signature matrix
n = Rows_of_signature_matrix
S = np.inf * np.ones((n,len(tv_all[title]))) # Initialize signature matrix with each element +inf

# Parameters for the hash functions
a = np.random.randint(0, 100*len(mw_set), size = n) # a & b random, but not insanely large
b = np.random.randint(1, 100*len(mw_set), size = n)
p = find_next_prime(len(mw_set)) # The mod value should be greater than the number of elements, but not extremely large to ensure the LSH-matrix will not have extremely long elements.

# Filling the signature matrix
h = np.zeros((n,1)) # Initiate the hash values array
for r in range(len(B)): # For each row r
    for k in range(n):
        h[k] = (a[k] + b[k] * r) % p
        
    for c in range(len(B[0])): # For each column c
        if B[r,c] == 1:
            for i in range(n):
                if h[i] < S[i,c]:
                    S[i,c] = h[i]
                    
S_complete = S # make a backup for when bootstrapping
                    
# Test if all values are < inf to see if something went wrong
if math.isinf(np.max(S)):
    print("The highest value element in the signature matrix is:", np.max(S))
    for i in range(len(tv_all[title])):
        if math.isinf(S[0,i]):
            print("Column ",i,' is inf!')




#%% Bootstrapping
if not Bootstrapping: # If not bootstrapping, continue with the normal 
    S = S_complete # Signature matrix stays te same TODO: Currently shouldn't work!
else:
    Threshold_best = np.zeros(Number_of_bootstraps)
    for bootstrap in range(Number_of_bootstraps): # Repeat it this many times
        print('Bootstrap number',bootstrap+1,'of', Number_of_bootstraps)
        print('----------------------------------------------------------')
        # Select a training set by bootstrap
        draw_list = list(range(len(S_complete[0]))) # A list of all TV indices
        training_set = set() # A set which will include the indices of the TV's drawn into the data set.
        test_set = set(draw_list)
        
        # Draw random numbers to select 
        draws = np.random.randint(0,len(S_complete[0]), len(S_complete[0]))
        for i in draws:
            training_set.add(draw_list[i]) # Add the element to the training set. Since it is a set it wont take duplicates
            try:
                test_set.remove(draw_list[i]) # Try to remove the element, if it is already gone it does nothing
            except:
                    pass # If error do nothing, element has already been removed.
        
        # Change sets to lists
        training_list = list(training_set)
        test_list = list(test_set) # Is not actually used fo jaccard similarity
        
        # Create new Signature matrix to use in the simulations as training set.
        S = np.zeros((len(S_complete), len(training_set))) # Initiate the new Signature matrix
        for i in range(len(training_list)):
            S[:,i] = S_complete[:,training_list[i]] # Fill the new signature matrix
        
        #True number of duplicates in bootstrap training data
        true_duplicates_training = 0
        for i in range(len(training_list)):
            for j in range(i+1, len(training_list)):
                if tv_all[modelID][training_list[i]] == tv_all[modelID][training_list[j]]:
                    true_duplicates_training += 1

        #True number of duplicates in bootstrap test data
        true_duplicates_test = 0
        for i in range(len(test_list)):
            for j in range(i+1, len(test_list)):
                if tv_all[modelID][test_list[i]] == tv_all[modelID][test_list[j]]:
                    true_duplicates_test += 1

        #%% Simulations
        # Repeat the simulation with different values of Threshold for each bootstrap to find the best threshold value

        for s in range(number_of_simulations):
            # print('>> B', bootstrap+1,'  S',s+1, 'of', number_of_simulations,', Threshold =', Threshold_values[s])
            # print('-----------------------------')
            
            #%% LSH to find candidate pairs
            # Parameters
            nob = LSH_number_of_bands # number of bands
            rpb = LSH_number_of_rows # rows per band
            if nob*rpb != len(S):
                print("Number of bands times number of rows is not equal to length of Signature matrix! \nChoose different values so that b*r=n! ")
            
            # LSH_threshold = (1/nob)**(1/rpb)
            # print('LSH Threshold ~=', LSH_threshold)
            
            # Divide signature matrix in bands
            LSH_matrix = np.zeros((S.shape[0]//nob,S.shape[1],nob))
            for b in range(nob):
                for i in range(rpb):
                    LSH_matrix[i,:,b] = S[b*rpb+i,:]
            
            
            # Hash into buckets
            bucket_number = find_next_prime(50*len(tv_all[title])) # Create a large number of buckets, so accidental candidate pairs are unlikely
            buckets = [[]] 
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
                      bucket = int((alpha[b] + beta[b] * value_to_hash) % bucket_number) # Use a different has function for each band
                      buckets[bucket].append(i) # Add the index of the current i to the bucket
                      
                candidate_pairs_new = [x for x in buckets if len(x)>1]# Remove buckets with less than 2 indices
                candidate_pairs_set.update(set(tuple(x) for x in candidate_pairs_new)) # add new candidate pairs to the set
            
            
            
            
            #%% Comparing the candidate pairs
            candidate_pairs = list(candidate_pairs_set) # Change the set to a list to iterate over the list.
            # print(candidate_pairs[i][j]), prints tuple value j of candidate pair i
            
            # Define the threshold value. If Jaccard score is over the threshold, consider the pair as a duplicate.
            threshold = Threshold_values[s]
            # print("The Jaccard threshold value is: ",threshold)
            
            duplicates = np.zeros((len(tv_all[title]),len(tv_all[title])))
            for i in range(len(candidate_pairs)):
                for j in range(len(candidate_pairs[i][:])):
                    for k in range(j+1,len(candidate_pairs[i])): # Only compare the right upper triangle of the matrix
                        indexA = candidate_pairs[i][j]
                        indexB = candidate_pairs[i][k]
                        
                        # Only compare tv's if their brand is equal or unknown AND their shop is different
                        brand_conditions = tv_all[brand][indexA] == tv_all[brand][indexB] or tv_all[brand][indexA] == '-'  or tv_all[brand][indexB] == '-'
                        shop_condition = tv_all[shop][indexA] == tv_all[shop][indexB]
                        if (brand_conditions and not shop_condition):
                            j_score = jaccard_score(B[:,indexA],B[:,indexB])
                        
                            if j_score > threshold: # If score is larger than threshold, classify them as a pair
                                duplicates[indexA, indexB] = 1
            
            
                        
            #%% Comparing found duplicates with true duplicates
            
            true_match_set = set()
            false_match_set = set()
            for i in range(len(tv_all[title])):
                for j in range(i+1,len(tv_all[title])): # Skip the diagonal part of the matrix
                    if duplicates[i,j] == 1:
                        if tv_all[modelID][i] == tv_all[modelID][j]:
                            value = [i,j]
                            true_match_set.add(tuple(value))
                        else:
                            value = [i,j]
                            false_match_set.add(tuple(value))
            
            
            
           
        
            
            #%% Performance measurements on training sets
            # The following performance measurements are required:
                # True positives: TP is given by the pairs of products that are predicted to be duplicates and are real duplicates.
                # False positives: FP is given by the pairs of products that are predicted to be duplicates but are real non-duplicates. 
                # True negatives: TN is given by pairs of products that are predicted to be nonduplicates and are real non-duplicates.
                # False negatives: FN is given by the pairs of products that are predicted to be non-duplicates but are real duplicates
                
                # Precision: TP/(TP+FP)
                # Recall: TP/(TP+FN)
                # F1-score: the harmonic mean between precision and recall
                
                # Pair quality: number of duplicates found/number of comparisons made
                # Pair completeness: number of duplicates found/total number of duplicates
                # F1*-score: the harmonic mean between pair quality and pair completeness
                
                # Fraction of comparisons:  (number of comparisons made/total number of possible comparisons).
            
            
            # F1 score
            TP[bootstrap,s] = len(true_match_set) # True positives
            FP[bootstrap,s] = len(false_match_set) # False positives
            FN[bootstrap,s] = true_duplicates_training - TP[bootstrap,s] # False negatives 
            # TN = # True negatives (not used)
            
            if TP[bootstrap,s] ==0 and FP[bootstrap,s]==0:
                Precision[bootstrap,s] = 0
            else:
                Precision[bootstrap,s] = TP[bootstrap,s]/(TP[bootstrap,s]+FP[bootstrap,s])
            Recall[bootstrap,s] = TP[bootstrap,s]/(TP[bootstrap,s]+FN[bootstrap,s])
            F1[bootstrap,s] = harmonic_mean(Precision[bootstrap,s], Recall[bootstrap,s])
            
            
            # F1* score
            number_duplicates_found[bootstrap,s] = TP[bootstrap,s]
            number_comparisons_made[bootstrap,s] = 0
            for i in range(len(candidate_pairs)):
                if len(candidate_pairs[i]) > 1:
                    number_comparisons_made[bootstrap,s] += math.comb(len(candidate_pairs[i]),2)
                 
            PQ[bootstrap,s] = number_duplicates_found[bootstrap,s] / number_comparisons_made[bootstrap,s] # Pair quality
            PC[bootstrap,s] = number_duplicates_found[bootstrap,s] / total_duplicates_number # Pair completeness
            F1star[bootstrap,s] = harmonic_mean(PQ[bootstrap,s], PC[bootstrap,s])
            
            # Fraction of comparisons
            max_possible_comparisons = int(len(S[0]) * (len(S[0])-1) /2)
            FOC[bootstrap,s] = number_comparisons_made[bootstrap,s] / max_possible_comparisons
            
            # print('True positives:', TP[bootstrap,s])
            # print('-----------------------------')
            # print('Precision =', Precision[s])
            # print('Recall =',Recall[s])
            # print('F1 =',F1[bootstrap,s])
            # print('-----------------------------')
            # print("Pair quality =",PQ[s])
            # print("Pair completeness =",PC[s])
            # print("F1* =",F1star[bootstrap,s])
            # print('-----------------------------')
            # print("FOC =", FOC[bootstrap,s])
            # print('-----------------------------------------------------')
            
        
        F1_best[bootstrap] = max(F1[bootstrap,:])
        for s in range(number_of_simulations):
            if F1[bootstrap,s] == F1_best[bootstrap]:
                Threshold_best[bootstrap] = Threshold_values[s]
        # print('The best F1 value (=', F1_best[bootstrap],') in bootstrap',bootstrap+1, 'is found for the threshold value of',Threshold_best[bootstrap])
        # print('-----------------------------------------------------')
        
        #%% Use best found threshold on the test set
        
        # Run the simulation on the test_data to find a F1 score
        Threshold_test = Threshold_best[bootstrap]
        
        # Create new Signature matrix to use in the simulations as test set.
        S = np.zeros((len(S_complete), len(test_set))) # Initiate the new Signature matrix
        for i in range(len(test_list)):
            S[:,i] = S_complete[:,test_list[i]] # Fill the new signature matrix
            
        # Divide signature matrix in bands
        LSH_matrix = np.zeros((S.shape[0]//nob,S.shape[1],nob))
        for b in range(nob):
            for i in range(rpb):
                LSH_matrix[i,:,b] = S[b*rpb+i,:]
                
        # Hash into buckets
        bucket_number = find_next_prime(50*len(tv_all[title])) # Create a large number of buckets, so accidental candidate pairs are unlikely
        buckets = [[]] 
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
                  bucket = int((alpha[b] + beta[b] * value_to_hash) % bucket_number) # Use a different has function for each band
                  buckets[bucket].append(i) # Add the index of the current i to the bucket
                  
            candidate_pairs_new = [x for x in buckets if len(x)>1]# Remove buckets with less than 2 indices
            candidate_pairs_set.update(set(tuple(x) for x in candidate_pairs_new)) # add new candidate pairs to the set
        
        # Comparing the candidate pairs in the test set
        candidate_pairs = list(candidate_pairs_set) # Change the set to a list to iterate over the list.
        
        duplicates = np.zeros((len(tv_all[title]),len(tv_all[title])))
        for i in range(len(candidate_pairs)):
            for j in range(len(candidate_pairs[i][:])):
                for k in range(j+1,len(candidate_pairs[i])): # Only compare the right upper triangle of the matrix
                    indexA = candidate_pairs[i][j]
                    indexB = candidate_pairs[i][k]
                    
                    # Only compare tv's if their brand is equal or unknown AND their shop is different
                    brand_conditions = tv_all[brand][indexA] == tv_all[brand][indexB] or tv_all[brand][indexA] == '-'  or tv_all[brand][indexB] == '-'
                    shop_condition = tv_all[shop][indexA] == tv_all[shop][indexB]
                    if (brand_conditions and not shop_condition):
                        j_score = jaccard_score(B[:,indexA],B[:,indexB])
                    
                        if j_score > Threshold_test: # If score is larger than threshold, classify them as a pair
                            duplicates[indexA, indexB] = 1
        
        # Compare estimated duplicate classification with true duplicates
        true_match_set_test = set()
        false_match_set_test = set()
        for i in range(len(tv_all[title])):
            for j in range(i+1,len(tv_all[title])): # Skip the diagonal part of the matrix
                if duplicates[i,j] == 1:
                    if tv_all[modelID][i] == tv_all[modelID][j]:
                        value = [i,j]
                        true_match_set_test.add(tuple(value))
                    else:
                        value = [i,j]
                        false_match_set_test.add(tuple(value))
        
        # Test set performance measures
        # F1 score
        TP_test[bootstrap] = len(true_match_set_test) # True positives
        FP_test[bootstrap] = len(false_match_set_test) # False positives
        FN_test[bootstrap] = true_duplicates_test - TP_test[bootstrap] # False negatives 
        # TN = # True negatives (not used)
        
        if TP_test[bootstrap]==0 and FP_test[bootstrap]==0:
            Precision_test[bootstrap] = 0
        else:
            Precision_test[bootstrap] = TP_test[bootstrap]/(TP_test[bootstrap]+FP_test[bootstrap])
        Recall_test[bootstrap] = TP_test[bootstrap]/(TP_test[bootstrap]+FN_test[bootstrap])
        F1_test[bootstrap] = harmonic_mean(Precision_test[bootstrap], Recall_test[bootstrap])
        
        
        # F1* score
        number_duplicates_found_test[bootstrap] = TP_test[bootstrap]
        number_comparisons_made[bootstrap] = 0
        for i in range(len(candidate_pairs)):
            if len(candidate_pairs[i]) > 1:
                number_comparisons_made_test[bootstrap] += math.comb(len(candidate_pairs[i]),2)
             
        PQ_test[bootstrap] = number_duplicates_found_test[bootstrap] / number_comparisons_made_test[bootstrap] # Pair quality
        PC_test[bootstrap] = number_duplicates_found_test[bootstrap] / true_duplicates_test # Pair completeness
        F1star_test[bootstrap] = harmonic_mean(PQ_test[bootstrap], PC_test[bootstrap])
        
        # Fraction of comparisons
        max_possible_comparisons_test = int(len(S[0]) * (len(S[0])-1) /2)
        FOC_test[bootstrap] = number_comparisons_made_test[bootstrap] / max_possible_comparisons_test
        
        
    
    print('The average F1 score on training data over all', Number_of_bootstraps, 'bootstraps was:', np.mean(F1_best))
    print('The Threshold values used were:', Threshold_best)
    print('The average F1 score on test data is:', np.mean(F1_test))
    print('-----------------------------------------------------')


    

    
    
    
    
    












































