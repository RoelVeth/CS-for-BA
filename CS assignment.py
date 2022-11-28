# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:11:48 2022

@author: Roel
"""
# Imports
import numpy as np
import json

# Loading the data
file = open('TVs-all-merged.json')
data = json.load(file)

for i in range(5):#range(len(data)):
    print(data[i].key)
    # for j in range(len(data[i])):
    #     print(data[i].key)

# my_list = []
# for i in data.values():
#     my_list.append(i)


























