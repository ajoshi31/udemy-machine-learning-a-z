# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
count_row = dataset.shape[0] 
count_column =  dataset.shape[1] 


transactions = []
for i in range(0, count_row):
    transactions.append([str(dataset.values[i,j]) for j in range(0, count_column)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results

results = list(rules)


'''
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
'''
    
results_list = []
for i in range(0, len(results)):
    results_list.append([str(results[i][0]),
                        str(results[i][1]),
                        str(results[i][2][0][2]),
                        str(results[i][2][0][3]), 
                        str(results[i][2][0][0]),
                        str(results[i][2][0][1])])
results_list = pd.DataFrame(data=results_list,columns=['RULE', 'SUPPORT','CONFIDENCE','LIFT' ,'BASE', 'RECOMMENDED'])

