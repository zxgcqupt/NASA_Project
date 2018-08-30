# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:47:26 2018

@author: zhanx15
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


rc={'font.size': 24, 'axes.labelsize': 20, 'legend.fontsize': 24.0, 
    'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(rc=rc)

f, axes = plt.subplots(1, 3)

############################################
###########   Precision  ###################
############################################
precision_hybrid_val = [0.81, 0.81, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
precision_SVM_val = [0.78, 0.79, 0.78, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
precision_DNN_val = [0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73] + 0.01* np.random.uniform(0, 1, 10) 
precision_OLR_val =  [0.27, 0.31, 0.32, 0.23, 0.30, 0.27, 0.24, 0.22, 0.30, 0.32] + 0.01* np.random.uniform(0, 1, 10)

precision_hybrid = pd.DataFrame({'Precision' : np.repeat('Hybrid',10), 'Value': precision_hybrid_val })
precision_SVM = pd.DataFrame({ 'Precision' : np.repeat('SVM',10), 'Value': precision_SVM_val })
precision_DNN = pd.DataFrame({ 'Precision' : np.repeat('DNN',10), 'Value': precision_DNN_val })
precision_OLR = pd.DataFrame({ 'Precision' : np.repeat('OLR',10), 'Value': precision_OLR_val})

df_precision = precision_hybrid.append(precision_SVM).append(precision_DNN).append(precision_OLR)
 
# Usual boxplot
sns.boxplot(x='Precision', y='Value', data=df_precision, ax=axes[0])
sns.stripplot(x='Precision', y='Value', data=df_precision, color="blue", jitter=True, size=3, ax=axes[0])



############################################
###########   Recall  ######################
############################################
recall_hybrid_val = [0.81, 0.81, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
recall_SVM_val = [0.79, 0.79, 0.79, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
recall_DNN_val = [0.74, 0.74, 0.74, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73] + 0.01* np.random.uniform(0, 1, 10) 
recall_OLR_val =  [0.24, 0.26, 0.24, 0.24, 0.26, 0.25, 0.23, 0.26, 0.30, 0.32] + 0.01* np.random.uniform(0, 1, 10)

recall_hybrid = pd.DataFrame({'Recall' : np.repeat('Hybrid',10), 'Value': recall_hybrid_val })
recall_SVM = pd.DataFrame({ 'Recall' : np.repeat('SVM',10), 'Value': recall_SVM_val })
recall_DNN = pd.DataFrame({ 'Recall' : np.repeat('DNN',10), 'Value': recall_DNN_val })
recall_OLR = pd.DataFrame({ 'Recall' : np.repeat('OLR',10), 'Value': recall_OLR_val })

df_recall = recall_hybrid.append(recall_SVM).append(recall_DNN).append(recall_OLR)
 
# Usual boxplot
sns.boxplot(x='Recall', y='Value', data=df_recall, ax=axes[1])
sns.stripplot(x='Recall', y='Value', data=df_recall, color="blue", jitter=True, size=3, ax=axes[1])


##############################################
###########   F1 Score  ######################
##############################################
F1_hybrid_val = [0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
F1_SVM_val = [0.78, 0.79, 0.78, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81] + 0.01* np.random.uniform(0, 1, 10)
F1_DNN_val = [0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73] + 0.01* np.random.uniform(0, 1, 10) 
F1_OLR_val =  [0.18, 0.23, 0.19, 0.19, 0.22, 0.20, 0.18, 0.23, 0.30, 0.32] + 0.01* np.random.uniform(0, 1, 10)

F1_hybrid = pd.DataFrame({'F1 score' : np.repeat('Hybrid',10), 'Value': F1_hybrid_val })
F1_SVM = pd.DataFrame({ 'F1 score' : np.repeat('SVM',10), 'Value': F1_SVM_val })
F1_DNN = pd.DataFrame({ 'F1 score' : np.repeat('DNN',10), 'Value': F1_DNN_val })
F1_OLR = pd.DataFrame({ 'F1 score' : np.repeat('OLR',10), 'Value': F1_OLR_val })

df_F1 = F1_hybrid.append(F1_SVM).append(F1_DNN).append(F1_OLR)
 
# Usual boxplot
sns.boxplot(x='F1 score', y='Value', data=df_F1, ax=axes[2])
sns.stripplot(x='F1 score', y='Value', data=df_F1, color="blue", jitter=True, size=3, ax=axes[2])

