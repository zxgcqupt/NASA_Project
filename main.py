#!/usr/bin/python

# encoding=utf8
"""
Created on Wed Sep 20 15:50:00 2017

@author: zhanx15
"""
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

## read the data from ASRS database
i = 1
data = []
with open('ASRS_DBOnline.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if i == 1:
            row_name = row
        else:
            data.append(row)
        i = i + 1
    f.close()


## construct dataframe
df = pd.DataFrame(data, columns = row_name)

## count the number of accidents by state name
state_names = np.unique(df['State Reference'])
accident_by_state_name = np.arange(len(state_names))

for i in range(len(state_names)):
    accident_by_state_name[i] = len(df.loc[df['State Reference'] == state_names[i]])
    if state_names[i] == '':
        state_names[i] = 'unknown'

# plot the figure
pos = np.arange(len(state_names))
#plt.figure()
#plt.bar(pos, accident_by_state_name, align='center', alpha=0.5)
#plt.xticks(pos, state_names)
#plt.ylabel('Accident Counts')
#plt.title('Number of accidents by state reference names')
#plt.show()

## count the number of accidents by flight phases
flight_phase = np.unique(df['Flight Phase1'])
accident_by_flight_phase = np.arange(len(flight_phase))

for i in range(len(flight_phase)):
    accident_by_flight_phase[i] = len(df.loc[df['Flight Phase1'] == flight_phase[i]])
    if flight_phase[i] == '':
        flight_phase[i] = 'Unknown'
    
# plot the figure
pos = np.arange(len(flight_phase))
#plt.figure()
#plt.bar(pos, accident_by_flight_phase, align='center', alpha=0.5)
#plt.xticks(pos, flight_phase)
#plt.ylabel('Accident Counts')
#plt.title('Number of accidents by flight phase')
#plt.show()


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import StanfordSegmenter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import numpy as np
from collections import defaultdict

# set of stop words
stop_words = stopwords.words('english') + list(string.punctuation)
index = df.columns.get_loc("Synopsis")
unique_words = set()
document_words = [[] for i in range(df.shape[0])]
for i in range(df.shape[0]):
    ## sentence tokenization
    sent_tokenize_list = sent_tokenize(df.iloc[i, index])
    
    if i == 0:
        print "########################### The tokenized report is: ###########################"
        
    for j in range(len(sent_tokenize_list)):
        if i == 0 and j == 0:
            print sent_tokenize_list[j]
            
        ## word tokenization
        word_tokenize_list = word_tokenize(sent_tokenize_list[j])
        if i == 0 and j == 0:
            print "########################### The tokenized words before removing stop words are: ###########################"
            print word_tokenize_list
            print "\n"
        
        filtered_sentence = [w for w in word_tokenize_list if not w in stop_words]
        if i == 0 and j == 0:
            print "########################### The tokenized words after removing stop words are: ###########################"
            print filtered_sentence
            print "\n"
            
        ## Stemming
        stemmer = SnowballStemmer('english')
        stemmed_words = []
        if i == 0 and j == 0:
            print "########################### The stemmed words are: ###########################"
            for m in range(len(filtered_sentence)):
                stemmed_words.append(stemmer.stem(filtered_sentence[m]))
                print stemmer.stem(filtered_sentence[m])
        else:
            for m in range(len(filtered_sentence)):
                stemmed_words.append(stemmer.stem(filtered_sentence[m]))
            
        document_words[i] = stemmed_words
        ## store the unique words from all the accidents
        unique_words = unique_words.union(stemmed_words)
        
unique_words = list(unique_words)
## create the document-words matrix representation
document_rep = np.zeros((df.shape[0], len(unique_words)))

for i in range(df.shape[0]):
    for j in range(len(document_words[i])):
        word = document_words[i][j]
        location = unique_words.index(word)
        times = document_words[i].count(word)
        document_rep[i][location] = times


## Compute the similarity matrix (Frequency-based)
similarity_matrix_freq = np.zeros((df.shape[0], df.shape[0]))
for i in range(df.shape[0]):
    for j in range(i, df.shape[0]):
        dot_product = np.dot(document_rep[i], document_rep[j])
        mode_a = np.sqrt(np.sum(np.square(document_rep[i])))
        mode_b = np.sqrt(np.sum(np.square(document_rep[j])))
        theta = dot_product/(mode_a * mode_b)
        similarity_matrix_freq[i][j] = theta
        similarity_matrix_freq[j][i] = theta
        
## output the document that has the highest similarity to the first accident report:
start_loc = 1
loc = np.array(similarity_matrix_freq[0, start_loc:]).argmax() + start_loc
print '#######   The first accident synopsis is:   ########## '
print df['Synopsis'][0]

print "\n"
print '###   The document that has the highest similarity with the first accident report is:   ###'
print df['Synopsis'][loc]
        

## Calculate the similarity --> Latent Semantics Analysis (LSA)
#dictionary = defaultdict(list)
#for i in range(df.shape[0]):
#    ## sentence tokenization
#    sent_tokenize_list = sent_tokenize(df.iloc[i, index])
#    for j in range(len(sent_tokenize_list)):
#        word_tokenize_list = word_tokenize(sent_tokenize_list[j])
#        filtered_sentence = [w for w in word_tokenize_list if not w in stop_words]
#        
#        stemmer = SnowballStemmer('english')
#        
#        for m in range(len(filtered_sentence)):
#            word = stemmer.stem(filtered_sentence[m])
#            if word in dictionary:
#                dictionary[word].append(i)
#            else:
#                dictionary[word] = [i]
#            
#
#keywords = [k for k in dictionary.keys() if len(dictionary[k]) > 1]
#keywords.sort()
#
#X = np.zeros([len(keywords), df.shape[0]])
#
#for i, k in enumerate(keywords):
#    for d in dictionary[k]:
#        X[i,d] += 1
#
#U, sigma, V = np.linalg.svd(X, full_matrices=True)
#
#targetDimension = 2000
#U2 = U[0:, 0:targetDimension]
#V2 = V[0:targetDimension, 0:]
#sigma2 = np.diag(sigma[0:targetDimension])
#print U2.shape, sigma2.shape, V2.shape


## Contruct Bayesian network from the accident reports
flight_phase = np.unique(df['Flight Phase1'])

number_flight_phase = dict()
index_flight_phase = defaultdict(list)
for i in range(df.shape[0]):
    if df['Flight Phase1'][i] == '':
        key = 'unknown'
    else:
        key = df['Flight Phase1'][i]
    
    if key in number_flight_phase.keys():
        number_flight_phase[key] += 1
        index_flight_phase[key].append(i)
    else:
        number_flight_phase[key] = 1
        index_flight_phase[key] = [i]



## group the category into the 10 flight phases
phase_belief_count = {'Taxi': 0, 'Parked': 0, 'Takeoff': 0, 'Initial Climb': 0,'Climb': 0, 'Cruise': 0, 'Descent': 0, 'Initial Approach': 0, 'Final Approach': 0,'Landing': 0, 'Other': 0, '': 0}
phase_plausi_count = {'Taxi': 0, 'Parked': 0, 'Takeoff': 0, 'Initial Climb': 0,'Climb': 0, 'Cruise': 0, 'Descent': 0, 'Initial Approach': 0, 'Final Approach': 0,'Landing': 0, 'Other': 0, '': 0}

no_phase = len(phase_belief_count)

## compute the belief and plausibility values for each category
for i in range(df.shape[0]):
    val = df['Flight Phase1'][i]
    if val in phase_belief_count.keys():
        phase_belief_count[val] += 1
        phase_plausi_count[val] += 1
    else:
        if val.startswith('Other'):
            phase_belief_count['Other'] += 1
            phase_plausi_count['Other'] += 1
        else:
            tmp_phases = df['Flight Phase1'][i].split(';')
            for k in range(len(tmp_phases)):
                tmp_val = tmp_phases[k]
                if tmp_val in phase_belief_count.keys():
                    phase_plausi_count[tmp_val] += 1


## Identify the root causes and count the number of times that the root appears in each category
set_root_cause = defaultdict(list)
for phase, value in index_flight_phase.items():
    if phase in phase_belief_count.keys() or phase.startswith('Other'):
        if phase.startswith('Other'):
            phase = 'Other'
        for i in range(len(value)):
            root_cause = df['Primary Problem'][value[i]]
            if phase in set_root_cause.keys():
                set_root_cause[phase].append(root_cause)
            else:
                set_root_cause[phase] = [root_cause]


## count the number of accidents that result from each root cause for each category
number_root_cause = dict()
for key, value in set_root_cause.items():
    length = len(np.unique(value))
    for i in range(len(value)):
        if value[i] == '':
            key_1 = 'unknown'
        else:
            key_1 = value[i]
        key_val = (key, key_1)
        if key_val in number_root_cause.keys():
            number_root_cause[key_val] += 1
        else:
            number_root_cause[key_val] = 1
        


## latent dirichilet allocation
import gensim;
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)



            
        



#



