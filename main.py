#!/usr/bin/python

# encoding=utf8
"""
Created on Wed Sep 20 15:50:00 2017

@author: zhanx15
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from os import listdir
from sklearn.model_selection import train_test_split
print ('The version of TensorFlow is {}'.format(tf.__version__))

root_path = './data'

appended_data = []
for file_name in listdir(root_path):
    file_path = root_path + '/' + file_name.encode().decode('utf-8')
    data_from_one_csv = pd.read_csv(file_path, skiprows=1)
    appended_data.append(data_from_one_csv)
    
data = pd.concat(appended_data, axis=0)
data = data.drop(columns = ['ACN', 'Date', 'Local Time Of Day', 'Ceiling', 'Callback', 'Callback.1', 'Unnamed: 96'])
data = data.rename(index=str, columns={"Flight Phase": "Flight Phase1"})

## drop the rows with empty synopsis description
data = data[pd.notnull(data['Synopsis'])]

X = data.drop(columns = 'Result')
Y_raw = pd.DataFrame(data['Result'])

processed_Y = []
for index, row in Y_raw.iterrows():
    #print (index, row['Result'])
    outcome = row['Result']
    if type(outcome) == np.float:
        res = 'unknown'
        processed_Y.append(res)
    elif ';' in outcome:
        res = str(outcome).split(';')[0]
        processed_Y.append(res)
    else:
        res = outcome
        processed_Y.append(res)

Y = pd.DataFrame(processed_Y, columns = ['Result'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~ Single file process ~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# file_name = 'ASRS_DBOnline.csv'
# data = pd.read_csv(file_name)
# data = data.drop(columns = ['ACN', 'Date', 'Local Time Of Day', 'Ceiling', 'Callback', 'Callback.1'])
# X = data.drop(columns = 'Result')
# Y = data['Result']

# for i in range(Y.shape[0]):
#     if ';' in str(Y[i]):
#         Y.set_value(i, Y[i].split(';')[0])
#     elif Y[i] is np.nan:
#         Y.set_value(i, 'unknown')


import matplotlib.pyplot as plt
statics = Y.apply(pd.value_counts)
plt.plot(statics, 'r*')
plt.xlabel('Label')
plt.ylabel('Frequency')



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#Y = pd.DataFrame(le.fit_transform(Y), index = X.index)


## compress the number of labels to be predicted --> map result to risk level
rate_nine = ['General Declared Emergency', 'General Physical Injury / Incapacitation', 'Flight Crew Inflight Shutdown', 
             'Air Traffic Control Separated Traffic', 'Aircraft Aircraft Damaged']

rate_seven = ['General Evacuated', 'Flight Crew Landed as Precaution', 'Flight Crew Regained Aircraft Control', 
              'Air Traffic Control Issued Advisory / Alert', 'Flight Crew Landed in Emergency Condition',
              'Flight Crew Landed In Emergency Condition']
rate_five = ['General Work Refused', 'Flight Crew Became Reoriented', 'Flight Crew Diverted', 
             'Flight Crew Executed Go Around / Missed Approach', 
             'Flight Crew Overcame Equipment Problem', 'Flight Crew Rejected Takeoff', 'Flight Crew Took Evasive Action', 
             'Air Traffic Control Issued New Clearance']
rate_three = ['General Maintenance Action', 'General Flight Cancelled / Delayed', 'General Release Refused / Aircraft Not Accepted', 
              'Flight Crew Overrode Automation', 'Flight Crew FLC Overrode Automation',
              'Flight Crew Exited Penetrated Airspace', 
              'Flight Crew Requested ATC Assistance / Clarification', 'Flight Crew Landed As Precaution',
              'Flight Crew Returned To Clearance', 'Flight Crew Returned To Departure Airport',
              'Aircraft Automation Overrode Flight Crew']
rate_one = ['General Police / Security Involved', 'Flight Crew Returned To Gate', 'Aircraft Equipment Problem Dissipated', 
            'unknown', 'Air Traffic Control Provided Assistance',
            'General None Reported / Taken', 'Flight Crew FLC complied w / Automation / Advisory']

Y_ = []
for i in range(Y.shape[0]):
    if Y['Result'][i] in rate_nine:
        Y_.append(5)
    elif Y['Result'][i] in rate_seven:
        Y_.append(4)
    elif Y['Result'][i] in rate_five:
        Y_.append(3)
    elif Y['Result'][i] in rate_three:
        Y_.append(2)
    elif Y['Result'][i] in rate_one:
        Y_.append(1)
    else:
        print (Y['Result'][i])

outcomes = np.asarray(Y_)
Y_pred = pd.DataFrame(Y_, index = X.index)
unique, counts = np.unique(outcomes, return_counts=True)

#n_classes = int(Y.max()[0]) + 1
#print ('There are {} different classes'.format(n_classes))


## change column names
new_col_name = []
for col in X.columns:
    #print(type(col))
    new_col_name.append(col.replace('/ ', '').replace(' ', '_'))
    
X.columns = new_col_name

## output the headers from the csv file
X.keys()

data_type = []
for item_name in X.keys():
    data_type.append(type(X[item_name][0]))

print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ('The unique data types across all the items are:', set(data_type))
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

for item_name in X.keys():
    ## find the number of NaN in this item
    no = np.sum(X[item_name].isna().astype(int))
    #print ('The number of {} with value equal to NaN is {}'.format(item_name, no))
    
    ## Replace the missing value with corresponding values
    if no > 0:
        if type(X[item_name][0]) == np.float64:
            X[item_name].fillna(-1, inplace = True)
        else:
            X[item_name].fillna('unknown', inplace = True)
X['Crew_Size'].head()


#########################################################
#############   Latent Dirichlet Allocation   ###########
#########################################################
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import gensim
import string

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
stop_words = stopwords.words('english') + list(string.punctuation)

# Create p_stemmer of class PorterStemmer
p_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = PorterStemmer()

df = X

texts = []
index = df.columns.get_loc("Synopsis")

for i in range(df.shape[0]):
    if type(df.iloc[i, index]) is str:
         # clean and tokenize document string
        raw = df.iloc[i, index].lower()

        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stop_words]

        # stem tokens
        stemmed_tokens = [wordnet_lemmatizer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

# turn the tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]



num_topics = 200

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word = dictionary, 
                                           distributed = False, passes=20, eval_every=10)

num_words = 10
topics = ldamodel.print_topics(num_topics = num_topics, num_words = num_words)


import re
topic_words = []
all_topics = []
document_no = df.shape[0]
for i in range(document_no):
    doc_ID = i
    topics_document= ldamodel.get_document_topics(dictionary.doc2bow(texts[doc_ID]))
    print ('~~~~~~~~~~~~~~~~~~~~ The original synopsis report  ~~~~~~~~~~~~~~~~')
    print (df.iloc[doc_ID, index].lower())
    #print (topics_document)
    #print ('\n')

    
    topic_with_max_prob = max(topics_document, key=lambda item: item[1])
    topic_ID, document_topics = topics[topic_with_max_prob[0]][0], topics[topic_with_max_prob[0]][1]
    prob = topic_with_max_prob[1]
    print ('\n')
    print ('Topics mined from the synopsis:')
    print (document_topics, ' --- with prob:', topic_with_max_prob[1])
    
    list_words = []
    quoted = re.compile('"[^"]*"')
    for value in quoted.findall(document_topics):
        list_words.append(value.replace('"',''))
        topic_words.append(value.replace('"',''))
    
    all_topics.append(list_words)
    
import numpy as np

topic_rep = np.zeros(shape = (df.shape[0], num_words))
unique_topic_words = list(set(topic_words))

i = 0
for doc_topic in all_topics:
    for j in range(len(doc_topic)):
        loc = unique_topic_words.index(doc_topic[j])
        topic_rep[i, j] = loc
    i = i + 1
    
topic_rep

X['word_1'] = topic_rep[:, 0].astype(str)
X['word_2'] = topic_rep[:, 1].astype(str)
X['word_3'] = topic_rep[:, 2].astype(str)
X['word_4'] = topic_rep[:, 3].astype(str)
X['word_5'] = topic_rep[:, 4].astype(str)
X['word_6'] = topic_rep[:, 5].astype(str)
X['word_7'] = topic_rep[:, 6].astype(str)
X['word_8'] = topic_rep[:, 7].astype(str)
X['word_9'] = topic_rep[:, 8].astype(str)
X['word_10'] = topic_rep[:, 9].astype(str)
X.head()


## Location
Locale_Reference = tf.feature_column.categorical_column_with_hash_bucket('Locale_Reference', 
                                                                         hash_bucket_size = len(set(X['Locale_Reference'])))
State_Reference = tf.feature_column.categorical_column_with_hash_bucket('State_Reference', 
                                                                        hash_bucket_size = len(set(X['State_Reference'])))


## Environment
Flight_Conditions = tf.feature_column.categorical_column_with_hash_bucket('Flight_Conditions', 
                                                                hash_bucket_size = len(set(X['State_Reference'])))
Weather_Elements_Visibility = tf.feature_column.categorical_column_with_hash_bucket('Weather_Elements_Visibility', 
                                                            hash_bucket_size = len(set(X['Weather_Elements_Visibility'])))
Work_Environment_Factor = tf.feature_column.categorical_column_with_hash_bucket('Work_Environment_Factor', 
                                                            hash_bucket_size = len(set(X['Work_Environment_Factor'])))
Light = tf.feature_column.categorical_column_with_hash_bucket('Light', hash_bucket_size = len(set(X['Work_Environment_Factor'])))


## Aircraft
ATC_Advisory = tf.feature_column.categorical_column_with_hash_bucket('ATC_Advisory', 
                                                            hash_bucket_size = len(set(X['ATC_Advisory'])))
Aircraft_Operator = tf.feature_column.categorical_column_with_hash_bucket('Aircraft_Operator', 
                                                                hash_bucket_size = len(set(X['Aircraft_Operator'])))
Make_Model_Name = tf.feature_column.categorical_column_with_hash_bucket('Make_Model_Name', 
                                                            hash_bucket_size = len(set(X['Make_Model_Name'])))
Crew_Size = tf.feature_column.numeric_column('Crew_Size', [1])
Flight_Plan = tf.feature_column.categorical_column_with_hash_bucket('Flight_Plan', 
                                                            hash_bucket_size = len(set(X['Flight_Plan'])))
Mission = tf.feature_column.categorical_column_with_hash_bucket('Mission', 
                                                                hash_bucket_size = len(set(X['Mission'])))
Flight_Phase1 = tf.feature_column.categorical_column_with_hash_bucket('Flight_Phase1', 
                                                                      hash_bucket_size = len(set(X['Flight_Phase1'])))
Route_In_Use = tf.feature_column.categorical_column_with_hash_bucket('Route_In_Use', 
                                                                     hash_bucket_size = len(set(X['Route_In_Use'])))
Airspace = tf.feature_column.categorical_column_with_hash_bucket('Airspace', 
                                                                 hash_bucket_size = len(set(X['Airspace'])))

## Component
Aircraft_Component = tf.feature_column.categorical_column_with_hash_bucket('Aircraft_Component', 
                                                             hash_bucket_size = len(set(X['Aircraft_Component'])))
Manufacturer = tf.feature_column.categorical_column_with_hash_bucket('Manufacturer', 
                                                        hash_bucket_size = len(set(X['Manufacturer'])))

## Person
Location_Of_Person = tf.feature_column.categorical_column_with_hash_bucket('Location_Of_Person', 
                                                                hash_bucket_size = len(set(X['Location_Of_Person'])))
Location_In_Aircraft = tf.feature_column.categorical_column_with_hash_bucket('Location_In_Aircraft',
                                                            hash_bucket_size = len(set(X['Location_In_Aircraft'])))
Reporter_Organization = tf.feature_column.categorical_column_with_hash_bucket('Reporter_Organization',
                                                            hash_bucket_size = len(set(X['Reporter_Organization'])))
Function = tf.feature_column.categorical_column_with_hash_bucket('Function', hash_bucket_size = len(set(X['Function'])))
Qualification = tf.feature_column.categorical_column_with_hash_bucket('Qualification', 
                                                                      hash_bucket_size = len(set(X['Qualification'])))
Human_Factors = tf.feature_column.categorical_column_with_hash_bucket('Human_Factors', 
                                                                      hash_bucket_size = len(set(X['Human_Factors'])))

## Events
Anomaly = tf.feature_column.categorical_column_with_hash_bucket('Anomaly', 
                                                                hash_bucket_size = len(set(X['Anomaly'])))
Detector = tf.feature_column.categorical_column_with_hash_bucket('Detector', 
                                                                 hash_bucket_size = len(set(X['Detector'])))
When_Detected = tf.feature_column.categorical_column_with_hash_bucket('When_Detected', 
                                                                      hash_bucket_size = len(set(X['When_Detected'])))
Were_Passengers_Involved_In_Event = tf.feature_column.categorical_column_with_hash_bucket('Were_Passengers_Involved_In_Event',
                                                    hash_bucket_size = len(set(X['Were_Passengers_Involved_In_Event'])))

## Assessments
Contributing_Factors_Situations = tf.feature_column.categorical_column_with_hash_bucket('Contributing_Factors_Situations', 
                                                   hash_bucket_size = len(set(X['Contributing_Factors_Situations'])))
Primary_Problem = tf.feature_column.categorical_column_with_hash_bucket('Primary_Problem', 
                                                        hash_bucket_size = len(set(X['Primary_Problem'])))


## Topic Mining from Accident Synopsis
#word_1 = tf.feature_column.categorical_column_with_hash_bucket('word_1', hash_bucket_size = len(set(unique_topic_words)))
#word_2 = tf.feature_column.categorical_column_with_hash_bucket('word_2', hash_bucket_size = len(unique_topic_words))
#word_3 = tf.feature_column.categorical_column_with_hash_bucket('word_3', hash_bucket_size = len(unique_topic_words))
#word_4 = tf.feature_column.categorical_column_with_hash_bucket('word_4', hash_bucket_size = len(unique_topic_words))
#word_5 = tf.feature_column.categorical_column_with_hash_bucket('word_5', hash_bucket_size = len(unique_topic_words))
#word_6 = tf.feature_column.categorical_column_with_hash_bucket('word_6', hash_bucket_size = len(unique_topic_words))
#word_7 = tf.feature_column.categorical_column_with_hash_bucket('word_7', hash_bucket_size = len(unique_topic_words))
#word_8 = tf.feature_column.categorical_column_with_hash_bucket('word_8', hash_bucket_size = len(unique_topic_words))
#word_9 = tf.feature_column.categorical_column_with_hash_bucket('word_9', hash_bucket_size = len(unique_topic_words))
#word_10 = tf.feature_column.categorical_column_with_hash_bucket('word_10', hash_bucket_size = len(unique_topic_words))

## Place
Locale_Reference = tf.feature_column.embedding_column(Locale_Reference, len(set(X['Locale_Reference'])))
State_Reference = tf.feature_column.embedding_column(State_Reference, len(set(X['State_Reference'])))


## Environment
Flight_Conditions = tf.feature_column.embedding_column(Flight_Conditions,  len(set(X['Flight_Conditions'])))
Weather_Elements_Visibility = tf.feature_column.embedding_column(Weather_Elements_Visibility,  
                                                                 len(set(X['Weather_Elements_Visibility'])))
Work_Environment_Factor = tf.feature_column.embedding_column(Work_Environment_Factor,  len(set(X['Work_Environment_Factor'])))
Light = tf.feature_column.embedding_column(Light, len(set(X['Light'])))


## Aircraft
ATC_Advisory = tf.feature_column.embedding_column(ATC_Advisory, len(set(X['ATC_Advisory'])))
Aircraft_Operator = tf.feature_column.embedding_column(Aircraft_Operator, len(set(X['Aircraft_Operator'])))
Make_Model_Name = tf.feature_column.embedding_column(Make_Model_Name, len(set(X['Make_Model_Name'])))
Flight_Plan = tf.feature_column.embedding_column(Flight_Plan, len(set(X['Flight_Plan'])))
Mission = tf.feature_column.embedding_column(Mission, len(set(X['Mission'])))
Flight_Phase1 = tf.feature_column.embedding_column(Flight_Phase1, len(set(X['Flight_Phase1'])))
Route_In_Use = tf.feature_column.embedding_column(Route_In_Use, len(set(X['Route_In_Use'])))
Airspace = tf.feature_column.embedding_column(Airspace, len(set(X['Airspace'])))

## Component
Aircraft_Component = tf.feature_column.embedding_column(Aircraft_Component, len(set(X['Aircraft_Component'])))
Manufacturer = tf.feature_column.embedding_column(Manufacturer, len(set(X['Manufacturer'])))

## Person
Location_Of_Person = tf.feature_column.embedding_column(Location_Of_Person, len(set(X['Location_Of_Person'])))
Location_In_Aircraft = tf.feature_column.embedding_column(Location_In_Aircraft, len(set(X['Location_In_Aircraft'])))
Reporter_Organization = tf.feature_column.embedding_column(Reporter_Organization, len(set(X['Reporter_Organization'])))
Function = tf.feature_column.embedding_column(Function, len(set(X['Function'])))
Qualification = tf.feature_column.embedding_column(Qualification, len(set(X['Qualification'])))
Human_Factors = tf.feature_column.embedding_column(Human_Factors, len(set(X['Human_Factors'])))

## Events
Anomaly = tf.feature_column.embedding_column(Anomaly, len(set(X['Anomaly'])))
Detector = tf.feature_column.embedding_column(Detector, len(set(X['Detector'])))
When_Detected = tf.feature_column.embedding_column(When_Detected, len(set(X['When_Detected'])))
Were_Passengers_Involved_In_Event = tf.feature_column.embedding_column(Were_Passengers_Involved_In_Event,
                                                                       len(set(X['Were_Passengers_Involved_In_Event'])))

## Assessments
Contributing_Factors_Situations = tf.feature_column.embedding_column(Contributing_Factors_Situations,
                                                                     len(set(X['Contributing_Factors_Situations'])))
Primary_Problem = tf.feature_column.embedding_column(Primary_Problem, len(set(X['Primary_Problem'])))


## Topic Mining from Accident Synopsis
#word_1 = tf.feature_column.embedding_column(word_1, len(set(unique_topic_words)))
#word_2 = tf.feature_column.embedding_column(word_2, len(unique_topic_words))
#word_3 = tf.feature_column.embedding_column(word_3, len(unique_topic_words))
#word_4 = tf.feature_column.embedding_column(word_4, len(unique_topic_words))
#word_5 = tf.feature_column.embedding_column(word_5, len(unique_topic_words))
#word_6 = tf.feature_column.embedding_column(word_6, len(unique_topic_words))
#word_7 = tf.feature_column.embedding_column(word_7, len(unique_topic_words))
#word_8 = tf.feature_column.embedding_column(word_8, len(unique_topic_words))
#word_9 = tf.feature_column.embedding_column(word_9, len(unique_topic_words))
#word_10 = tf.feature_column.embedding_column(word_10, len(unique_topic_words))

from sklearn.model_selection import train_test_split
X_sub = X[['Locale_Reference', 'State_Reference', 'Flight_Conditions', 'Weather_Elements_Visibility', 
            'Work_Environment_Factor', 'Light', 'ATC_Advisory', 'Aircraft_Operator', 'Make_Model_Name', 
            'Crew_Size', 'Flight_Plan', 'Mission', 'Flight_Phase1',
            'Route_In_Use','Airspace', 'Aircraft_Component', 'Manufacturer', 'Location_Of_Person', 'Location_In_Aircraft',
            'Reporter_Organization', 'Function', 'Qualification', 'Human_Factors', 'Anomaly', 'Detector', 'When_Detected',
            'Were_Passengers_Involved_In_Event', 'Contributing_Factors_Situations', 'Primary_Problem']]

X_train, X_test, Y_train, Y_test = train_test_split(X_sub, Y_pred, test_size = 0.15)

label = []
for i in range(5):
    print ('Train the {} model, please keep waiting !!!'.format(i+1))
    print ('\n')
    
    X_train_set, X_test_tmp, Y_train_set, Y_test_tmp = train_test_split(X_train, Y_train, test_size = 0.2, random_state = i)

    ## define input function
    input_func = tf.estimator.inputs.pandas_input_fn(x = X_train_set, y = Y_train_set, batch_size = 500, 
                                                        num_epochs = 1000, shuffle = True)

    ## define the feature columns
    feat_cols = [Locale_Reference, State_Reference, Flight_Conditions, Weather_Elements_Visibility, Work_Environment_Factor, 
                     Light, ATC_Advisory, Aircraft_Operator, Make_Model_Name, Crew_Size, Flight_Plan, Mission, Flight_Phase1, 
                     Route_In_Use, Airspace, Aircraft_Component, Manufacturer, Location_Of_Person, Location_In_Aircraft, 
                     Reporter_Organization, Function, Qualification, Human_Factors, Anomaly, Detector, When_Detected, 
                     Were_Passengers_Involved_In_Event, Contributing_Factors_Situations, Primary_Problem]

    ## build the model
    model = tf.estimator.DNNClassifier(hidden_units = [40, 40, 40, 40, 40, 40, 40, 40], feature_columns = feat_cols,
                                           n_classes = 6, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))
        
    ## train the model
    model.train(input_fn = input_func, steps = 4000)
    
    
    ## make predictions
    eval_input = tf.estimator.inputs.pandas_input_fn(x = X_test, shuffle = False)
    prediction = list(model.predict(eval_input))

    pred_label = [int(pred['class_ids']) for pred in prediction]
    
    label.append(pred_label)
    
ensembel_pred = []
for j in range(len(label[0])):
    x = np.zeros(shape = (len(label), 1)) - 1
    for i in range(len(label)):
        x[i] =  label[i][j]
    (values, counts) = np.unique(x, return_counts=True)
    ind = np.argmax(counts)
    ensembel_pred.append((values[ind]))

ensembel_pred

from sklearn.metrics import classification_report
target_names = [str(i) for i in range(1, 5+1)]
print(classification_report(Y_test, ensembel_pred, target_names=target_names))