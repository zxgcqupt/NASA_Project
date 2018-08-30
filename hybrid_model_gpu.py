
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import tensorflow as tf
from os import listdir
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#print ('The version of TensorFlow is {}'.format(tf.__version__))


# ## Load the 12-year incident/accident data from ASRS (Aviation Safety Reporting System)

# In[12]:


root_path = './Data'

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
count_multiple_outcome = 0
for index, row in Y_raw.iterrows():
    #print (index, row['Result'])
    outcome = row['Result']
    if type(outcome) == np.float:
        res = 'unknown'
        processed_Y.append([res])
    elif ';' in outcome:
        count_multiple_outcome += 1
        res = str(outcome).split(';')
        # remove the space at the beginning of each event outcome
        for i in range(len(res)):
            res[i] = res[i].strip()
        #print (res)
        processed_Y.append(res)
    else:
        res = outcome
        processed_Y.append([res])
        
X['res'] = processed_Y ## add the res column first for the use in the subsequent subcategory models


# In[13]:


count_multiple_outcome/X.shape[0]


# In[14]:


unique_anomaly = list(set(X['Anomaly']))
#for i in range(len(unique_anomaly)):
#    print (i, ": ", unique_anomaly[i])


# ## Perform risk-based event outcome cetegorization

# In[15]:


## compress the number of labels to be predicted --> map result to risk level
rate_five = ['General Declared Emergency', 'General Physical Injury / Incapacitation', 'Flight Crew Inflight Shutdown', 
             'Air Traffic Control Separated Traffic', 'Aircraft Aircraft Damaged']

rate_four = ['General Evacuated', 'Flight Crew Regained Aircraft Control', 
              'Air Traffic Control Issued Advisory / Alert', 'Flight Crew Landed in Emergency Condition',
              'Flight Crew Landed In Emergency Condition']

rate_three = ['General Work Refused', 'Flight Crew Became Reoriented', 'Flight Crew Diverted', 
             'Flight Crew Executed Go Around / Missed Approach', 
             'Flight Crew Overcame Equipment Problem', 'Flight Crew Rejected Takeoff', 'Flight Crew Took Evasive Action', 
             'Air Traffic Control Issued New Clearance']

rate_two = ['General Maintenance Action', 'General Flight Cancelled / Delayed', 
              'General Release Refused / Aircraft Not Accepted', 
              'Flight Crew Overrode Automation', 'Flight Crew FLC Overrode Automation',
              'Flight Crew Exited Penetrated Airspace', 
              'Flight Crew Requested ATC Assistance / Clarification', 'Flight Crew Landed As Precaution',
              'Flight Crew Returned To Clearance', 'Flight Crew Returned To Departure Airport',
              'Aircraft Automation Overrode Flight Crew']

rate_one = ['General Police / Security Involved', 'Flight Crew Returned To Gate', 'Aircraft Equipment Problem Dissipated', 
            'unknown', 'Air Traffic Control Provided Assistance',
            'General None Reported / Taken', 'Flight Crew FLC complied w / Automation / Advisory']

def risk_quantification(val):
    event_risk = []
    for i in range(len(val)):
        item = val[i].lstrip() ## remove the space at the start of each item
        if item in rate_five:
            event_risk.append(5)
        elif item in rate_four:
            event_risk.append(4)
        elif item in rate_three:
            event_risk.append(3)
        elif item in rate_two:
            event_risk.append(2)
        elif item in rate_one:
            event_risk.append(1)
    return max(event_risk)

Y_ = []
for i in range(len(processed_Y)):
    if len(processed_Y[i]) > 1:
        val = risk_quantification(processed_Y[i])
        Y_.append(val)
    else:
        item_val = "".join(processed_Y[i]) ## convert a list to a string
        #print (item_val)
        if item_val in rate_five:
            Y_.append(5)
        elif item_val in rate_four:
            Y_.append(4)
        elif item_val in rate_three:
            Y_.append(3)
        elif item_val in rate_two:
            Y_.append(2)
        elif item_val in rate_one:
            Y_.append(1)
        else:
            print (Y['Result'][i])

outcomes = np.asarray(Y_)
Y_true = pd.DataFrame(Y_, index = X.index, columns = ['Result'])
unique, counts = np.unique(outcomes, return_counts=True)
print (unique, counts)


# ## Up-sampling the minority classes

# In[16]:


from sklearn.utils import resample

data_rev = X.copy(deep=True)
data_rev['Result'] = Y_true

df_majority_1 = data_rev[data_rev['Result']==1]
df_majority_3 = data_rev[data_rev['Result']==3]
df_minority_2 = data_rev[data_rev['Result']==2]
df_minority_4 = data_rev[data_rev['Result']==4]
df_minority_5 = data_rev[data_rev['Result']==5]

# Upsample minority class
df_minority_2_upsampled = resample(df_minority_2, 
                                 replace=True,     # sample with replacement
                                 n_samples=18841,    # to match majority class
                                 random_state=145) # reproducible results
df_minority_4_upsampled = resample(df_minority_4, 
                                 replace=True,     # sample with replacement
                                 n_samples=18841,    # to match majority class
                                 random_state=145) # reproducible results
df_minority_5_upsampled = resample(df_minority_5, 
                                 replace=True,     # sample with replacement
                                 n_samples=18841,    # to match majority class
                                 random_state=145) # reproducible results

df_upsampled = pd.concat([df_majority_1, df_majority_3, df_minority_2_upsampled, df_minority_4_upsampled, 
                          df_minority_5_upsampled])

## reset the index of concatnated dataframe

df_upsampled.reset_index(drop=True)
df_upsampled['Result'].value_counts()


X = df_upsampled.drop(columns = 'Result')
Y_true = df_upsampled['Result']

unique, counts = np.unique(Y_true, return_counts=True)
print ('After the upsampling, the number of each item is: \n')
print (unique)
print (counts)


# In[17]:


## copy the data
X_org = X.copy(deep=True)
Y_org = Y_true.copy(deep=True)
X_org.shape


# ## Processing categorical data

# In[18]:


## change column names
new_col_name = []
for col in X_org.columns:
    #print(type(col))
    new_col_name.append(col.replace('/ ', '').replace(' ', '_'))
    
X_org.columns = new_col_name


data_type = []
for item_name in X_org.keys():
    data_type.append(type(X_org[item_name][0]))

print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ('The unique data types across all the items are:', set(data_type))
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

for item_name in X_org.keys():
    ## find the number of NaN in this item
    no = np.sum(X_org[item_name].isna().astype(int))
    #print ('The number of {} with value equal to NaN is {}'.format(item_name, no))
    
    ## Replace the missing value with corresponding values
    if no > 0:
        if type(X_org[item_name][0]) == np.float64:
            X_org[item_name].fillna(-1, inplace = True)
        else:
            X_org[item_name].fillna('unknown', inplace = True)
X_org['Crew_Size'].head()


# In[19]:


#####################################################################################
############### Construct classification report from confusion matrix ###############
#####################################################################################

np.set_printoptions(suppress=True)
def construct_classification_report(confusion_matrix):
    no_of_class = len(confusion_matrix)
    confusion_report = np.zeros((no_of_class, 4))
    
    for i in range(len(confusion_matrix)):
        confusion_report[i, 0] = confusion_matrix[i, i]/np.sum(confusion_matrix[:, i])
        confusion_report[i, 1] = confusion_matrix[i, i]/np.sum(confusion_matrix[i,:])
        confusion_report[i, 2] = 2 * (confusion_report[i, 0] * confusion_report[i, 1])/(confusion_report[i, 0] + confusion_report[i, 1])
        confusion_report[i, -1] = np.sum(confusion_matrix[i, :])
    
    return np.round(confusion_report, decimals = 3)


# ## Perform Cross Validation
# 
# ### Split the data, the data has three parts: 
# ##### X_train, Y_train: train the data
# ##### X_validation, Y_validation: trial data to obtain the performance metrics
# ##### X_test, Y_test: test data used to compare the performance of hybrid model with SVM and DNN

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

test_random_state = 111
cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = test_random_state)

test_size_ratio = 0.06
random_split_seed = 200
if __name__ == "__main__":
  for k, (data_index, test_index) in enumerate(cv.split(X_org, Y_org)):
      print ('current fold: ', k+1)
      
      if k >= 3:
        ### Split the data into three parts: 
        ### X_train, Y_train: train the data
        ### X_validation, Y_validation: trial data to obtain the performance metrics
        ### X_test, Y_test: test data used to compare the performance of hybrid model with SVM and DNN
        
        X = X_org.iloc[data_index]
        Y = Y_org.iloc[data_index]
        
        X_test = X_org.iloc[test_index]
        Y_test = Y_org.iloc[test_index]
        
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = test_size_ratio, 
                                                        random_state = random_split_seed + i)
        
        ###########################################################
        ################# Support Vector Machine ##################
        ###########################################################
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', SGDClassifier(loss='epsilon_insensitive', penalty='l2',
                                                    alpha=1e-5, random_state=40,
                                                    max_iter=10, tol=None)),
                            ])


        parameters = {'clf__loss': ['epsilon_insensitive', 'hinge', 'log', 'huber', 'modified_huber', 'perceptron', 
                                    'squared_loss', 'squared_epsilon_insensitive', 'squared_hinge'],
                      'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5),
                      'clf__penalty': ['l1', 'l2', 'elasticnet'],
                      'clf__max_iter': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150)
         }

        optimal_parameters = {'clf__loss': ['modified_huber'],
                      'vect__ngram_range':  [(1, 2)],
                      'tfidf__use_idf': [True],
                      'clf__alpha': [1e-5],
                      'clf__penalty': ['elasticnet'],
                      'clf__max_iter': [80],
         }

        gs_clf = GridSearchCV(text_clf, optimal_parameters, n_jobs=-1)

        gs_clf.fit(X_train['Synopsis'], Y_train)
        pred_label_SVM = gs_clf.predict(X_validation['Synopsis'])

        #from sklearn.metrics import classification_report
        #target_names = [str(i) for i in range(1, 6)]
        #print(classification_report(Y_validation, pred_label_SVM, target_names=target_names))
        
        print ('Accuracy: ', np.sum(np.equal(Y_validation, pred_label_SVM).astype(int))/len(Y_validation))
        print ('The best set of parameters is \n', gs_clf.best_params_)
        
        #######################################################################
        ################## Construct confusion matrix for SVM #################
        #######################################################################
        SVM_confusion_matrix = confusion_matrix(Y_validation, pred_label_SVM)
        model_SVM = construct_classification_report(SVM_confusion_matrix)
        
        
        #######################################################################
        ######################## Deep Neural Network ##########################
        #######################################################################
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
        Light = tf.feature_column.categorical_column_with_hash_bucket('Light', hash_bucket_size = 
                                                                      len(set(X['Work_Environment_Factor'])))
        
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
        
        
        ### start training the model
        X_sub = X[['Locale_Reference', 'State_Reference', 'Flight_Conditions', 'Weather_Elements_Visibility', 
                    'Work_Environment_Factor', 'Light', 'ATC_Advisory', 'Aircraft_Operator', 'Make_Model_Name', 
                    'Crew_Size', 'Flight_Plan', 'Mission', 'Flight_Phase1',
                    'Route_In_Use','Airspace', 'Aircraft_Component', 'Manufacturer', 'Location_Of_Person', 'Location_In_Aircraft',
                    'Reporter_Organization', 'Function', 'Qualification', 'Human_Factors', 'Anomaly', 'Detector', 'When_Detected',
                    'Were_Passengers_Involved_In_Event', 'Contributing_Factors_Situations', 'Primary_Problem']]

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_sub, Y, test_size = test_size_ratio, 
                                                            random_state = random_split_seed + i)

        ## extract the test data
        X_test_sub = X_test[['Locale_Reference', 'State_Reference', 'Flight_Conditions', 'Weather_Elements_Visibility', 
                    'Work_Environment_Factor', 'Light', 'ATC_Advisory', 'Aircraft_Operator', 'Make_Model_Name', 
                    'Crew_Size', 'Flight_Plan', 'Mission', 'Flight_Phase1',
                    'Route_In_Use','Airspace', 'Aircraft_Component', 'Manufacturer', 'Location_Of_Person', 'Location_In_Aircraft',
                    'Reporter_Organization', 'Function', 'Qualification', 'Human_Factors', 'Anomaly', 'Detector', 'When_Detected',
                    'Were_Passengers_Involved_In_Event', 'Contributing_Factors_Situations', 'Primary_Problem']]
        
        
        label_trial = []
        label_test = []
        number_models = 10
        tf.reset_default_graph()
        for i in range(number_models):
            print ('\n\n')
            print ('Train the {} model, please keep waiting !!!'.format(i+1))
            
            X_train_set, X_test_tmp, Y_train_set, Y_test_tmp = train_test_split(X_train, Y_train, test_size = 0.15, 
                                                                                random_state = 20 + i)
            
            ## define input function
            input_func = tf.estimator.inputs.pandas_input_fn(x = X_train_set, y = Y_train_set, batch_size = 800, 
                                                                num_epochs = 600, shuffle = True)

            ## define the feature columns
            feat_cols = [Locale_Reference, State_Reference, Flight_Conditions, Weather_Elements_Visibility, Work_Environment_Factor, 
                             Light, ATC_Advisory, Aircraft_Operator, Make_Model_Name, Crew_Size, Flight_Plan, Mission, Flight_Phase1, 
                             Route_In_Use, Airspace, Aircraft_Component, Manufacturer, Location_Of_Person, Location_In_Aircraft, 
                             Reporter_Organization, Function, Qualification, Human_Factors, Anomaly, Detector, When_Detected, 
                             Were_Passengers_Involved_In_Event, Contributing_Factors_Situations, Primary_Problem]

            ## build the model
            model = tf.estimator.DNNClassifier(hidden_units = [24, 12], feature_columns = feat_cols,
                                               n_classes = 6, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

            ## train the model
            model.train(input_fn = input_func, steps = 4000)


            ## make predictions on the trial test data
            eval_input = tf.estimator.inputs.pandas_input_fn(x = X_validation, shuffle = False)
            prediction = list(model.predict(eval_input))
            pred_label = [int(pred['class_ids']) for pred in prediction]
            label_trial.append(pred_label)


            ## make predictions on the test data
            eval_input = tf.estimator.inputs.pandas_input_fn(x = X_test_sub, shuffle = False)
            prediction = list(model.predict(eval_input))
            pred_label = [int(pred['class_ids']) for pred in prediction]
            label_test.append(pred_label)
        
        
        #######################################################################
        ################## Construct confusion matrix for DNN #################
        #######################################################################
        ensembel_trial_pred = []
        for j in range(len(label_trial[0])):
            x = np.zeros(shape = (len(label_trial), 1)) - 1
            for i in range(len(label_trial)):
                x[i] =  label_trial[i][j]
            (values, counts) = np.unique(x, return_counts=True)
            ind = np.argmax(counts)
            ensembel_trial_pred.append((values[ind]))

        ## DNN confusion matrix
        DNN_confusion_matrix = confusion_matrix(Y_validation, ensembel_trial_pred)
        model_NN = construct_classification_report(DNN_confusion_matrix)
        
        
        #######################################################################
        ######################### Building hybrid model #######################
        #######################################################################
        validation_list = list(Y_validation)
        dict_count = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        common_count = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        for i in range(len(ensembel_trial_pred)):
            if ensembel_trial_pred[i] == pred_label_SVM[i]:
                dict_count[ensembel_trial_pred[i]] += 1
                if ensembel_trial_pred[i] == validation_list[i]:
                    common_count[ensembel_trial_pred[i]] += 1
                    
        print ('common count', common_count)
        print ('dict_count', dict_count)
                    
        accuracy = []
        for (key, val) in dict_count.items():
            common_acuracy = common_count[key]/float(dict_count[key])
            #print ('common_acuracy val ', common_count[key])
            #print ('dict value ', dict_count[key])
            #print (common_count[key]/float(dict_count[key]))
            #print ('common_acuracy', common_acuracy)
            accuracy.append(common_acuracy)
        accuracy = np.array(accuracy)
        
        print ('accuracy', accuracy)
        
        ## The predictions from the trained SVM model on the test data
        pred_label_test_SVM = gs_clf.predict(X_test['Synopsis'])
        SVM_prob = gs_clf.predict_proba(X_test['Synopsis'])

        ## The predictions from deep learning ensemble on the test data
        ensembel_test_pred = []
        ensembel_prob = []
        ensembel_prob_full = []
        for j in range(len(label_test[0])):
            x = np.zeros(shape = (len(label_test), 1)) - 1
            for i in range(len(label_test)):
                x[i] =  label_test[i][j]
            (values, counts) = np.unique(x, return_counts=True)
            #print (values, counts)
            prob_tmp = np.zeros(shape = 5)

            for j in range(len(values)):
                prob_tmp[int(values[j]-1)] = counts[j]/10.0
            ensembel_prob_full.append(prob_tmp)    
            #print (prob_tmp)

            ind = np.argmax(counts)
            ensembel_test_pred.append((values[ind]))
            ensembel_prob.append(counts[ind]/10.0)

        ensembel_prob_full = np.array(ensembel_prob_full)
        
        
        ## Blend the predictions from the two models
        final_pred = []

        total_unidentified = 0.0
        proportion = []
        for i in range(5):
            proportion.append(model_NN[i][3] - accuracy[i]*dict_count[i + 1])
            total_unidentified += model_NN[i][3] - accuracy[i]*dict_count[i + 1]
        proportion = np.array(proportion/total_unidentified)
        
        print ('proportion of disagreed records', proportion)
        proportion = proportion/0.2
        

        total_count = 0
        count = 0
        count_class = 0
        ### Compute the confusion matrix from the validation dataset
        from sklearn.preprocessing import normalize
        from sklearn.metrics import confusion_matrix
        confusion_validation_SVM = confusion_matrix(Y_validation, pred_label_SVM)
        normed_matrix_SVM = normalize(confusion_validation_SVM, axis=0, norm='l1')
        confusion_validation_DNN = confusion_matrix(Y_validation, ensembel_trial_pred)
        normed_matrix_DNN = normalize(confusion_validation_DNN, axis=0, norm='l1')

        count_SVM = 0
        count_DNN = 0
        count_SVM_correct = 0
        count_DNN_correct = 0
        for i in range(len(ensembel_test_pred)):
            if ensembel_test_pred[i] == pred_label_test_SVM[i]:
                final_pred.append(ensembel_test_pred[i])
                if ensembel_test_pred[i] == 2:
                    count_class += 1
            else:
                total_count += 1

                #################  Method 3  ###################
                productSVM = np.multiply(SVM_prob[i], proportion)
                productSVM = productSVM/np.sum(productSVM)
                svm_prob_i = np.dot(normed_matrix_SVM, np.multiply(SVM_prob[i], proportion))
                svm_prob_i = svm_prob_i/np.sum(svm_prob_i)
                #print ('SVM prob: ---------------->  ', svm_prob_i)

                productDNN = np.multiply(ensembel_prob_full[i], proportion)
                productDNN = productDNN/np.sum(productDNN)
                dnn_prob_i = np.dot(normed_matrix_DNN, productDNN)
                #print ('DNN prob: ---------------->  ', np.sum(dnn_prob_i))

                if np.max(svm_prob_i) > np.max(dnn_prob_i):
                    final_pred.append(np.argmax(svm_prob_i)+1)
                    if np.argmax(svm_prob_i) + 1 == Y_test[i]:
                        count += 1
                        count_SVM_correct += 1
                    count_SVM += 1
                else:
                    final_pred.append(np.argmax(dnn_prob_i)+1)
                    if np.argmax(dnn_prob_i) + 1 == Y_test[i]:
                        count += 1
                        count_DNN_correct += 1
                    count_DNN += 1
                    
        ### Report performance #####   
        target_names = [str(i) for i in range(1, 6)]         
        print ('Classification report on Hybrid model:')
        print(classification_report(Y_test, final_pred, target_names=target_names))
        
        print ('Classification report on SVM:')
        print(classification_report(Y_test, pred_label_test_SVM, target_names=target_names))
        
        print ('Classification report on deep learning:')
        print(classification_report(Y_test, ensembel_test_pred, target_names=target_names))




