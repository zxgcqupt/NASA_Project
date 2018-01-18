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

X = data.drop(columns = 'Result')
Y_raw = pd.DataFrame(data['Result'])

processed_Y = []
for index, row in Y_raw.iterrows():
    #print (index, row['Result'])
    outcome = row['Result']
    if type(outcome) == np.float:
        res = 'unknown'
    elif ';' in outcome:
        res = str(outcome).split(';')[0]
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


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = pd.DataFrame(le.fit_transform(Y), index = X.index)
        
n_classes = int(Y.max()[0]) + 1


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

print ('\n')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ('The unique data types across all the items are:', set(data_type))
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

for item_name in X.keys():
    ## find the number of NaN in this item
    no = np.sum(X[item_name].isna().astype(int))
    #print ('The number of {} with value equal to NaN is {}'.format(item_name, no))
    
    ## Replace the missing value with corresponding values
    if no > 0:
        X[item_name].fillna('unknown', inplace = True)
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


from sklearn.model_selection import train_test_split
X_sub = X[['Locale_Reference', 'State_Reference', 'Flight_Conditions', 'Weather_Elements_Visibility', 
            'Work_Environment_Factor', 'Light', 'ATC_Advisory', 'Aircraft_Operator', 'Make_Model_Name', 
            'Crew_Size', 'Flight_Plan', 'Mission', 'Flight_Phase1',
            'Route_In_Use','Airspace', 'Aircraft_Component', 'Manufacturer', 'Location_Of_Person', 'Location_In_Aircraft',
            'Reporter_Organization', 'Function', 'Qualification', 'Human_Factors', 'Anomaly', 'Detector', 'When_Detected',
            'Were_Passengers_Involved_In_Event', 'Contributing_Factors_Situations', 'Primary_Problem' ]]

X_train, X_test, Y_train, Y_test = train_test_split(X_sub, Y, test_size = 0.2, random_state = 100)
X_train.head()

## define input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = Y_train, batch_size = 50, 
                                                num_epochs = 2000, shuffle = True)

## define the feature columns
feat_cols = [Locale_Reference, State_Reference, Flight_Conditions, Weather_Elements_Visibility, Work_Environment_Factor, 
             Light, ATC_Advisory, Aircraft_Operator, Make_Model_Name, Crew_Size, Flight_Plan, Mission, Flight_Phase1, 
             Route_In_Use, Airspace, Aircraft_Component, Manufacturer, Location_Of_Person, Location_In_Aircraft, 
             Reporter_Organization, Function, Qualification, Human_Factors, Anomaly, Detector, When_Detected, 
             Were_Passengers_Involved_In_Event, Contributing_Factors_Situations, Primary_Problem]

## build the model
model = tf.estimator.DNNClassifier(hidden_units = [35, 35, 35, 35, 35, 35, 35], feature_columns = feat_cols,
                                   n_classes = n_classes, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

## train the model
model.train(input_fn = input_func, steps = 4000)
