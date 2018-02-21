import pandas as pd
import numpy as np
import tensorflow as tf
from os import listdir

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
        Y_.append(9)
    elif Y['Result'][i] in rate_seven:
        Y_.append(7)
    elif Y['Result'][i] in rate_five:
        Y_.append(5)
    elif Y['Result'][i] in rate_three:
        Y_.append(3)
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



##########################################################
####################  Random Forest ######################
##########################################################
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

le = preprocessing.LabelEncoder()
Locale_Reference = le.fit_transform(X['Locale_Reference'])
le = preprocessing.LabelEncoder()
State_Reference = le.fit_transform(X['State_Reference'])
le = preprocessing.LabelEncoder()
Flight_Conditions = le.fit_transform(X['Flight_Conditions'])
le = preprocessing.LabelEncoder()
Weather_Elements_Visibility = le.fit_transform(X['Weather_Elements_Visibility'])
le = preprocessing.LabelEncoder()
Work_Environment_Factor = le.fit_transform(X['Work_Environment_Factor'])
le = preprocessing.LabelEncoder()
Light = le.fit_transform(X['Light'])
le = preprocessing.LabelEncoder()
ATC_Advisory = le.fit_transform(X['ATC_Advisory'])
le = preprocessing.LabelEncoder()
Aircraft_Operator = le.fit_transform(X['Aircraft_Operator'])
le = preprocessing.LabelEncoder()
Make_Model_Name = le.fit_transform(X['Make_Model_Name'])
le = preprocessing.LabelEncoder()
Crew_Size = le.fit_transform(X['Crew_Size'])
le = preprocessing.LabelEncoder()
Flight_Plan = le.fit_transform(X['Flight_Plan'])
le = preprocessing.LabelEncoder()
Mission = le.fit_transform(X['Mission'])
le = preprocessing.LabelEncoder()
Flight_Phase1 = le.fit_transform(X['Flight_Phase1'])
le = preprocessing.LabelEncoder()
Route_In_Use = le.fit_transform(X['Route_In_Use'])
le = preprocessing.LabelEncoder()
Airspace = le.fit_transform(X['Airspace'])
le = preprocessing.LabelEncoder()
Aircraft_Component = le.fit_transform(X['Aircraft_Component'])
le = preprocessing.LabelEncoder()
Manufacturer = le.fit_transform(X['Manufacturer'])
le = preprocessing.LabelEncoder()
Location_Of_Person = le.fit_transform(X['Location_Of_Person'])
le = preprocessing.LabelEncoder()
Location_In_Aircraft = le.fit_transform(X['Location_In_Aircraft'])
le = preprocessing.LabelEncoder()
Reporter_Organization = le.fit_transform(X['Reporter_Organization'])
le = preprocessing.LabelEncoder()
Function = le.fit_transform(X['Function'])
le = preprocessing.LabelEncoder()
Qualification = le.fit_transform(X['Qualification'])
le = preprocessing.LabelEncoder()
Human_Factors = le.fit_transform(X['Human_Factors'])
le = preprocessing.LabelEncoder()
Anomaly = le.fit_transform(X['Anomaly'])
le = preprocessing.LabelEncoder()
Detector = le.fit_transform(X['Detector'])
le = preprocessing.LabelEncoder()
When_Detected = le.fit_transform(X['When_Detected'])
le = preprocessing.LabelEncoder()
Were_Passengers_Involved_In_Event = le.fit_transform(X['Were_Passengers_Involved_In_Event'])
le = preprocessing.LabelEncoder()
Contributing_Factors_Situations = le.fit_transform(X['Contributing_Factors_Situations'])
le = preprocessing.LabelEncoder()
Primary_Problem = le.fit_transform(X['Primary_Problem'])


X_array = [Locale_Reference, State_Reference, Flight_Conditions, Weather_Elements_Visibility, 
            Work_Environment_Factor, Light, ATC_Advisory, Aircraft_Operator, Make_Model_Name, 
            Crew_Size, Flight_Plan, Mission, Flight_Phase1,
            Route_In_Use, Airspace, Aircraft_Component, Manufacturer, Location_Of_Person, Location_In_Aircraft,
            Reporter_Organization, Function, Qualification, Human_Factors, Anomaly, Detector, When_Detected,
            Were_Passengers_Involved_In_Event, Contributing_Factors_Situations, Primary_Problem]

X_sub = pd.DataFrame(np.transpose(X_array), index = Y_pred.index, columns = ['Locale_Reference', 'State_Reference', 'Flight_Conditions', 'Weather_Elements_Visibility', 
            'Work_Environment_Factor', 'Light', 'ATC_Advisory', 'Aircraft_Operator', 'Make_Model_Name', 
            'Crew_Size', 'Flight_Plan', 'Mission', 'Flight_Phase1',
            'Route_In_Use','Airspace', 'Aircraft_Component', 'Manufacturer', 'Location_Of_Person', 'Location_In_Aircraft',
            'Reporter_Organization', 'Function', 'Qualification', 'Human_Factors', 'Anomaly', 'Detector', 'When_Detected',
            'Were_Passengers_Involved_In_Event', 'Contributing_Factors_Situations', 'Primary_Problem'])


X_train, X_test, Y_train, Y_test = train_test_split(X_sub, Y_pred, test_size = 0.1)
clf = RandomForestClassifier(n_estimators=100, oob_score = True, max_depth=None, max_features = 5, min_samples_split=2)
clf.fit(X_train, Y_train)
scores = cross_val_score(clf, X_train, Y_train)
print (scores.mean())


predictions = clf.predict(X_test)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(Y_train, clf.predict(X_train)))
print ("Test Accuracy  :: ", accuracy_score(Y_test, predictions))
