# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle as pickle 
import sys
import os
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from math import sin, cos, sqrt, atan2, radians

def distance_cal(a,b,c,d):    
# approximate radius of earth in km
    R = 6373.0
    lat1 = radians(a)
    lon1 = radians(b)
    lat2 = radians(c)
    lon2 = radians(d)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c

    return distance


data_path = 'C:/IQ/msc/ml/'

    
data = pd.read_csv(data_path+'train.csv')  
data['pickup_time'] =  pd.to_datetime(data['pickup_time'], format='%m/%d/%Y %H:%M')
data['drop_time'] =  pd.to_datetime(data['drop_time'], format='%m/%d/%Y %H:%M')
data['interval_time'] = data['drop_time'] - data['pickup_time']
data['interval_time'] = data['interval_time'].apply(lambda x: x.total_seconds())/60

columns_feature = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare','interval_time', 'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
for each in columns_feature:
    data[each]= data[each].fillna(data[each].mean())

data['distance'] = data.apply(lambda row: distance_cal(row['pick_lat'],row['pick_lon'],row['drop_lat'],row['drop_lon']),axis=1)


#data['interval_time'] = data['interval_time'].total_seconds()
data = data[['tripid', 'additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare','interval_time', 'meter_waiting_till_pickup','distance', 'fare',
       'label']]
#
label_encoder_no_of_class = LabelEncoder()

label_encoder_no_of_class  =label_encoder_no_of_class.fit(data['label'])

data["label"] = label_encoder_no_of_class.transform(data["label"])

# Utility function to report best scores
def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

print('step 1')
clf = RandomForestClassifier()

#
#param_dist = {"clf__max_features": sp_randint(2, 10),
#              "clf__max_depth": sp_randint(160,300),
#              "clf__bootstrap": [True, False],
#              'clf__warm_start':[True, False],
#              "clf__n_estimators":sp_randint(160, 170),
#              'clf__min_samples_split' : sp_randint(2,6),
#              "clf__min_samples_leaf": sp_randint(2,10),
#              "clf__max_samples":sp_randint(2,10),
#              "clf__criterion": ["gini","entropy"]}

param_dist = {"clf__max_features": sp_randint(1, 8),
              "clf__bootstrap": [True, False],
              'clf__warm_start':[True, False],
              "clf__n_estimators":sp_randint(50, 150),
              'clf__min_samples_split' : sp_randint(2,20),
              "clf__criterion": ["gini","entropy"]}

pipeline = Pipeline([
        ('clf', clf)
    ])
print('step 2')    
# run randomized search
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                   n_iter=100, error_score=0.0, random_state=0)
_features = data.loc[:,'additional_fare':'fare']
_labels = data.loc[:,'label']
#clf.fit(_features, _labels)
#SMOTE
#oversampler=SMOTE(random_state=0, k_neighbors=5)
#_features,_labels=oversampler.fit_sample(_features, _labels)

random_search.fit(_features, _labels)
#report(random_search.cv_results_)

print('step 3')    
test_data = pd.read_csv(data_path+'test.csv')
test_data['pickup_time'] =  pd.to_datetime(test_data['pickup_time'], format='%m/%d/%Y %H:%M')
test_data['drop_time'] =  pd.to_datetime(test_data['drop_time'], format='%m/%d/%Y %H:%M')
test_data['interval_time'] = test_data['drop_time'] - test_data['pickup_time']
test_data['interval_time'] = test_data['interval_time'].apply(lambda x: x.total_seconds())/60

test_data['distance'] = test_data.apply(lambda row: distance_cal(row['pick_lat'],row['pick_lon'],row['drop_lat'],row['drop_lon']),axis=1)

test_data = test_data[['tripid', 'additional_fare', 'duration', 'meter_waiting','meter_waiting_fare','interval_time',
                       'meter_waiting_till_pickup','distance', 'fare']]
test = test_data.loc[:,'additional_fare':'fare']
print('step 4')   
prediction = random_search.best_estimator_.predict(test)
print(random_search.best_score_)

###random_search= pickle.load(open("classifier.pkl","rb"))
pickle.dump(random_search, open("classifier4.pkl","wb"))

prediction_original = list(label_encoder_no_of_class.inverse_transform(prediction))

prediction_result = pd.read_csv(data_path+'sample_submission.csv')
prediction_result['prediction'] = prediction_original
prediction_result = prediction_result.replace({'prediction': {'correct': 1,'incorrect': 0}})

prediction_result.to_csv('cse_ml_Prediction4.csv', index=False)






#df2['new'] = df2.apply(lambda row: distance_cal(row['a'],row['b'],row['c'],row['d']),axis=1)

