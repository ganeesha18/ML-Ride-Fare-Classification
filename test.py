# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


    
data = pd.read_csv('train.csv')  

columns_feature = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare','meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
for each in columns_feature:
    data[each]= data[each].fillna(data[each].mean())


data = data[['tripid', 'additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare', 'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare',
       'label']]

label_encoder_no_of_class = LabelEncoder()

label_encoder_no_of_class  =label_encoder_no_of_class.fit(data['label'])

data["label"] = label_encoder_no_of_class.transform(data["label"])


clf = RandomForestClassifier(oob_score=True)


_features = data.loc[:,'additional_fare':'fare']
_labels = data.loc[:,'label']


clf.fit(_features, _labels)
    
test_data = pd.read_csv('test.csv')

test_data = test_data[['tripid', 'additional_fare', 'duration', 'meter_waiting','meter_waiting_fare',
                       'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']]
test = test_data.loc[:,'additional_fare':'fare']
 
prediction = clf.predict(test)

prediction_original = list(label_encoder_no_of_class.inverse_transform(prediction))


############################ uncoment following code section to save results

#prediction_result = pd.read_csv('sample_submission.csv')
#prediction_result['prediction'] = prediction_original
#prediction_result = prediction_result.replace({'prediction': {'correct': 1,'incorrect': 0}})
#
#prediction_result.to_csv('cse_ml_Prediction.csv', index=False)






#df2['new'] = df2.apply(lambda row: distance_cal(row['a'],row['b'],row['c'],row['d']),axis=1)



