print('Xgboost test begin!\n')
import pandas as pd
import numpy as np
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
x = titanic[['pclass','age','sex']]
y = titanic['survived']

x['age'].fillna(x['age'].mean(),inplace=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train,y_train)
# y_predict = random_forest_classifier.predict(x_test)
score1 = random_forest_classifier.score(x_test, y_test)

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
# y_predict_xgbc = xgbc.predict(x_test)
score_xgbc = random_forest_classifier.score(x_test, y_test)
print('Score of random forest classifier:  %.5f'%score1)
print('Score of XGBClassifier           :  %.5f'%score_xgbc)

print('\nXgboost test end!')