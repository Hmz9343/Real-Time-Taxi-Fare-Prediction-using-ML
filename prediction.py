import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from math import *
import pickle

dataset = pd.read_csv("/home/hamza/new_app/data/train.csv",nrows=100)

#dataset = dataset.append({'key' : 'xyz' ,'fare_amount': 44.5, 'pickup_datetime' : '2012-04-21 04:30:42 UTC', 'pickup_longitude' : -73.9871, 'pickup_latitude' : 40.7331, 'dropoff_longitude' : -73.9916, 'dropoff_latitude' : 40.7581, 'passenger_count': 1} , ignore_index=True)



"""Convertime to each aspects of time"""
dataset['pickup_datetime']=pd.to_datetime(dataset['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

dataset['pickup_date']= dataset['pickup_datetime'].dt.date
dataset['pickup_day']=dataset['pickup_datetime'].apply(lambda x:x.day)
dataset['pickup_hour']=dataset['pickup_datetime'].apply(lambda x:x.hour)
dataset['pickup_day_of_week']=dataset['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
dataset['pickup_month']=dataset['pickup_datetime'].apply(lambda x:x.month)
dataset['pickup_year']=dataset['pickup_datetime'].apply(lambda x:x.year)

# print(dataset.info())
"""Removing NaN values"""
dataset = dataset.dropna(how = 'any', axis = 'rows')
# print(dataset.isnull().sum())

"""Distribution of fare_amount"""
import seaborn as sns
plt.figure(figsize=(8,5))
sns.kdeplot(dataset['fare_amount']).set_title("Distribution of Trip Fare")

"""Negative Fares"""
dataset = dataset[dataset.fare_amount>0]

"""Setting 0 fare for no passengers"""
#dataset.loc[dataset['fare_amount'] != 0 & dataset['passenger_count'] == 0, 'fare_amount'] = 0

def abs_diff(df):
    df['abs_longitude_diff'] = (df.pickup_longitude-df.dropoff_longitude).abs()
    df['abs_latitude_diff'] = (df.pickup_latitude-df.dropoff_latitude).abs()
abs_diff(dataset)

"""distribution curve for coordinates diff"""
# dataset.iloc[:2000].plot.scatter(['abs_longitude_diff'], ['abs_latitude_diff'])
# plt.show()

dataset = dataset[(dataset['abs_latitude_diff']<5.0) & (dataset['abs_longitude_diff']<5.0)]
 	
"""Trip Distance in kms"""
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))

dataset['trip_distance'] = distance(dataset['pickup_latitude'],dataset['dropoff_latitude'],dataset['pickup_longitude'],dataset['dropoff_longitude'])
dataset.head()

X = dataset.iloc[:,[7,9,10,11,12,13,14,15,16]].values
y = dataset.iloc[:,[1]].values



"""Encoding categorical variables"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray().astype(int)

"""Dummy Variable Trap"""
X = X[:,1:]

test_input = X[-1,:]
test_input = test_input.reshape(1,-1)

"""Removing last row from X and y"""
X_train = X[:-1,:]
y_train = y[:-1,:]

# """Splitting"""
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

"""Regresson to training set"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

pickle.dump(regressor, open("model.pkl","wb"))
#haha=pickle.load(open("model.pkl","rb"))
#y_pred = haha.predict(test_input)


"""Random Forest"""
#from sklearn.ensemble import RandomForestRegressor
#regressor1 = RandomForestRegressor(n_estimators = 1000)
#regressor1.fit(X_train, y_train)
#y_pred1 = regressor1.predict(test_input)

# from sklearn.metrics import mean_squared_error

# rms = sqrt(mean_squared_error(y_test, y_pred))
# rms1 = sqrt(mean_squared_error(y_test, y_pred1))

#print("Linear Regression ",y_pred)
#print("Random Forest ",y_pred1)