import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from math import *
import pickle
def predict():
	dataset = pd.read_csv("/home/hamza/new_app/data/train.csv",nrows=100)
	#dataset = dataset.append({'key' : 'xyz' ,'fare_amount': 44.5, 'pickup_datetime' : '2012-04-21 04:30:42 UTC', 'pickup_longitude' : -73.9871, 'pickup_latitude' : 40.7331, 'dropoff_longitude' : -73.9916, 'dropoff_latitude' : 40.7581, 'passenger_count': 1} , ignore_index=True)
	dataset['pickup_datetime']=pd.to_datetime(dataset['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
	dataset['pickup_date']= dataset['pickup_datetime'].dt.date
	dataset['pickup_day']=dataset['pickup_datetime'].apply(lambda x:x.day)
	dataset['pickup_hour']=dataset['pickup_datetime'].apply(lambda x:x.hour)
	dataset['pickup_day_of_week']=dataset['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
	dataset['pickup_month']=dataset['pickup_datetime'].apply(lambda x:x.month)
	dataset['pickup_year']=dataset['pickup_datetime'].apply(lambda x:x.year)
	dataset = dataset.dropna(how = 'any', axis = 'rows')

	import seaborn as sns
	plt.figure(figsize=(8,5))
	sns.kdeplot(dataset['fare_amount']).set_title("Distribution of Trip Fare")
	
	dataset = dataset[dataset.fare_amount>0]
	
	def abs_diff(df):
	    df['abs_longitude_diff'] = (df.pickup_longitude-df.dropoff_longitude).abs()
	    df['abs_latitude_diff'] = (df.pickup_latitude-df.dropoff_latitude).abs()

	abs_diff(dataset)

	dataset = dataset[(dataset['abs_latitude_diff']<5.0) & (dataset['abs_longitude_diff']<5.0)]
 	def distance(lat1,lat2,lon1,lon2):
		p = 0.017453292519943295 # Pi/180
    	a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    	return 12742 * np.arcsin(np.sqrt(a))
	
	dataset['trip_distance'] = distance(dataset['pickup_latitude'],dataset['dropoff_latitude'],dataset['pickup_longitude'],dataset['dropoff_longitude'])
	dataset.head()

	X = dataset.iloc[:,[7,9,10,11,12,13,14,15,16]].values
	y = dataset.iloc[:,[1]].values

	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_X = LabelEncoder()
	X[:,3] = labelencoder_X.fit_transform(X[:,3])
	onehotencoder = OneHotEncoder(categorical_features = [3])
	X = onehotencoder.fit_transform(X).toarray().astype(int)

	X = X[:,1:]

	test_input = X[-1,:]
	test_input = test_input.reshape(1,-1)
	
	X_train = X[:-1,:]
	y_train = y[:-1,:]
	
	from sklearn.linear_model import LinearRegression
	regressor = LinearRegression()
	regressor.fit(X_train,y_train)
	
	y_pred = regressor.predict(X_test)

	pickle.dump(regressor, open("model.pkl","wb"))
	