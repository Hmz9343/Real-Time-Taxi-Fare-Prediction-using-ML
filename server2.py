from prediction3 import taxi
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)


@app.route('/')
def student():
   return render_template('get_data_map.html')

@app.route('/predict', methods=['POST'])
def predict():
	key='xyz'
	fare_amount='44.5'
	pickup_datetime= request.form['Pickup_Datetime']
	pickup_longitude= request.form['Pickup_Longitude']
	pickup_latitude=request.form['Pickup_Latitude']
	dropoff_longitude=request.form['Dropoff_Longitude']
	dropoff_latitude=request.form['Dropoff_Latitude']
	passenger_count=request.form['Passenger_Count']
	
	json_string='{"key" : "'+key+'" ,"fare_amount": '+fare_amount+', "pickup_datetime" : "'+pickup_datetime+'", "pickup_longitude" : '+pickup_longitude+', "pickup_latitude" : '+pickup_latitude+', "dropoff_longitude" : '+dropoff_longitude+', "dropoff_latitude" : '+dropoff_latitude+', "passenger_count": '+passenger_count+'}'
	print(json_string)
	#json_string='{"key" : "xyz" ,"fare_amount": 44.5, "pickup_datetime" : "2012-04-21 04:30:42 UTC", "pickup_longitude" : -73.9871, "pickup_latitude" : 40.7331, "dropoff_longitude" : -73.9916, "dropoff_latitude" : 40.7581, "passenger_count": 1}'
	model=pickle.load(open("model2.pkl","rb"))
	
	fare1=taxi()
	
	prediction=model.predict_func(json_string)
	output=prediction[0]
	
	return jsonify(output)

if __name__ == '__main__':
	print("dslfasd")
	haha=pickle.load(open("model2.pkl","rb"))
	app.run(debug = True)