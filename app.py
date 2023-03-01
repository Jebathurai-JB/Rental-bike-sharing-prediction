import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, url_for, render_template, request


def get_season(data):
	_, month = get_year_and_month(data)
	if 1 <= month <= 3:
		return 1
	if 4 <= month <= 6:
		return 2
	if 7 <= month <= 9:
		return 3
	else:
		return 4


def get_year_and_month(data):	
	year = int(data[0].split('-')[0][2:])
	year = 0 if year == 2011 else 1
	month = int(data[0].split('-')[1])	
	return year, month


def get_weekday(data):
	try:
		date = datetime.strptime(data[0], '%Y-%m-%dT%H:%M').date()
	except:
		date = datetime.strptime(data[0], '%Y-%m-%d').date()
	weekday = int(date.isoweekday())
	if weekday == 7:
		return 0
	return weekday
	


def get_workingday(holiday, weekday):
	if weekday == 0 or weekday == 6 or holiday == 1:
		return 0
	else:
		return 1
	

def get_hour(data):
	date = datetime.strptime(data[0], '%Y-%m-%dT%H:%M')
	return date.hour



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():

	if request.method == 'POST':

		data_form = request.form
		print(data_form)

		data = [data_form[i] for i in data_form]
		print(data[0])
		season = get_season(data)
		year, month = get_year_and_month(data)
		holiday = int(data[1])
		weekday = get_weekday(data)
		workingday = get_workingday(holiday, weekday)
		weather_sit = int(data[2])
		temp = float(data[3])
		hum = float(data[4])
		wind = float(data[5])

		if 'day' in data_form:
			category = 'day'
			record = np.array([season, year, month, holiday, weekday, workingday, weather_sit, temp, hum, wind])
			record = record.reshape(1, -1)		
	
			model_accuracy = pd.read_csv('model_accuracy.csv', index_col='Unnamed: 0')
			day_model_name = model_accuracy['Day'].idxmax()

			day_model = pickle.load(open(f'saved models/day_models/{day_model_name}.pkl', 'rb'))
			prediction = day_model.predict(record)
			

		else:
			category = 'hour'
			hour = get_hour(data)
			record = np.array([season, year, month, hour, holiday, weekday, workingday, weather_sit, temp, hum, wind])
			print(record)
			record = record.reshape(1, -1)
			model_accuracy = pd.read_csv('model_accuracy.csv', index_col='Unnamed: 0')
			hour_model_name = model_accuracy['Hour'].idxmax()

			hour_model = pickle.load(open(f'saved models/hour_models/{hour_model_name}.pkl', 'rb'))
			prediction = hour_model.predict(record)
			print(prediction)
			

		return render_template('home.html', prediction=prediction[0], category=category)
	
	else:
		return render_template('home.html')


if __name__ == "__main__":
	app.run(debug=True)