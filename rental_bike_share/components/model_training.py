import os
import sys
import pickle
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from rental_bike_share.logger import logging
from rental_bike_share.exception import BikeSharingException

class ModelTrainer:
	def __init__(self, day_path, hour_path):
		try:
			self.day_path = day_path
			self.hour_path = hour_path

		except Exception as e:
			raise BikeSharingException(e, sys)

	def adaboost(self, X, y):
		try:
			model = AdaBoostRegressor()
			model.fit(X, y)
			return model
		except Exception as e:
			raise BikeSharingException(e, sys)

	def xgboost(self, X, y):
		try:
			model = XGBRegressor()
			model.fit(X, y)
			return model
		except Exception as e:
			raise BikeSharingException(e, sys)

	def random_forest(self, X, y):
		try:
			model = RandomForestRegressor()
			model.fit(X, y)
			return model
		except Exception as e:
			raise BikeSharingException(e, sys)

	def decision_tree(self, X, y):
		try:
			model = DecisionTreeRegressor()
			model.fit(X, y)
			return model
		except Exception as e:
			raise BikeSharingException(e, sys)


	def initiate_model_trainer(self):
		try:

			regression_models = {'xgboost': self.xgboost,
								 'adaboost': self.adaboost,
								 'random_forest': self.random_forest,
								 'decision_tree': self.decision_tree}

			logging.info(f'creating accuracy table to store the accuracy of machine learning models')
			accuracy_table = pd.DataFrame(columns=['Day', 'Hour'], 	
										  index=regression_models.keys())

			# MODEL TRAINING FOR DAY DATASET
			day_train = pd.read_csv(f'{self.day_path}/day_train.csv')
			day_test = pd.read_csv(f'{self.day_path}/day_test.csv')

			x_train, y_train = day_train.iloc[:,:-1], day_train.iloc[:, -1]
			x_test, y_test = day_test.iloc[:, :-1], day_test.iloc[:, -1] 

			for i, model in enumerate(regression_models.values()):

				logging.info(f'training model using {accuracy_table.index[i]}')
				day_model = model(x_train, y_train)

				predictions = day_model.predict(x_test)
				score = r2_score(predictions, y_test)
				score = round(score*100, 2)
				logging.info(f'accuracy using {accuracy_table.index[i]}: {score}')

				accuracy_table['Day'][i] = score

				day_model_dir = f'saved models/day_models'
				os.makedirs(day_model_dir, exist_ok=True)

				pickle.dump(day_model, open(f'{day_model_dir}/{accuracy_table.index[i]}.pkl', 'wb'))

			# MODEL TRAINING FOR HOURS DATASET
			hour_train = pd.read_csv(f'{self.hour_path}/hour_train.csv')
			hour_test = pd.read_csv(f'{self.hour_path}/hour_test.csv')

			x_train, y_train = hour_train.iloc[:,:-1], hour_train.iloc[:, -1]
			x_test, y_test = hour_test.iloc[:, :-1], hour_test.iloc[:, -1] 

			for i, model in enumerate(regression_models.values()):

				logging.info(f'training model using {accuracy_table.index[i]}')
				hour_model = model(x_train, y_train)

				predictions = hour_model.predict(x_test)
				score = r2_score(predictions, y_test)
				score = round(score*100, 2)
				logging.info(f'accuracy using {accuracy_table.index[i]}: {score}')

				accuracy_table['Hour'][i] = score

				hour_model_dir = f'saved models/hour_models'
				os.makedirs(hour_model_dir, exist_ok=True)

				pickle.dump(hour_model, open(f'{hour_model_dir}/{accuracy_table.index[i]}.pkl', 'wb'))

			accuracy_table.to_csv('model_accuracy.csv')

		except Exception as e:
			raise BikeSharingException(e, sys)