import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from rental_bike_share.logger import logging
from rental_bike_share.exception import BikeSharingException


class DataCleaning:
	def __init__(self, raw_data_path):
		try:
			self.day_path = f'{raw_data_path}/day.csv'
			self.hour_path = f'{raw_data_path}/hour.csv'

		except Exception as e:
			raise BikeSharingException(e, sys)


	def initiate_data_cleaning(self):
		try:
			unwanted_columns = ['_id', 'instant', 'dteday', 'atemp', 'casual', 'registered']

			logging.info(f'drop unwanted columns from day dataset')
			df_day = pd.read_csv(self.day_path)
			df_day = df_day.drop(unwanted_columns, axis=1)

			logging.info(f'drop unwanted columns from day dataset')
			df_hour = pd.read_csv(self.hour_path)
			df_hour = df_hour.drop(unwanted_columns, axis=1)

			day_train, day_test = train_test_split(df_day, test_size=0.15, random_state=1)

			hour_train, hour_test = train_test_split(df_hour, test_size=0.30, random_state=1)

			clean_day_data_dir = os.path.join(os.getcwd(), 'Data/cleaned_data/day')
			os.makedirs(clean_day_data_dir, exist_ok=True)

			day_train.to_csv(path_or_buf=f'{clean_day_data_dir}/day_train.csv', index=False, header=True)
			day_test.to_csv(path_or_buf=f'{clean_day_data_dir}/day_test.csv', index=False, header=True)

			clean_hour_data_dir = os.path.join(os.getcwd(), 'Data/cleaned_data/hour')
			os.makedirs(clean_hour_data_dir, exist_ok=True)

			hour_train.to_csv(path_or_buf=f'{clean_hour_data_dir}/hour_train.csv', index=False, header=True)
			hour_test.to_csv(path_or_buf=f'{clean_hour_data_dir}/hour_test.csv', index=False, header=True)

			return clean_day_data_dir, clean_hour_data_dir

		except Exception as e:
			raise BikeSharingException(e, sys)