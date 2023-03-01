import os
import sys
import pandas as pd
from rental_bike_share.logger import logging
from rental_bike_share.exception import BikeSharingException


class DataIngestion:
	def __init__(self, mongo_client, database_name, collection_name1, collection_name2):
		try:
			self.mongo_client = mongo_client
			self.database_name = database_name
			self.collection_name1 = collection_name1
			self.collection_name2 = collection_name2
		except Exception as e:
			raise BikeSharingException(e, sys)

	def initiate_data_ingestion(self):
		try:
			logging.info(f'exporting day data as dataframe from mongodb database')
			df_day = pd.DataFrame(list(self.mongo_client[self.database_name][self.collection_name1].find()))
			
			logging.info(f'exporting hour data as dataframe from mongodb database')
			df_hour = pd.DataFrame(list(self.mongo_client[self.database_name][self.collection_name2].find()))


			logging.info(f'creating a dataset directory')
			raw_data_dir = os.path.join(os.getcwd(), 'Data/raw_data')
			os.makedirs(raw_data_dir, exist_ok=True)

			logging.info(f'saving day dataset in raw_dataset directory')
			df_day.to_csv(path_or_buf=f'{raw_data_dir}/day.csv', index=False, header=True)

			logging.info(f'saving hour dataset in raw_dataset directory')
			df_hour.to_csv(path_or_buf=f'{raw_data_dir}/hour.csv', index=False, header=True)

			return raw_data_dir

		except Exception as e:
			raise BikeSharingException(e, sys)