import pymongo
from rental_bike_share.components.data_ingestion import DataIngestion
from rental_bike_share.components.data_cleaning import DataCleaning
from rental_bike_share.components.model_training import ModelTrainer

database_name = "Rentalbike"
collection_name1 = "Rentalbikesharing_days"
collection_name2 = "Rentalbikesharing_hours"
mongo_client = pymongo.MongoClient("mongodb+srv://juliusmaharajan11:Halamadrid@cluster0.sfnu7mg.mongodb.net/?retryWrites=true&w=majority")


if __name__ == "__main__":

	data_ingestion = DataIngestion(mongo_client=mongo_client, database_name=database_name, collection_name1=collection_name1, collection_name2=collection_name2)
	raw_data_path = data_ingestion.initiate_data_ingestion()

	data_cleaning = DataCleaning(raw_data_path=raw_data_path)
	day_data_path, hour_data_path = data_cleaning.initiate_data_cleaning()
	
	model_training = ModelTrainer(day_path=day_data_path, hour_path=hour_data_path)
	model_training.initiate_model_trainer()

