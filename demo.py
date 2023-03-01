import numpy as np
import pickle

data = np.array([1,1,1,0,0,0,0,2,0.46,0.88,0.2985])
data=data.reshape(1,-1)

hour_model = pickle.load(open(f'saved models/hour_models/xgboost.pkl', 'rb'))

print(hour_model.predict(data))