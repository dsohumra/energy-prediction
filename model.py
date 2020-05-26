import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd  
import pickle
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('HomeC2.csv') 

#output=pd.cut(dataset.use, bins=8, labels=False)

X = dataset.iloc[:, [2,3,4]].values  
y = dataset.iloc[:, -1].values
#y = output.values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

  
regressor = RandomForestRegressor (n_estimators = 100, random_state = 60)
regressor.fit(X, y)


#ar = [[0.449041,  0.858824,  0.447556]]
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.449041,  0.858824,  0.447556]]))