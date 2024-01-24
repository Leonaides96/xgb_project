import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# reading the data
data = pd.read_csv("titanic.csv")

# data cleaning, dropna or filling

# target and input 
x = ...
y = ... 

# train and split ()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=)

# instance the model
model = XGBClassifier()
## feeding the model 
model.fit(x_train, y_train)

## prediction
y_pred = model.prediction(x_test)
prediction = [round(v) for v in y_pred] # if the rounding is necessary

# accurancy checking
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, prediction)
