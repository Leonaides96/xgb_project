import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# reading the data
data = pd.read_csv("titanic.csv")

# data cleaning, dropna or filling

# target and input 
y = ... 
x = ...

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

###############################################################################################################################################################################################################################################################
## This is where part from the ChatGPT
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Load your time series data into a pandas DataFrame and perform necessary data cleaning.

# Step 2: Feature Engineering
# Create lag features and additional relevant features.

# Step 3: Train-Test Split
# Split the data into training and testing sets.

# Step 4: XGBoost Model Setup
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)

# Step 5: Parameter Tuning (Optional)
# Tune hyperparameters using grid search or randomized search.

# Step 6: Model Training
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 8: Hyperparameter Tuning (Optional)

# Step 9: Make Predictions (Optional)
# Predict on future time points or unseen data.

# Step 10: Visualization
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.show()

########################################################