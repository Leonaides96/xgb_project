import pickle 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#### Example below have completed the reading, normalizing, split-train and modeling
# Step 1: Data Preparation
# Load your time series data into a pandas DataFrame and perform necessary data cleaning.

# Step 2: Feature Engineering
# Create lag features and additional relevant features.

# Step 3: Train-Test Split
# Split the data into training and testing sets.

# Step 4: XGBoost Model Setup
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Step 5: Prediction

# Step 6 : Serialized the model, for reusablilities 
## Kindly remember pickle is to object to byte, or byte to object (deserielized), using the context manager the mode or wb or rb, any with binary mode
with open("model_registry.dat", "wb"): ## notice that the extension is ".dat" where to storing the object -> bytes -> to disk format. # 
pickle.dump(model, )