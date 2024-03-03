#%%
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
model = RandomForestClassifier()

X, y = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=0,
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits as needed
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='accuracy')
grid_search.fit(X, y)

best_params = grid_search.best_params_
print("Best Param:", best_params)

scores = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.imshow(scores, interpolation='nearest', cmap='viridis')
plt.xlabel('min_samples_split')
plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.title('Grid Search Mean Test Scores')
plt.show()
# %%
