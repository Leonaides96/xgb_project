#%%
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd


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
X = pd.DataFrame(X)
y= pd.Series(y)

#%%
cv = TimeSeriesSplit(n_splits=5, max_train_size=30)
min_features_to_select = 1  # Minimum number of features to consider
clf = LogisticRegression()

for train_index, test_index in cv.split(X,y):
    print(train_index)

    X_window = X.iloc[train_index]
    y_window = y.iloc[train_index]

    processor = make_pipeline(RFECV(estimator=clf, step=1, cv=cv, scoring="neg_mean_squared_error", min_features_to_select=min_features_to_select, n_jobs=2, ))
    processor.fit(X_window, y_window)


#%%


rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="neg_mean_squared_error",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")

#%%
n_scores = len(rfecv.cv_results_["mean_test_score"])
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    range(min_features_to_select, n_scores + min_features_to_select),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()
# %%
