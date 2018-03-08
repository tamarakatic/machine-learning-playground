import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regression_OLS.summary())
