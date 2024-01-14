from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 20)

# Reshaping for model
X = X[:, np.newaxis]
y = y[:, np.newaxis]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Transforming data for polynomial regression
polynomial_features = PolynomialFeatures(degree=10)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.transform(X_test)

# Training the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Making predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculating MSE and R-squared for both sets
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Plotting for visualization
plt.scatter(X_train, y_train, color='red', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
# Sorting values for a smooth curve
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_train,y_train_pred), key=sort_axis)
X_train, y_train_pred = zip(*sorted_zip)
plt.plot(X_train, y_train_pred, color='blue', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (Degree 10)')
plt.legend()
plt.show()

mse_train, r2_train, mse_test, r2_test
