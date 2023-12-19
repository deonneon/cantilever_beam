import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


# Load FEA results
fea_results = pd.read_csv('data.csv')

# Extract features and targets from FEA results
X_fea = fea_results.iloc[:, :-2].values  # All columns except the last two
y_fea_x = fea_results.iloc[:, -2].values  # Second last column is the target variable for x-axis deflection
y_fea_y = fea_results.iloc[:, -1].values  # Last column is the target variable for y-axis deflection
y_fea = np.vstack((y_fea_x, y_fea_y)).T  # Combine the target variables into a single array

# Number of features is the number of columns in X_fea
num_features_fea = X_fea.shape[1]

# Scale the FEA features
scaler = StandardScaler()
X_fea_scaled = scaler.fit_transform(X_fea)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_fea_scaled, y_fea, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the ANN model
tf.random.set_seed(0)
model = Sequential()
model.add(Dense(64, input_dim=num_features_fea, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear')) # Two output neurons for x and y deflections

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=20, batch_size=5)

# Optionally: Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predictions for evaluation
y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test R-squared: {r_squared}")

# Save model
#model.save('beam_deflection_model.h5')

#print("ANN model trained and evaluated.")
