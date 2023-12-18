import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(0)

# Beam Properties (from JSON)
length = 100  # mm
width = 10    # mm
height = 15   # mm
E = 200000  # MPa (200 GPa converted to MPa)
I = (1/12) * width * height**3  # mm^4

# Load Range (from JSON)
min_load = 20  # N/mm
max_load = 58.36  # N/mm

# Number of Load Cases and Features
num_load_cases = 40
num_features = 360

# Generate loads and corresponding deflections
np.random.seed(0)  # For reproducibility
loads = np.linspace(min_load, max_load, num_load_cases)

# Simulated FEM-like deflection
deflections = (loads * length**4) / (8 * E * I)
deflections += deflections * np.random.normal(0, 0.05, size=deflections.shape)  # Adding noise
deflections = deflections * (1 + 0.1 * np.sin(loads))

# Generate fake features
features = np.random.rand(num_load_cases, num_features)

# Creating DataFrame
data = pd.DataFrame(features, columns=[f'Feature_{i+1}' for i in range(num_features)])
data['Load'] = loads
data['Deflection'] = deflections

# Preparing Data for ANN
scaler = StandardScaler()
X = data.iloc[:, :-2].values  # Features
y = data['Deflection'].values  # Target

# Scale the features
X_scaled = scaler.fit_transform(X)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=1/3, random_state=42)

# Build ANN Model
model = Sequential()
model.add(Dense(64, input_dim=num_features, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Single output neuron for regression

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
