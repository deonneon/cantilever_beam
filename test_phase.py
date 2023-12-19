# Extract the last 10 rows for testing
test_data = results_df.tail(10)

# Separate features and targets
X_test_data = test_data.iloc[:, :-2].values  # Assuming the last two columns are targets
y_test_data_x = test_data.iloc[:, -2].values
y_test_data_y = test_data.iloc[:, -1].values
y_test_data = np.vstack((y_test_data_x, y_test_data_y)).T

# Scale the features using the same scaler as before
X_test_data_scaled = scaler.transform(X_test_data)  # Use transform, not fit_transform

# Make predictions
predictions = model.predict(X_test_data_scaled)

# Optional: Compare predictions with actual values
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i]}, Actual: {y_test_data[i]}")