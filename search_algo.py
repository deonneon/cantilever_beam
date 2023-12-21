from sklearn.metrics import mean_squared_error
import random

# Function to generate a random configuration
def generate_random_configuration():
    material = random.randint(0, 1)
    length = random.uniform(10, 20)
    width = random.uniform(1, 5)
    return {'Material': material, 'Length': length, 'Width': width}

# Number of iterations for the random search
n_iterations = 1000

# Store the best configuration and its corresponding derivative
best_configuration = None
lowest_derivative = float('inf')

# Random Search
for _ in range(n_iterations):
    # Generate a random configuration
    config = generate_random_configuration()
    config_df = pd.DataFrame([config])
    
    # Predict the derivative for this configuration
    derivative_prediction = model.predict(config_df).mean()  # Using the mean derivative as the criterion

    # Update the best configuration if this is the lowest derivative so far
    if derivative_prediction < lowest_derivative:
        lowest_derivative = derivative_prediction
        best_configuration = config

# Output the best configuration and its derivative
best_configuration, lowest_derivative

