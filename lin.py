from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Synthetic Data Generation
np.random.seed(0)  # For reproducibility

# Example features: Material (0 or 1), Length (10-20), Width (1-5)
n_samples = 100
materials = np.random.randint(0, 2, n_samples)
lengths = np.random.uniform(10, 20, n_samples)
widths = np.random.uniform(1, 5, n_samples)

# Generating synthetic force-displacement curves (simplified for demonstration)
# Force = k * displacement, where k depends on material, length, and width
k_values = materials * 5 + lengths / widths
displacements = np.linspace(1, 5, 5)  # Displacements from 1 to 5
forces = np.array([k * displacements for k in k_values])  # Force values

# Calculating derivatives (slopes)
derivatives = np.gradient(forces, axis=1)  # Derivatives along each row

# Preparing the dataset
features = pd.DataFrame({'Material': materials, 'Length': lengths, 'Width': widths})
targets = pd.DataFrame(derivatives, columns=[f'Derivative_{d}mm' for d in displacements])

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# Model Definition and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting for a new configuration
new_config = pd.DataFrame({'Material': [1], 'Length': [15], 'Width': [2]})
predicted_derivatives = model.predict(new_config)

# Outputting the predicted derivatives
predicted_derivatives



