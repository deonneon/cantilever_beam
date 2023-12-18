#pip install scikit-fem[all]

import numpy as np
import pandas as pd
from skfem import *
from skfem.helpers import sym_grad
from skfem.models.elasticity import linear_elasticity, lame_parameters

# Beam dimensions from JSON data
length = 100  # mm
width = 10    # mm
height = 15   # mm

# Create mesh with approximately 1140 elements
n_x, n_y, n_z = 5, 10, 10  # Adjust these values to get close to 1140 elements
m = MeshHex.init_tensor(
    np.linspace(0, length, n_x),
    np.linspace(0, width, n_y),
    np.linspace(0, height, n_z),
)

# Use trilinear hexahedral elements
e = ElementVectorH1(ElementHex1())

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Load', 'Max Strain', 'Min Strain', 'Max Von Mises Stress', 'Min Von Mises Stress'])

# Generate 40 random load values between 2 and 5.836 N/mm
np.random.seed(0)  # Set a seed for reproducibility
load_values = np.random.uniform(2, 5.836, 50)
result_dfs = []

for load in load_values:
    # Build stiffness matrix
    basis = InteriorBasis(m, e)
    K = linear_elasticity(*lame_parameters(200000, 0.3)).assemble(basis)

    # Build load vector for distributed loading
    facet_basis = FacetBasis(m, e, facets=m.facets_satisfying(lambda x: x[0] == length))

    @LinearForm
    def linf(v, w):
        return -load * v.value[1]  # Negative for downward force

    f = linf.assemble(facet_basis)

    # Solve displacements, zero displacement at fixed end
    x = solve(*condense(K, f, D=basis.get_dofs(lambda x: x[0] == 0.0)))

    # Calculate strain tensor and mean value
    @Functional
    def strain(w):
        return sym_grad(w['disp'])[0, 0]

    exx = strain.elemental(basis, disp=basis.interpolate(x))

    # Separate displacement into components
    displacement_x = x[basis.nodal_dofs[0]]
    displacement_y = x[basis.nodal_dofs[1]]
    displacement_z = x[basis.nodal_dofs[2]]

    @Functional
    def full_strain(w):
        strain = sym_grad(w['disp'])
        return strain[0, 0], strain[1, 1], strain[2, 2], strain[0, 1], strain[1, 2], strain[0, 2]

    strain_values = full_strain.elemental(basis, disp=basis.interpolate(x))

    # Process strain_values to get full tensor for each element
    # strain_values will have shape (n_elements, 6), corresponding to the six components of the strain tensor
    strain_tensors = np.array(strain_values).reshape(-1, 6)

    # Elastic constants
    E, nu = 200000, 0.3  # Young's modulus and Poisson's ratio
    lmbda, mu = lame_parameters(E, nu)

    # Calculate stress tensor
    strain_tensors_reshaped = np.zeros((strain_tensors.shape[0], 3, 3))
    strain_tensors_reshaped[:, 0, 0] = strain_tensors[:, 0]
    strain_tensors_reshaped[:, 1, 1] = strain_tensors[:, 1]
    strain_tensors_reshaped[:, 2, 2] = strain_tensors[:, 2]
    strain_tensors_reshaped[:, 0, 1] = strain_tensors_reshaped[:, 1, 0] = strain_tensors[:, 3]
    strain_tensors_reshaped[:, 1, 2] = strain_tensors_reshaped[:, 2, 1] = strain_tensors[:, 4]
    strain_tensors_reshaped[:, 0, 2] = strain_tensors_reshaped[:, 2, 0] = strain_tensors[:, 5]

    # Calculate stress tensor for each element
    stress_tensors = np.array([lmbda * np.trace(strain_tensor) * np.eye(3) + 2 * mu * strain_tensor
                              for strain_tensor in strain_tensors_reshaped])

    von_mises_stress = np.sqrt(0.5 * ((stress_tensors[:, 0, 0] - stress_tensors[:, 1, 1])**2 +
                                      (stress_tensors[:, 1, 1] - stress_tensors[:, 2, 2])**2 +
                                      (stress_tensors[:, 0, 0] - stress_tensors[:, 2, 2])**2 +
                                      6 * (stress_tensors[:, 0, 1]**2 +
                                          stress_tensors[:, 1, 2]**2 +
                                          stress_tensors[:, 0, 2]**2)))

    # Max and min values
    max_strain = np.max(strain_tensors)
    min_strain = np.min(strain_tensors)
    max_von_mises_stress = np.max(von_mises_stress)
    min_von_mises_stress = np.min(von_mises_stress)

    # Append results to the DataFrame
    # Append results to a temporary DataFrame
    temp_df = pd.DataFrame({'Load': [load],
                            'Max Strain': [max_strain],
                            'Min Strain': [min_strain],
                            'Max Von Mises Stress': [max_von_mises_stress],
                            'Min Von Mises Stress': [min_von_mises_stress],
                            'Length': [length],
                            'Width': [width],
                            'Height': [height],
                            'Modulus': [200000],
                            'Poison': [0.3],
                            'Displacement_x': [np.min(displacement_x)],
                            'Displacement_y': [np.min(displacement_y)]})

    result_dfs.append(temp_df)

# Concatenate all the temporary DataFrames into a single DataFrame
results_df = pd.concat(result_dfs, ignore_index=True)

# Display the DataFrame with results
#print(results_df)

# Sort the DataFrame by the "Load" column in ascending order
results_df.sort_values(by='Load', inplace=True)

# Take the first 30 rows
top_40_rows = results_df.head(40)

# Display the sorted and filtered DataFrame
print(top_40_rows)