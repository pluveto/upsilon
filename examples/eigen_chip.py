import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 2x3x4 tensor
tensor = np.arange(24).reshape((2, 3, 4), order='F')

# setvalues
tensor = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get coordinates
x, y, z = np.indices(tensor.shape)

# Flatten the coordinates and tensor values
x = x.flatten()
y = y.flatten()
z = z.flatten()
tensor_values = tensor.flatten()

# Define perspective coefficient
perspective_coefficient = 0.05  # Adjust this value to control perspective effect

# Calculate distance from each point to observer
distance = np.sqrt(x**2 + y**2 + z**2)

# Calculate scaling factor for sphere size based on perspective
sphere_scaling = 3 / (1 + perspective_coefficient * distance)

# Plot points
ax.scatter(x, y, z, c=tensor_values, cmap='viridis', s=100 * sphere_scaling)

# Annotate a few points
for i in range(len(tensor_values)):
    ax.text(x[i], y[i], z[i], f'{tensor_values[i]}', color='black', fontsize=10, ha='center', va='center')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
