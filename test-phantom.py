import numpy as np
import matplotlib.pyplot as plt
from vtomo import *
from phantom import *

#####################################################
# Create a test 3D vector field 
#####################################################

scale = 1
shape = (scale*64, scale*64, scale*64)
centers = np.array([
    (scale*32, scale*32, scale*32),  # Center of first circle
    (scale*24, scale*36, scale*32),  # Center of second circle
    (scale*42, scale*24, scale*32),  # Center of third circle
])
radii = np.array([
    scale*24,  # Radius of first circle
    scale*10,  # Radius of second circle
    scale*6,   # Radius of third circle
]) 
domains = np.array([
    (0, np.pi/2),        # X direction
    (np.pi/2, np.pi/2),  # Y direction  
    (0, 0)               # Z direction
])

# Create the phantom
field, mask = create_vector_field_phantom_3d(
    shape, centers, radii, domain_angles=domains, transition_width=scale*1.0)

print (field.shape)

plt.quiver(field[:, :, 32, 0], field[:, :, 32, 1])
plt.show()  

# plt.figure(figsize=(12, 3))
# plt.subplot(1, 3, 1)
# plt.imshow(field[32, :, :, 0], cmap='gray')
# plt.colorbar(label='Scalar Value')
# plt.title('Vector Field Phantom (x)')

# plt.subplot(1, 3, 2)
# plt.imshow(field[32, :, :, 1], cmap='gray')
# plt.colorbar(label='Scalar Value')
# plt.title('Vector Field Phantom (y)')

# plt.subplot(1, 3, 3)
# plt.imshow(field[32, :, :, 2], cmap='gray')
# plt.colorbar(label='Scalar Value')
# plt.title('Vector Field Phantom (z)')

# plt.show()