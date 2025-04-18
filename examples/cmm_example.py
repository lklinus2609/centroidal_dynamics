import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum

# Load the robot model
robot_model = pin.buildModelFromUrdf('path/to/urdf/file')
robot_data = robot_model.createData()

# Set a configuration
q = pin.neutral(robot_model)
v = np.random.rand(robot_model.nv)  # Random joint velocities

# Compute the CMM
AG = compute_cmm(robot_model, robot_data, q)
print("Centroidal Momentum Matrix (CMM):")
print(AG)

# Compute centroidal momentum
hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
print("\nCentroidal Momentum:")
print(f"Angular momentum: {hG.angular}")
print(f"Linear momentum: {hG.linear}")

# Verify: hG = AG * v
hG_from_cmm = AG @ v
print("\nCentroidal Momentum computed from CMM:")
print(hG_from_cmm)