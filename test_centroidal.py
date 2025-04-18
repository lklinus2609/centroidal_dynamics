# test_centroidal.py
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum, compute_centroidal_inertia
from centroidal_dynamics.avg_velocity import compute_average_spatial_velocity, compute_kinetic_energy

# Load the robot model
robot_model = pin.buildModelFromUrdf('/home/linus/robotics/g1_23dof.urdf')
robot_data = robot_model.createData()

# Set a configuration and velocity
q = pin.neutral(robot_model)
v = np.random.rand(robot_model.nv) * 0.1  # Small random velocities

# Compute and print centroidal quantities
print("=== Centroidal Dynamics Test ===")

# Compute CMM
AG = compute_cmm(robot_model, robot_data, q)
print(f"CMM shape: {AG.shape}")
print(f"Joint DOFs: {robot_model.nv}")
print("CMM (first 3 rows):")
print(AG[:3, :])

# Compute centroidal momentum
hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
print("\nCentroidal Momentum:")
print(f"Angular: {hG.angular}")
print(f"Linear: {hG.linear}")

# Verify: hG = AG * v
hG_from_cmm = AG @ v
print("\nCentroidal Momentum Verification:")
print(f"Direct computation: {hG.vector}")
print(f"From CMM * v: {hG_from_cmm}")
print(f"Difference norm: {np.linalg.norm(hG.vector - hG_from_cmm)}")

# Compute centroidal inertia
IG = compute_centroidal_inertia(robot_model, robot_data, q)
print("\nCentroidal Inertia Matrix:")
print(IG)

# Compute average spatial velocity
vG = compute_average_spatial_velocity(robot_model, robot_data, q, v)
print("\nAverage Spatial Velocity:")
print(f"Angular: {vG[:3]}")
print(f"Linear: {vG[3:]}")

# Compute kinetic energy decomposition
T, T_avg, T_rel = compute_kinetic_energy(robot_model, robot_data, q, v)
print("\nKinetic Energy Decomposition:")
print(f"Total kinetic energy: {T}")
print(f"Kinetic energy from average motion: {T_avg}")
print(f"Kinetic energy from relative motion: {T_rel}")
print(f"Verification - sum of components: {T_avg + T_rel}")