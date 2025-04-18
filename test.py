import mujoco
import numpy as np
import pinocchio as pin

# Example for loading a robot model in Pinocchio
robot_model = pin.buildModelFromUrdf('/home/linus/robotics/g1_23dof.urdf')
robot_data = robot_model.createData()

# Set a configuration
q = pin.neutral(robot_model)
v = np.random.rand(robot_model.nv)

# Update kinematics
pin.forwardKinematics(robot_model, robot_data, q, v)
pin.updateFramePlacements(robot_model, robot_data)

# Compute centroidal momentum and CMM
hG = pin.computeCentroidalMomentum(robot_model, robot_data)
AG = pin.computeCentroidalMap(robot_model, robot_data, q)

# Verify: hG = AG * v
hG_from_cmm = AG @ v
print("Centroidal momentum from direct computation:", hG.vector)
print("Centroidal momentum from CMM * v:", hG_from_cmm)

# Compute centroidal inertia
IG = pin.crba(robot_model, robot_data, q)  # This is the joint space inertia matrix
# To get the centroidal inertia, we need additional steps (implementation to follow)

