# tests/test_advanced_balance_controller.py
"""
Test script for the advanced momentum-based balance controller.
"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centroidal_dynamics.balance_control import BalanceController

# Load the robot model
robot_model = pin.buildModelFromUrdf('/home/linus/robotics/g1_23dof.urdf')
robot_data = robot_model.createData()

# print("Available frames in the robot model:")
# for i, frame in enumerate(robot_model.frames):
#     print(f"Frame {i}: {frame.name}")


contact_frame_ids = [16, 30]

# Initialize the balance controller
controller = BalanceController(robot_model, contact_frame_ids=contact_frame_ids)

# Set initial configuration
q = pin.neutral(robot_model)
v = np.zeros(robot_model.nv)

# Update kinematics to get initial values
pin.forwardKinematics(robot_model, robot_data, q, v)
pin.updateFramePlacements(robot_model, robot_data)

# Set desired values
com_des = pin.centerOfMass(robot_model, robot_data, q)
vcom_des = np.zeros(3)
ang_mom_des = np.zeros(3)

# Set gains - tuned for stability
mass = sum(robot_model.inertias[i].mass for i in range(robot_model.njoints))
# Position gain (P term for position)
gain_pos = np.diag([0.5, 0.25, 0.5]) / mass 
gain_vel = np.diag([2.0, 1.0, 2.0]) / mass
gain_ang_mom = np.diag([1.0, 1.0, 1.0])
# Higher damping
gain_ang_mom_damping = np.diag([4.0, 4.0, 4.0])
gain_lin_mom_damping = np.diag([10.0, 5.0, 10.0]) / mass

# Simulation parameters
dt = 0.01
controller.dt = dt  # Update the controller's dt
simulation_time = 30.0  # 5 seconds
num_steps = int(simulation_time / dt)

# Define external push
push_time = 10.0  # Apply push after 1 second
push_duration = 0.1  # Push lasts for 0.1 seconds
push_force = np.array([115.0, 0, 0])  # 50N in x-direction

# Arrays to store results
time_array = np.zeros(num_steps)
com_array = np.zeros((num_steps, 3))
angular_momentum_array = np.zeros((num_steps, 3))
linear_momentum_array = np.zeros((num_steps, 3))

# Simulation loop
for i in range(num_steps):
    time = i * dt
    time_array[i] = time
    
    # Record current state
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)
    com = pin.centerOfMass(robot_model, robot_data, q)
    com_array[i] = com
    
    # Get current momentum
    kG, lG = controller.compute_centroidal_momentum(q, v)
    
    angular_momentum_array[i] = kG
    linear_momentum_array[i] = lG
    
    # Check if we're applying the push
    external_force = np.zeros(6)
    if push_time <= time < push_time + push_duration:
        # Apply push as an external force
        external_force[3:] = push_force  # Linear force
        external_force[:3] = np.cross(com, push_force)  # Induced moment
    
    # Compute desired momentum rate with damping
    hG_dot_des = controller.compute_desired_momentum_rate(
        q, v, com_des, vcom_des, ang_mom_des,
        gain_vel, gain_pos, gain_ang_mom,
        gain_ang_mom_damping, gain_lin_mom_damping
    )
    
    # Add external force effect
    hG_dot_des += external_force
    
    # Compute admissible momentum rate
    hG_dot_adm = controller.compute_admissible_momentum_rate(q, v, hG_dot_des)
    
    # Compute joint accelerations
    ddq = controller.compute_joint_accelerations(q, v, hG_dot_adm)
    
    # Update state
    q, v = controller.update_state(q, v, ddq, dt)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(time_array, com_array[:, 0])
plt.title('CoM Position (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(time_array, linear_momentum_array[:, 0])
plt.title('Linear Momentum (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m/s)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(time_array, angular_momentum_array[:, 1])
plt.title('Angular Momentum (Y-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m²/s)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(time_array, com_array[:, 1])
plt.title('CoM Position (Y-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

plt.tight_layout()
plt.show()