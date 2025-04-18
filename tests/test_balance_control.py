# At the top of balance_control.py
"""
Momentum-based balance controller for humanoid robots.

Note on tuning: 
- The current implementation shows oscillatory behavior in response to pushes
- This is likely due to high gains, lack of damping, and missing physical constraints
- Future improvements should include:
  1. Better gain tuning
  2. Implementation of CoP constraints
  3. Adding damping terms to the controller
  4. More sophisticated admissible momentum rate calculation
"""

# In compute_desired_momentum_rate function, add:
    # NOTE: Gain tuning is critical for stable behavior
    # Too high gains may cause oscillations and instability
    # Future work: Implement adaptive gains based on disturbance magnitude

# In compute_admissible_momentum_rate function, add:
    # TODO: Implement proper CoP constraints and physical feasibility checks
    # The current implementation uses the desired momentum rate directly
    # which can lead to physically impossible commands and unstable behavior
    
# In compute_joint_accelerations function, add:
    # TODO: Improve numerical stability of the solution
    # Consider adding damping terms to reduce oscillations

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum
from centroidal_dynamics.balance_control import (
    compute_desired_momentum_rate,
    compute_admissible_momentum_rate,
    compute_joint_accelerations,
    compute_control_torques
)

# Load the robot model
robot_model = pin.buildModelFromUrdf('/home/linus/robotics/g1_23dof.urdf')
robot_data = robot_model.createData()

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

# Set gains
mass = sum(robot_model.inertias[i].mass for i in range(robot_model.njoints))
    #gain values here
gain_pos = np.diag([4, 2, 4]) / mass 
gain_vel = np.diag([20, 10, 20]) / mass
gain_ang_mom = np.diag([10, 10, 10])

# Simulation parameters
dt = 0.01
simulation_time = 3.0  # 3 seconds
num_steps = int(simulation_time / dt)

# Define external push
push_time = 1.0  # Apply push after 1 second
push_duration = 0.1  # Push lasts for 0.1 seconds
push_force = np.array([50.0, 0, 0])  # 50N in x-direction

# Arrays to store results
time_array = np.zeros(num_steps)
com_array = np.zeros((num_steps, 3))
angular_momentum_array = np.zeros((num_steps, 3))
linear_momentum_array = np.zeros((num_steps, 3))

# Simple integration simulation
for i in range(num_steps):
    time = i * dt
    time_array[i] = time
    
    # Record current state
    com_array[i] = pin.centerOfMass(robot_model, robot_data, q)
    hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
    angular_momentum_array[i] = hG.angular
    linear_momentum_array[i] = hG.linear
    
    # Check if we're applying the push
    external_force = np.zeros(6)
    if push_time <= time < push_time + push_duration:
        # Apply push as an external force
        com = pin.centerOfMass(robot_model, robot_data, q)
        external_force[3:] = push_force  # Linear force
        external_force[:3] = np.cross(com, push_force)  # Induced moment
    
    # Compute desired momentum rate
    hG_dot_des = compute_desired_momentum_rate(
        robot_model, robot_data, q, v,
        com_des, vcom_des, ang_mom_des,
        gain_vel, gain_pos, gain_ang_mom
    )
    
    # Add external force effect
    hG_dot_des += external_force
    
    # Compute admissible momentum rate
    hG_dot_adm = compute_admissible_momentum_rate(
        robot_model, robot_data, q, v, hG_dot_des
    )
    
    # Compute joint accelerations
    ddq = compute_joint_accelerations(
        robot_model, robot_data, q, v, hG_dot_adm
    )
    
    # Integrate to get new velocities and positions
    # v_new = v + ddq * dt
    v_new = np.clip(v + ddq * dt, -10, 10)
    q_new = pin.integrate(robot_model, q, v * dt)
    
    # Update state
    q = q_new
    v = v_new
    
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)

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
plt.plot(time_array, angular_momentum_array[:, 1])  # Y-axis corresponds to rotation in XZ plane
plt.title('Angular Momentum (Y-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m²/s)')
plt.grid(True)

plt.tight_layout()
plt.show()