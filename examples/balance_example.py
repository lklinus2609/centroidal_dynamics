import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum, compute_centroidal_inertia
from centroidal_dynamics.avg_velocity import compute_average_spatial_velocity
from centroidal_dynamics.balance_control import compute_admissible_momentum_rate, compute_joint_accelerations

# Load the robot model
robot_model = pin.buildModelFromUrdf('path/to/urdf/file')
robot_data = robot_model.createData()

# Set a configuration
q = pin.neutral(robot_model)
v = np.zeros(robot_model.nv)

# Define desired COM position and velocity
cG_des = pin.centerOfMass(robot_model, robot_data, q)
vG_des = np.zeros(3)

# Define desired angular momentum
kG_des = np.zeros(3)

# Define gain matrices
Gamma_11 = np.diag([40, 20, 40]) / robot_model.mass[0]  # Assuming mass is stored in the first link
Gamma_12 = np.diag([8, 3, 8]) / robot_model.mass[0]
Gamma_21 = np.diag([20, 20, 20])

# Simulate a lateral push (external force)
external_force = np.array([115, 0, 0])  # 115N push in the x-direction
external_force_duration = 0.1  # 0.1 seconds

# Run simulation
dt = 0.01
simulation_time = 5.0  # 5 seconds
num_steps = int(simulation_time / dt)

# Arrays to store results
time_array = np.zeros(num_steps)
com_pos_array = np.zeros((num_steps, 3))
ang_momentum_array = np.zeros((num_steps, 3))
lin_momentum_array = np.zeros((num_steps, 3))

# Apply external force at a specific time
force_start_time = 1.0  # Apply force after 1 second
force_end_time = force_start_time + external_force_duration

# Simulation loop
for i in range(num_steps):
    time = i * dt
    time_array[i] = time
    
    # Record data
    com_pos_array[i] = pin.centerOfMass(robot_model, robot_data, q)
    hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
    ang_momentum_array[i] = hG.angular
    lin_momentum_array[i] = hG.linear
    
    # Apply external force if in the force application window
    if force_start_time <= time < force_end_time:
        # Convert external force to momentum rate change
        external_momentum_rate = np.concatenate([np.cross(com_pos_array[i], external_force), external_force])
    else:
        external_momentum_rate = np.zeros(6)
    
    # Compute admissible momentum rate
    hG_dot_a = compute_admissible_momentum_rate(
        robot_model, robot_data, q, v, 
        cG_des, vG_des, kG_des,
        Gamma_11, Gamma_12, Gamma_21
    )
    
    # Add external force effect
    hG_dot_a += external_momentum_rate
    
    # Compute joint accelerations
    q_ddot = compute_joint_accelerations(robot_model, robot_data, q, v, hG_dot_a)
    
    # Integrate to get new state
    v = v + q_ddot * dt
    q = pin.integrate(robot_model, q, v * dt)
    
    # Update the robot state
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time_array, com_pos_array[:, 0])
plt.title('CoM Position (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.subplot(2, 2, 2)
plt.plot(time_array, lin_momentum_array[:, 0])
plt.title('Linear Momentum (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m/s)')

plt.subplot(2, 2, 3)
plt.plot(time_array, ang_momentum_array[:, 1])  # Y-axis for rotation around coronal plane
plt.title('Angular Momentum (Coronal Plane)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m²/s)')

plt.tight_layout()
plt.show()