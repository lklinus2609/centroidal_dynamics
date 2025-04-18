# In examples/mujoco_balance_example.py
import numpy as np
import matplotlib.pyplot as plt
from centroidal_dynamics.mujoco_sim import MujocoSimulator
from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum
from centroidal_dynamics.balance_control import compute_admissible_momentum_rate, compute_joint_accelerations

# Create simulator
simulator = MujocoSimulator('path/to/mujoco/xml/file')
simulator.reset()

# Get initial state
q, v = simulator.get_state()

# Define desired COM position and velocity
cG_des = simulator.pin_data.com[0].copy()
vG_des = np.zeros(3)

# Define desired angular momentum
kG_des = np.zeros(3)

# Define gain matrices
M = sum([i.mass for i in simulator.pin_model.inertias])
Gamma_11 = np.diag([40, 20, 40]) / M
Gamma_12 = np.diag([8, 3, 8]) / M
Gamma_21 = np.diag([20, 20, 20])

# Simulation parameters
dt = 0.01
simulation_time = 5.0
num_steps = int(simulation_time / dt)

# Arrays to store results
time_array = np.zeros(num_steps)
com_pos_array = np.zeros((num_steps, 3))
ang_momentum_array = np.zeros((num_steps, 3))
lin_momentum_array = np.zeros((num_steps, 3))

# Apply external force at a specific time
force_start_time = 1.0
force_end_time = 1.1
external_force = np.array([115, 0, 0])  # 115N push in the x-direction

# Simulation loop
for i in range(num_steps):
    time = i * dt
    time_array[i] = time
    
    # Get current state
    q, v = simulator.get_state()
    
    # Compute centroidal dynamics
    pin.forwardKinematics(simulator.pin_model, simulator.pin_data, q, v)
    pin.updateFramePlacements(simulator.pin_model, simulator.pin_data)
    
    hG = pin.computeCentroidalMomentum(simulator.pin_model, simulator.pin_data)
    com = pin.centerOfMass(simulator.pin_model, simulator.pin_data, q)
    
    # Record data
    com_pos_array[i] = com
    ang_momentum_array[i] = hG.angular
    lin_momentum_array[i] = hG.linear
    
    # Apply external force if in the force application window
    if force_start_time <= time < force_end_time:
        # Apply the force in MuJoCo
        # This depends on how you want to apply the force (e.g., to a specific body)
        body_id = simulator.model.body('torso').id  # Replace with your body name
        simulator.data.xfrc_applied[body_id, :3] = external_force
        
        # Convert external force to momentum rate change for controller
        external_momentum_rate = np.concatenate([np.cross(com, external_force), external_force])
    else:
        simulator.data.xfrc_applied[:] = 0
        external_momentum_rate = np.zeros(6)
    
    # Compute admissible momentum rate
    hG_dot_a = compute_admissible_momentum_rate(
        simulator.pin_model, simulator.pin_data, q, v, 
        cG_des, vG_des, kG_des,
        Gamma_11, Gamma_12, Gamma_21
    )
    
    # Add external force effect
    hG_dot_a += external_momentum_rate
    
    # Compute joint accelerations
    q_ddot = compute_joint_accelerations(simulator.pin_model, simulator.pin_data, q, v, hG_dot_a)
    
    # Convert to joint torques using inverse dynamics
    pin.computeGeneralizedGravity(simulator.pin_model, simulator.pin_data, q)
    tau = pin.rnea(simulator.pin_model, simulator.pin_data, q, v, q_ddot)
    
    # Step the simulation with computed torques
    observation = simulator.step(tau, dt)
    
    # Optionally render
    simulator.render()

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