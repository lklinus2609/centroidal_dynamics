# tests/test_balance_controller_mujoco.py
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centroidal_dynamics.cmm import compute_cmm, compute_centroidal_momentum
from centroidal_dynamics.balance_control import (
    compute_desired_momentum_rate,
    compute_admissible_momentum_rate,
    compute_joint_accelerations,
    compute_control_torques
)
from centroidal_dynamics.mujoco_sim import MujocoSimulator

# Debug utility functions
def debug_header(phase, message):
    """Print a debug header with phase information"""
    print(f"\n{'='*10} PHASE {phase}: {message} {'='*10}")

def debug_value(name, value, threshold=1e-6):
    """Print a debug value, highlighting non-zero values"""
    if isinstance(value, np.ndarray):
        max_abs = np.max(np.abs(value)) if value.size > 0 else 0
        is_zero = max_abs < threshold
        print(f"{name}: shape={value.shape}, max_abs={max_abs:.6f} {'(ZERO)' if is_zero else '(NON-ZERO)'}")
    else:
        is_zero = abs(value) < threshold
        print(f"{name}: {value} {'(ZERO)' if is_zero else '(NON-ZERO)'}")

def debug_state(q, v, time_val=None):
    """Print debug information about the current state"""
    time_str = f" at t={time_val:.2f}" if time_val is not None else ""
    print(f"\n--- State{time_str} ---")
    debug_value("q", q)
    debug_value("v", v)

# Path to MuJoCo XML model file
mujoco_model_path = '/home/linus/robotics/g1_23dof.xml'

# Path to URDF model file for Pinocchio
urdf_model_path = '/home/linus/robotics/g1_23dof.urdf'

# Initialize the simulator with both paths
simulator = MujocoSimulator(mujoco_model_path, urdf_model_path)

# Get robot model and data from Pinocchio
robot_model = simulator.pin_model
robot_data = simulator.pin_data

# Print model information for debugging
simulator.print_model_info()
print(f"Pinocchio model dimensions: nq={robot_model.nq}, nv={robot_model.nv}")

# Use Pinocchio's neutral configuration for initial state
q_pin = pin.neutral(robot_model)
v_pin = np.zeros(robot_model.nv)

# Update kinematics to get initial values
pin.forwardKinematics(robot_model, robot_data, q_pin, v_pin)
pin.updateFramePlacements(robot_model, robot_data)

# Set desired values
com_des = pin.centerOfMass(robot_model, robot_data, q_pin)
vcom_des = np.zeros(3)
ang_mom_des = np.zeros(3)

# Set gains (reduced to avoid oscillations)
mass = sum(robot_model.inertias[i].mass for i in range(robot_model.njoints))
gain_pos = np.diag([2, 1, 2]) / mass  # Reduced gains
gain_vel = np.diag([10, 5, 10]) / mass  # Reduced gains
gain_ang_mom = np.diag([5, 5, 5])  # Reduced gains

# Simulation parameters
dt = 0.01
simulation_time = 5.0  # 5 seconds
num_steps = int(simulation_time / dt)

# Define external push
push_time = 1.0  # Apply push after 1 second
push_duration = 0.1  # Push lasts for 0.1 seconds
push_force = np.array([50.0, 0, 0])  # 50N in x-direction

# Find the torso body ID for applying the push
torso_id = None
for i in range(simulator.model.nbody):
    if simulator.model.body(i).name == "torso_link":
        torso_id = i
        print(f"Found torso body ID: {torso_id}")
        break

if torso_id is None:
    print("WARNING: Torso body not found, using pelvis (body ID 1) for push application")
    torso_id = 1

# Arrays to store results
time_array = np.zeros(num_steps)
com_array = np.zeros((num_steps, 3))
momentum_array = np.zeros((num_steps, 6))
grf_array = np.zeros((num_steps, 2, 3))  # [time, left/right, xyz]
cop_array = np.zeros((num_steps, 2, 2))  # [time, left/right, xy]
ankle_torque_array = np.zeros((num_steps, 2, 3))  # [time, left/right, xyz]

# Initialize debug variables
debug_step = 10  # Print debug every 10 steps
last_hg_dot_adm = np.zeros(6)

# Simulation loop
for i in range(num_steps):
    time_array[i] = i * dt
    
    # Phase 1: Debug state and CMM - only print every debug_step iterations
    if i % debug_step == 0 or i < 5:  # Print first few steps and then periodically
        debug_header(1, f"State and CMM at step {i}, time {time_array[i]:.2f}s")
        q_pin, v_pin = simulator.get_pin_state()
        debug_state(q_pin, v_pin, time_array[i])
    
    # Get current state
    q_pin, v_pin = simulator.get_pin_state()
    
    # Update Pinocchio model
    pin.forwardKinematics(robot_model, robot_data, q_pin, v_pin)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Record data
    com_array[i] = pin.centerOfMass(robot_model, robot_data, q_pin)
    hG = compute_centroidal_momentum(robot_model, robot_data, q_pin, v_pin)
    momentum_array[i, 0:3] = hG.angular
    momentum_array[i, 3:6] = hG.linear
    
    # Get contact data from MuJoCo
    try:
        foot_forces, foot_cops = simulator.get_contact_data()
        grf_array[i] = foot_forces
        cop_array[i] = foot_cops
        
        if i % debug_step == 0:
            debug_header(1.5, f"Contact data at step {i}, time {time_array[i]:.2f}s")
            debug_value("foot_forces", foot_forces)
            debug_value("foot_cops", foot_cops)
    except Exception as e:
        print(f"Error getting contact data: {e}")
    
    # Get ankle torques
    try:
        ankle_torques = simulator.get_ankle_torques()
        ankle_torque_array[i] = ankle_torques
        
        if i % debug_step == 0:
            debug_value("ankle_torques", ankle_torques)
    except Exception as e:
        print(f"Error getting ankle torques: {e}")
    
    # Phase 2: Debug push application
    external_force = None
    ext_momentum_force = np.zeros(6)
    
    if push_time <= time_array[i] < push_time + push_duration:
        debug_header(2, f"PUSH APPLIED at t={time_array[i]:.2f}s")
        print(f"Push force: {push_force}")
        
        # Create external force for MuJoCo
        force_vector = np.zeros(6)
        force_vector[0] = push_force[0]  # fx
        force_vector[1] = push_force[1]  # fy
        force_vector[2] = push_force[2]  # fz
        external_force = {torso_id: force_vector}
        
        # Create external force for momentum calculation
        com = pin.centerOfMass(robot_model, robot_data, q_pin)
        ext_momentum_force[3:] = push_force  # Linear force
        ext_momentum_force[:3] = np.cross(com, push_force)  # Induced moment
        debug_value("External momentum force", ext_momentum_force)
    
    # Compute desired momentum rate
    hG_dot_des = compute_desired_momentum_rate(
        robot_model, robot_data, q_pin, v_pin,
        com_des, vcom_des, ang_mom_des,
        gain_vel, gain_pos, gain_ang_mom
    )
    
    # Print before/after adding external force
    if push_time <= time_array[i] < push_time + push_duration:
        print("Desired momentum rate BEFORE adding external force:")
        debug_value("hG_dot_des (before)", hG_dot_des)
        
        # Add external force effect
        hG_dot_des += ext_momentum_force
        
        print("Desired momentum rate AFTER adding external force:")
        debug_value("hG_dot_des (after)", hG_dot_des)
    else:
        # Add external force effect (will be zero outside push time)
        hG_dot_des += ext_momentum_force
    
    # Phase 3: Debug admissible momentum rate
    hG_dot_adm = compute_admissible_momentum_rate(
        robot_model, robot_data, q_pin, v_pin, hG_dot_des
    )
    
    # Only print if admissible momentum changes significantly
    if np.max(np.abs(hG_dot_adm - last_hg_dot_adm)) > 1e-6:
        debug_header(3, f"ADMISSIBLE MOMENTUM CHANGED at t={time_array[i]:.2f}s")
        debug_value("hG_dot_des", hG_dot_des)
        debug_value("hG_dot_adm", hG_dot_adm)
        last_hg_dot_adm = hG_dot_adm.copy()
    
    # Phase 4: Debug joint acceleration and torque
    ddq = compute_joint_accelerations(
        robot_model, robot_data, q_pin, v_pin, hG_dot_adm
    )
    
    # Only print if joint accelerations are non-zero
    if np.max(np.abs(ddq)) > 1e-6:
        debug_header(4, f"NON-ZERO JOINT ACCELERATION at t={time_array[i]:.2f}s")
        debug_value("ddq", ddq)
    
    # Compute control torques
    tau = compute_control_torques(
        robot_model, robot_data, q_pin, v_pin, ddq
    )
    
    # Only print if torques are non-zero
    if np.max(np.abs(tau)) > 1e-6:
        debug_header(5, f"NON-ZERO CONTROL TORQUE at t={time_array[i]:.2f}s")
        debug_value("tau", tau)
    
    # Phase 5: Debug simulation step
    try:
        simulator.apply_pin_control(tau)
        simulator.step(external_force=external_force, dt=dt)
        
        # Periodically check if anything has changed in simulation
        if i % debug_step == 0:
            mj_q, mj_v = simulator.get_state()
            debug_header(5, f"MUJOCO STATE at t={time_array[i]:.2f}s")
            debug_value("mj_q", mj_q)
            debug_value("mj_v", mj_v)
    except Exception as e:
        print(f"Error in simulation step: {e}")
    
    # Skip rendering to avoid GLFW errors
    # simulator.render()
    
    # Add a small delay to allow viewing output
    if i % debug_step == 0:
        time.sleep(0.1)

# Close the simulator
simulator.close()

# Plot results similar to Figure 7
plt.figure(figsize=(15, 12))

# Plot CoM position (similar to Fig 7a)
plt.subplot(3, 3, 1)
plt.plot(time_array, com_array[:, 0])
plt.title('CoM Position (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

# Plot linear momentum (similar to Fig 7b)
plt.subplot(3, 3, 2)
plt.plot(time_array, momentum_array[:, 3])  # Linear momentum X
plt.title('Linear Momentum (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m/s)')
plt.grid(True)

# Plot angular momentum (similar to Fig 7c)
plt.subplot(3, 3, 3)
plt.plot(time_array, momentum_array[:, 1])  # Angular momentum Y
plt.title('Angular Momentum (Y-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m²/s)')
plt.grid(True)

# Plot foot GRF vertical (similar to Fig 7d)
plt.subplot(3, 3, 4)
plt.plot(time_array, grf_array[:, 0, 1])  # Right foot, vertical force
plt.title('Right Foot GRF (Vertical)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)

# Plot foot GRF forward (similar to part of Fig 7d)
plt.subplot(3, 3, 5)
plt.plot(time_array, grf_array[:, 0, 0])  # Right foot, forward force
plt.title('Right Foot GRF (Forward)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)

# Plot foot CoP (similar to Fig 7e)
plt.subplot(3, 3, 6)
plt.plot(time_array, cop_array[:, 0, 0])  # Right foot CoP, x position
plt.title('Right Foot CoP (Forward)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)

# Plot ankle torque (similar to Fig 7h)
plt.subplot(3, 3, 7)
plt.plot(time_array, ankle_torque_array[:, 0, 0])  # Right ankle, x torque
plt.title('Right Ankle Torque (Roll)')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.grid(True)

# Plot both feet GRF comparison
plt.subplot(3, 3, 8)
plt.plot(time_array, grf_array[:, 0, 1], 'b-', label='Right')
plt.plot(time_array, grf_array[:, 1, 1], 'r-', label='Left')
plt.title('Left & Right Foot GRF (Vertical)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

# Plot both feet CoP comparison
plt.subplot(3, 3, 9)
plt.plot(time_array, cop_array[:, 0, 0], 'b-', label='Right')
plt.plot(time_array, cop_array[:, 1, 0], 'r-', label='Left')
plt.title('Left & Right Foot CoP (Forward)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()