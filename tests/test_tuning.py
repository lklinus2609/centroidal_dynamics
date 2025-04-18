# tests/test_balance_controller_tuning.py
"""
Script to test different gain values for the momentum-based balance controller
and compare their convergence properties.
"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import sys
import os
from itertools import product

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centroidal_dynamics.balance_control import BalanceController

# Load the robot model
robot_model = pin.buildModelFromUrdf('/home/linus/robotics/g1_23dof.urdf')

# Define contact frame IDs
contact_frame_ids = [16, 30]  # left_ankle_roll_link, right_ankle_roll_link

# Simulation parameters
dt = 0.01
simulation_time = 5.0  # 5 seconds
num_steps = int(simulation_time / dt)

# Define external push
push_time = 1.0  # Apply push after 1 second
push_duration = 0.1  # Push lasts for 0.1 seconds
push_force = np.array([50.0, 0, 0])  # 50N in x-direction

# Define gain combinations to test - use smaller values to find stable configurations
pos_gain_values = [0.5, 1.0, 2.0]  # Position gain scaling factors (lower values)
vel_gain_values = [2.0, 5.0, 10.0]  # Velocity gain scaling factors (lower values)
ang_p_gain_values = [2.0, 3.0, 5.0]  # Angular P gain values (lower values)
ang_d_gain_values = [1.0, 2.0, 4.0]  # Angular D gain values (increased damping)
lin_d_gain_values = [2.0, 5.0, 10.0]  # Linear momentum damping values (increased damping)

# To limit the number of combinations, we'll test a small subset
# Focus on lower gains with higher damping
gain_configurations = [
    # (pos_scale, vel_scale, ang_p, ang_d, lin_d)
    (0.5, 2.0, 2.0, 2.0, 5.0),   # Low gains, medium damping
    (1.0, 5.0, 3.0, 2.0, 5.0),   # Medium gains, medium damping
    (2.0, 10.0, 5.0, 2.0, 5.0),  # High gains, medium damping
    
    (0.5, 2.0, 2.0, 4.0, 10.0),  # Low gains, high damping
    (1.0, 5.0, 3.0, 4.0, 10.0),  # Medium gains, high damping
    (2.0, 10.0, 5.0, 4.0, 10.0), # High gains, high damping
    
    (0.25, 1.0, 1.0, 4.0, 10.0), # Very low gains, high damping
    (0.5, 2.0, 1.0, 4.0, 10.0),  # Low-P, low-AP, high damping
    (0.5, 1.0, 2.0, 4.0, 10.0),  # Low-P, low-V, high damping
]

# Results dictionary
results = {}

# For each gain combination
for config in gain_configurations:
    pos_scale, vel_scale, ang_p, ang_d, lin_d = config
    print(f"Testing gains: Pos={pos_scale}, Vel={vel_scale}, AngP={ang_p}, AngD={ang_d}, LinD={lin_d}")
    
    # Initialize controller
    controller = BalanceController(robot_model, contact_frame_ids=contact_frame_ids)
    controller.dt = dt
    
    # Create data structure for this model instance
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
    
    # Set gains for this test
    mass = sum(robot_model.inertias[i].mass for i in range(robot_model.njoints))
    gain_pos = np.diag([pos_scale, pos_scale/2, pos_scale]) / mass 
    gain_vel = np.diag([vel_scale, vel_scale/2, vel_scale]) / mass
    gain_ang_mom = np.diag([ang_p, ang_p, ang_p])
    gain_ang_mom_damping = np.diag([ang_d, ang_d, ang_d])
    gain_lin_mom_damping = np.diag([lin_d, lin_d/2, lin_d]) / mass
    
    # Arrays to store results
    time_array = np.zeros(num_steps)
    com_array = np.zeros((num_steps, 3))
    angular_momentum_array = np.zeros((num_steps, 3))
    linear_momentum_array = np.zeros((num_steps, 3))
    
    # Track stability metrics
    stability_metrics = {
        'oscillation_count': 0,
        'max_deviation': 0,
        'final_deviation': 0
    }
    
    # Simple integration simulation
    prev_sign = 0
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
        
        # Track oscillations (zero crossings)
        if i > 0 and push_time < time:
            current_sign = np.sign(com[0] - com_des[0])
            if current_sign != 0 and prev_sign != 0 and current_sign != prev_sign:
                stability_metrics['oscillation_count'] += 1
            prev_sign = current_sign if current_sign != 0 else prev_sign
            
            # Track maximum deviation
            deviation = abs(com[0] - com_des[0])
            if deviation > stability_metrics['max_deviation']:
                stability_metrics['max_deviation'] = deviation
        
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
    
    # Store final deviation
    stability_metrics['final_deviation'] = abs(com_array[-1, 0] - com_des[0])
    
    # Store results for this gain combination
    gain_key = f"P{pos_scale}_V{vel_scale}_AP{ang_p}_AD{ang_d}_LD{lin_d}"
    results[gain_key] = {
        'time': time_array,
        'com': com_array,
        'angular_momentum': angular_momentum_array,
        'linear_momentum': linear_momentum_array,
        'params': {
            'pos_gain': pos_scale,
            'vel_gain': vel_scale,
            'ang_p_gain': ang_p,
            'ang_d_gain': ang_d,
            'lin_d_gain': lin_d
        },
        'metrics': stability_metrics
    }

# Plot results
plt.figure(figsize=(15, 10))

# Plot CoM x-position for each gain combination
plt.subplot(2, 2, 1)
for key, data in results.items():
    p = data['params']
    label = f"P={p['pos_gain']}, V={p['vel_gain']}, AP={p['ang_p_gain']}, AD={p['ang_d_gain']}, LD={p['lin_d_gain']}"
    plt.plot(data['time'], data['com'][:, 0], label=label)

plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Show desired position
plt.title('CoM Position (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend(fontsize='xx-small')

# Plot Linear Momentum x-component
plt.subplot(2, 2, 2)
for key, data in results.items():
    p = data['params']
    label = f"P={p['pos_gain']}, V={p['vel_gain']}, AP={p['ang_p_gain']}, AD={p['ang_d_gain']}, LD={p['lin_d_gain']}"
    plt.plot(data['time'], data['linear_momentum'][:, 0], label=label)

plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Show desired momentum
plt.title('Linear Momentum (X-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m/s)')
plt.grid(True)
plt.legend(fontsize='xx-small')

# Plot Angular Momentum y-component
plt.subplot(2, 2, 3)
for key, data in results.items():
    p = data['params']
    label = f"P={p['pos_gain']}, V={p['vel_gain']}, AP={p['ang_p_gain']}, AD={p['ang_d_gain']}, LD={p['lin_d_gain']}"
    plt.plot(data['time'], data['angular_momentum'][:, 1], label=label)

plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Show desired momentum
plt.title('Angular Momentum (Y-axis)')
plt.xlabel('Time (s)')
plt.ylabel('Momentum (kg*m²/s)')
plt.grid(True)
plt.legend(fontsize='xx-small')

# Plot stability metrics as bar chart
plt.subplot(2, 2, 4)
metrics = ['oscillation_count', 'max_deviation', 'final_deviation']
metric_labels = ['Oscillation Count', 'Max Deviation (m)', 'Final Deviation (m)']
metric_data = np.zeros((len(results), len(metrics)))
config_labels = []

for i, (key, data) in enumerate(results.items()):
    for j, metric in enumerate(metrics):
        metric_data[i, j] = data['metrics'][metric]
    
    p = data['params']
    config_labels.append(f"P{p['pos_gain']}_V{p['vel_gain']}_AP{p['ang_p_gain']}_AD{p['ang_d_gain']}_LD{p['lin_d_gain']}")

x = np.arange(len(config_labels))
width = 0.25

for i, metric in enumerate(metrics):
    plt.bar(x + (i - 1) * width, metric_data[:, i], width, label=metric_labels[i])

plt.xlabel('Gain Configuration')
plt.ylabel('Metric Value')
plt.title('Stability Metrics')
plt.xticks(x, config_labels, rotation=90, fontsize='xx-small')
plt.legend(fontsize='small')

plt.tight_layout()
plt.savefig('gain_tuning_results.png', dpi=300)  # Save the figure
plt.show()

# Print the best parameter set based on stability metrics
# Best = lowest combination of oscillations, max deviation, and final deviation
normalized_metrics = metric_data.copy()
for j in range(metric_data.shape[1]):
    if np.max(metric_data[:, j]) > 0:
        normalized_metrics[:, j] = metric_data[:, j] / np.max(metric_data[:, j])

combined_score = np.sum(normalized_metrics, axis=1)
best_idx = np.argmin(combined_score)

print(f"\nBest parameter set (based on combined stability metrics):")
print(f"  {config_labels[best_idx]}")
print(f"  Oscillation count: {metric_data[best_idx, 0]}")
print(f"  Max deviation: {metric_data[best_idx, 1]:.4f} meters")
print(f"  Final deviation: {metric_data[best_idx, 2]:.4f} meters")

# Print rankings of all configurations
print("\nRanking of all configurations:")
ranking = np.argsort(combined_score)
for i, idx in enumerate(ranking):
    print(f"{i+1}. {config_labels[idx]}")
    print(f"   Oscillation count: {metric_data[idx, 0]}")
    print(f"   Max deviation: {metric_data[idx, 1]:.4f} meters")
    print(f"   Final deviation: {metric_data[idx, 2]:.4f} meters")
    print(f"   Combined score: {combined_score[idx]:.4f}")