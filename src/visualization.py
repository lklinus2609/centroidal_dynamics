#!/usr/bin/env python3
# visualization.py
"""
Visualization tools for the balance control experiment.

This module provides functions to visualize and plot the results
of the balance control experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def extract_ground_reaction_forces(contact_forces):
    """
    Extract vertical ground reaction forces from contact data
    
    Args:
        contact_forces: List of contact force data
    
    Returns:
        vertical_forces: Vertical ground reaction forces over time
    """
    vertical_forces = []
    
    for time_step in contact_forces:
        # Sum up all vertical forces
        total_vertical = 0
        for contact in time_step:
            # Extract z-component (vertical)
            total_vertical += contact['force'][2]
        
        vertical_forces.append(total_vertical)
    
    return vertical_forces


def compute_cop_from_forces(contact_forces):
    """
    Compute Center of Pressure (CoP) from contact forces
    
    Args:
        contact_forces: List of contact force data
    
    Returns:
        cop_positions: CoP positions over time
    """
    cop_positions = []
    
    for time_step in contact_forces:
        # If no contacts, use [0, 0, 0]
        if not time_step:
            cop_positions.append(np.zeros(3))
            continue
            
        # Calculate CoP
        total_force = 0
        weighted_pos = np.zeros(3)
        
        for contact in time_step:
            force_magnitude = contact['force'][2]  # Vertical component
            
            if force_magnitude > 0:
                total_force += force_magnitude
                weighted_pos += contact['pos'] * force_magnitude
        
        if total_force > 0:
            cop = weighted_pos / total_force
        else:
            cop = np.zeros(3)
            
        cop_positions.append(cop)
    
    return cop_positions


def compute_joint_torques_from_data(simulation_data):
    """
    Extract joint torques for key joints from simulation data
    
    Args:
        simulation_data: Dictionary of simulation results
    
    Returns:
        joint_torques: Dictionary of joint torques
    """
    if 'joint_torques' not in simulation_data or len(simulation_data['joint_torques']) == 0:
        print("Warning: No joint torque data found!")
        n_steps = len(simulation_data['time'])
        return {
            'ankle_pitch': np.zeros(n_steps),
            'knee': np.zeros(n_steps),
            'hip_pitch': np.zeros(n_steps)
        }
    
    # Extract torques using the correct actuator indices
    # Note: Actuator indices usually correspond to joint indices minus the floating base
    joint_torques = simulation_data['joint_torques']
    
    # Indices based on the actuator ordering in the XML file
    # Hip pitch, knee, and ankle pitch indices
    hip_pitch_indices = [0, 6]  # Left and right hip pitch
    knee_indices = [3, 9]       # Left and right knee
    ankle_pitch_indices = [4, 10]  # Left and right ankle pitch
    
    # Average the torques from left and right legs
    hip_pitch_torque = np.zeros(len(joint_torques))
    knee_torque = np.zeros(len(joint_torques))
    ankle_pitch_torque = np.zeros(len(joint_torques))
    
    # Safely extract torques
    if joint_torques.shape[1] > max(max(hip_pitch_indices), max(knee_indices), max(ankle_pitch_indices)):
        hip_pitch_torque = np.mean([joint_torques[:, i] for i in hip_pitch_indices], axis=0)
        knee_torque = np.mean([joint_torques[:, i] for i in knee_indices], axis=0)
        ankle_pitch_torque = np.mean([joint_torques[:, i] for i in ankle_pitch_indices], axis=0)
    else:
        print(f"Warning: joint_torques shape {joint_torques.shape} too small for indices")
    
    return {
        'hip_pitch': hip_pitch_torque,
        'knee': knee_torque,
        'ankle_pitch': ankle_pitch_torque
    }

def plot_experiment_results(simulation_data, output_dir):
    """
    Plot results similar to Figure 9 in the paper
    
    Args:
        simulation_data: Dictionary containing simulation results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    # Time data
    time = simulation_data['time']
    
    # Plot CoM position (X-forward direction)
    axs[0, 0].plot(time, [pos[0] for pos in simulation_data['com_positions']])
    axs[0, 0].set_title('CoM (forward direction)')
    axs[0, 0].set_xlabel('Time (sec)')
    axs[0, 0].set_ylabel('Position (m)')
    
    # Plot linear momentum (X-forward direction)
    axs[0, 1].plot(time, [mom[0] for mom in simulation_data['linear_momentum']])
    axs[0, 1].set_title('Linear Momentum (forward direction)')
    axs[0, 1].set_xlabel('Time (sec)')
    axs[0, 1].set_ylabel('Momentum (kg⋅m/s)')
    
    # Plot angular momentum (Y-sagittal plane for forward push)
    axs[0, 2].plot(time, [mom[1] for mom in simulation_data['angular_momentum']])
    axs[0, 2].set_title('Angular Momentum (sagittal plane)')
    axs[0, 2].set_xlabel('Time (sec)')
    axs[0, 2].set_ylabel('Momentum (kg⋅m²/s)')
    
    # Plot trunk angle
    axs[1, 0].plot(time, simulation_data['trunk_angles'])
    axs[1, 0].set_title('Trunk Angle')
    axs[1, 0].set_xlabel('Time (sec)')
    axs[1, 0].set_ylabel('Angle (deg)')
    
    # Plot vertical GRF
    contact_forces = extract_ground_reaction_forces(simulation_data['contact_forces'])
    axs[1, 1].plot(time, contact_forces)
    axs[1, 1].set_title('Ground Reaction Force (vertical)')
    axs[1, 1].set_xlabel('Time (sec)')
    axs[1, 1].set_ylabel('Force (N)')
    
    # Plot CoP position
    cop_positions = compute_cop_from_forces(simulation_data['contact_forces'])
    axs[1, 2].plot(time, [cop[0] for cop in cop_positions])
    axs[1, 2].set_title('CoP (forward direction)')
    axs[1, 2].set_xlabel('Time (sec)')
    axs[1, 2].set_ylabel('Position (m)')
    
    # Plot joint torques for key joints
    joint_torques = compute_joint_torques_from_data(simulation_data)
    
    axs[2, 0].plot(time, joint_torques['ankle_pitch'])
    axs[2, 0].set_title('Ankle Pitch Torque')
    axs[2, 0].set_xlabel('Time (sec)')
    axs[2, 0].set_ylabel('Torque (N⋅m)')
    
    axs[2, 1].plot(time, joint_torques['knee'])
    axs[2, 1].set_title('Knee Torque')
    axs[2, 1].set_xlabel('Time (sec)')
    axs[2, 1].set_ylabel('Torque (N⋅m)')
    
    axs[2, 2].plot(time, joint_torques['hip_pitch'])
    axs[2, 2].set_title('Hip Pitch Torque')
    axs[2, 2].set_xlabel('Time (sec)')
    axs[2, 2].set_ylabel('Torque (N⋅m)')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'balance_control_results.png'), dpi=300)
    plt.show()


def create_animation(simulation_data, mj_model, output_file):
    """
    Create animation of the experiment
    
    Args:
        simulation_data: Dictionary containing simulation results
        mj_model: MuJoCo model
        output_file: Path to save animation
    """
    print(f"Creating animation at {output_file}")
    print("Animation creation is not implemented in this version")
    # Animation creation requires a more complex setup with MuJoCo viewer
    # This is a placeholder for future implementation


def save_data(simulation_data, output_file):
    """
    Save simulation data to file
    
    Args:
        simulation_data: Dictionary containing simulation results
        output_file: Path to save data
    """
    # Save data to numpy file
    np.savez(
        output_file,
        time=simulation_data['time'],
        com_positions=simulation_data['com_positions'],
        angular_momentum=simulation_data['angular_momentum'],
        linear_momentum=simulation_data['linear_momentum'],
        trunk_angles=simulation_data['trunk_angles'],
        joint_torques=simulation_data.get('joint_torques', np.array([]))
    )
    
    print(f"Saved simulation data to {output_file}")