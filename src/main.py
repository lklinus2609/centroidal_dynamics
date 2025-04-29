#!/usr/bin/env python3
# main.py
"""
Main execution script for the momentum-based balance controller experiment.

This script runs the balance control experiment for the Unitree G1 robot
and generates visualizations of the results.
"""

import os
import argparse
import numpy as np

from src.simulation import MujocoSimulation
from src.momentum_controller import momentum_balance_controller
from src.visualization import plot_experiment_results, save_data, create_animation


def get_standing_pose(mj_model):
    """
    Get the initial standing pose for the robot with 45 degree knee bend
    
    Args:
        mj_model: MuJoCo model
    
    Returns:
        q_init: Initial joint configuration
    """
    # Start with zeros
    q_init = np.zeros(mj_model.nq)
    
    # Set quaternion to identity (w=1)
    q_init[3] = 1.0
    
    # Set robot to appropriate standing height (z coordinate)
    # With bent knees, we need to lower the height a bit
    q_init[2] = 0.65  # Adjust based on knee angle
    
    # Use the correct indices from our test script
    left_knee_idx = 10
    right_knee_idx = 16
    left_hip_pitch_idx = 7
    right_hip_pitch_idx = 13
    left_ankle_pitch_idx = 11
    right_ankle_pitch_idx = 17
    
    # Set knee angle to 45 degrees (0.785 radians)
    knee_angle = 0.785  # 45 degrees
    q_init[left_knee_idx] = knee_angle
    q_init[right_knee_idx] = knee_angle
    
    # Adjust ankles to keep feet flat on ground
    ankle_angle = -0.4  # Compensate for knee bend
    q_init[left_ankle_pitch_idx] = ankle_angle
    q_init[right_ankle_pitch_idx] = ankle_angle
    
    # Adjust hip pitch to keep torso upright
    hip_angle = -0.4  # Compensate for knee bend
    q_init[left_hip_pitch_idx] = hip_angle
    q_init[right_hip_pitch_idx] = hip_angle
    
    return q_init


def compute_desired_com(mj_model, mj_data):
    """
    Compute the desired CoM position (above the feet)
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
    
    Returns:
        desired_com: Desired CoM position
    """
    # Find foot positions
    left_foot = None
    right_foot = None
    
    # Search for foot bodies
    for i in range(mj_model.nbody):
        body_name = mj_model.body(i).name.lower()
        if 'left' in body_name and ('foot' in body_name or 'ankle' in body_name):
            left_foot = mj_data.xpos[i].copy()
        elif 'right' in body_name and ('foot' in body_name or 'ankle' in body_name):
            right_foot = mj_data.xpos[i].copy()
    
    # If we couldn't find the feet, use current CoM
    if left_foot is None or right_foot is None:
        desired_com = mj_data.subtree_com[0].copy()
    else:
        # Set desired CoM to midpoint of feet, with current height
        desired_com = (left_foot + right_foot) / 2.0
        desired_com[2] = mj_data.subtree_com[0][2]  # Keep current height
    
    return desired_com


def compare_with_arms_locked(simulation, desired_com, output_dir):
    """
    Compare balance performance with arms locked vs. free
    
    Args:
        simulation: MujocoSimulation object
        desired_com: Desired CoM position
        output_dir: Directory to save results
    
    Returns:
        comparative_data: Dictionary with comparison results
    """
    # This is a placeholder for the arms-locked experiment
    print("Comparison with arms locked not implemented in this version")
    
    return None


def main():
    """Main function to run the balance control experiment"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Momentum-based balance controller for Unitree G1')
    parser.add_argument('--config', default='config/simulation_params.yaml', help='Path to simulation config')
    parser.add_argument('--model', default='models/unitree_g1.xml', help='Path to MuJoCo model')
    parser.add_argument('--output_dir', default='data', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize simulation
    simulation = MujocoSimulation(args.model, args.config)
    
    # Set initial state - standing pose
    q_init = get_standing_pose(simulation.mj_model)
    simulation.set_initial_state(q_init)
    
    # Compute desired CoM position
    desired_com = compute_desired_com(simulation.mj_model, simulation.mj_data)
    print(f"Desired CoM position: {desired_com}")
    
    # Run simulation with momentum-based controller
    print("Running simulation with momentum-based balance controller...")
    simulation_data = simulation.run_simulation(
        momentum_balance_controller,
        desired_com
    )
    
    # Save simulation data
    save_data(simulation_data, os.path.join(args.output_dir, 'simulation_results.npz'))
    
    # Plot results
    print("Plotting results...")
    plot_experiment_results(simulation_data, args.output_dir)
    
    # Create animation
    if simulation.config['visualization']['record_video']:
        print("Creating animation...")
        create_animation(
            simulation_data, 
            simulation.mj_model, 
            os.path.join(args.output_dir, 'animation.mp4')
        )
    
    print(f"Experiment completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()