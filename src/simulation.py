#!/usr/bin/env python3
# simulation.py
"""
MuJoCo simulation environment for the balance control experiment.

This module handles the physics simulation of the Unitree G1 robot
and the application of external disturbances.
"""

import os
import numpy as np
import mujoco
import yaml
import pinocchio as pin

from .state_conversion import mj_to_pin_position, mj_to_pin_velocity, pin_to_mj_torque


def load_yaml(file_path):
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def find_closest_body_to_point(mj_model, mj_data, point):
    """Find the body index closest to a point in space"""
    min_dist = float('inf')
    closest_body = 0
    
    for i in range(1, mj_model.nbody):  # Skip the world body
        body_pos = mj_data.xpos[i]
        dist = np.linalg.norm(body_pos - point)
        if dist < min_dist:
            min_dist = dist
            closest_body = i
            
    return closest_body


def get_contact_frames(pin_model):
    """Get the frame IDs for contact points (feet)"""
    # Search for frames that correspond to feet
    contact_frames = []
    
    # Common frame name patterns for feet
    foot_patterns = ['foot', 'ankle', 'sole']
    
    for i in range(pin_model.nframes):
        frame_name = pin_model.frames[i].name
        if any(pattern in frame_name.lower() for pattern in foot_patterns):
            if 'left' in frame_name.lower() or 'right' in frame_name.lower():
                contact_frames.append(i)
    
    return contact_frames


def get_trunk_angle(mj_model, mj_data):
    """Get the trunk bend angle in degrees"""
    # Find trunk body index (usually the torso or pelvis)
    trunk_idx = -1
    for i in range(mj_model.nbody):
        if 'torso' in mj_model.body(i).name.lower() or 'trunk' in mj_model.body(i).name.lower():
            trunk_idx = i
            break
    
    if trunk_idx == -1:
        return 0.0
    
    # Get rotation matrix of the trunk
    rot = mj_data.xmat[trunk_idx].reshape(3, 3)
    
    # Calculate angle around y-axis (pitch)
    angle_y = np.arctan2(-rot[2, 0], np.sqrt(rot[0, 0]**2 + rot[1, 0]**2))
    
    # Convert to degrees
    return np.degrees(angle_y)


class MujocoSimulation:
    """Class to handle MuJoCo simulation for the balance control experiment"""
    
    def __init__(self, model_path, config_path):
        """
        Initialize the simulation
        
        Args:
            model_path: Path to MuJoCo XML model
            config_path: Path to simulation configuration file
        """
        # Load configuration
        self.config = load_yaml(config_path)
        
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Initialize MuJoCo renderer if visualization is enabled
        self.viewer = None
        if self.config['visualization']['record_video']:
            try:
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.mj_model, self.mj_data)
            except ImportError:
                print("Warning: mujoco.viewer not available, visualization disabled")
                self.config['visualization']['record_video'] = False
        
        # Set simulation parameters
        self.mj_model.opt.timestep = self.config['simulation']['timestep']
        self.mj_model.opt.gravity = tuple(self.config['simulation']['gravity'])
        
        # Initialize Pinocchio model from URDF file
        urdf_path = model_path.replace('.xml', '.urdf')
        if not os.path.exists(urdf_path):
            # Try to find URDF in the same directory
            base_dir = os.path.dirname(model_path)
            for file in os.listdir(base_dir):
                if file.endswith('.urdf'):
                    urdf_path = os.path.join(base_dir, file)
                    break
                    
        # Load Pinocchio model
        try:
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
        except Exception as e:
            print(f"Error loading Pinocchio model from {urdf_path}: {e}")
            raise
            
        print(f"Loaded Pinocchio model with {self.pin_model.nq} position DoFs and {self.pin_model.nv} velocity DoFs")
        
        # Initialize data recording
        self.reset_data_recording()
    
    def reset_data_recording(self):
        """Reset data recording structures"""
        self.time_data = []
        self.com_positions = []
        self.joint_angles = []
        self.joint_velocities = []
        self.angular_momentum = []
        self.linear_momentum = []
        self.trunk_angles = []
        self.contact_forces = []
        self.joint_torques = []
    
    def set_initial_state(self, q_init):
        """Set initial robot configuration"""
        if len(q_init) != self.mj_model.nq:
            raise ValueError(f"Initial configuration dimension mismatch: expected {self.mj_model.nq}, got {len(q_init)}")
            
        self.mj_data.qpos[:] = q_init
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def apply_disturbance(self, t):
        """
        Apply external disturbance force based on configuration
        
        Args:
            t: Current simulation time
        """
        disturbance = self.config['disturbance']
        
        # Check if we should apply the push now
        if disturbance['push_time'] <= t < disturbance['push_time'] + disturbance['push_duration']:
            # Get push parameters
            force = disturbance['push_force']
            direction = np.array(disturbance['push_direction'])
            
            # Apply the force
            if disturbance['push_location'] == "COM":
                # Find body index closest to CoM
                com = self.mj_data.subtree_com[0].copy()
                body_idx = find_closest_body_to_point(self.mj_model, self.mj_data, com)
                
                # Apply force to this body
                self.mj_data.xfrc_applied[body_idx, :3] = force * direction
            else:
                # Apply to specified body
                body_idx = self.mj_model.body(disturbance['push_location']).id
                self.mj_data.xfrc_applied[body_idx, :3] = force * direction
        else:
            # Clear any applied forces
            self.mj_data.xfrc_applied[:] = 0.0
    
    def record_data(self, t):
        """
        Record data for analysis
        
        Args:
            t: Current simulation time
        """
        # Convert state to Pinocchio format
        q = mj_to_pin_position(self.mj_data, self.pin_model)
        v = mj_to_pin_velocity(self.mj_data, self.pin_model)
        
        # Update Pinocchio model
        pin.forwardKinematics(self.pin_model, self.pin_data, q, v)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # Compute centroidal dynamics
        pin.ccrba(self.pin_model, self.pin_data, q, v)
        pin.centerOfMass(self.pin_model, self.pin_data, q, v, True)
        
        # Record data
        self.time_data.append(t)
        self.com_positions.append(self.pin_data.com[0].copy())
        self.joint_angles.append(q.copy())
        self.joint_velocities.append(v.copy())
        self.angular_momentum.append(np.array(self.pin_data.hg.angular))
        self.linear_momentum.append(np.array(self.pin_data.hg.linear))
        self.trunk_angles.append(get_trunk_angle(self.mj_model, self.mj_data))
        self.joint_torques.append(self.mj_data.ctrl.copy())  # Add this line to record torques
        
        # Record contact forces
        contact_forces = []
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
            contact_forces.append({
                'pos': contact.pos.copy(),
                'force': force.copy()
            })
        self.contact_forces.append(contact_forces)
    
    def run_simulation(self, controller, desired_com, duration=None):
        """
        Run the simulation with the given controller
        
        Args:
            controller: Balance controller function
            desired_com: Desired CoM position
            duration: Simulation duration (if None, use config value)
        
        Returns:
            simulation_data: Recorded simulation data
        """
        if duration is None:
            duration = self.config['simulation']['duration']

        # Reset simulation
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.reset_data_recording()
        
        # Initialize contact frames (feet)
        contact_frames = get_contact_frames(self.pin_model)
        print(f"Contact frames: {[self.pin_model.frames[i].name for i in contact_frames]}")
        
        # Load controller parameters
        controller_params = self.config['controller']
        
        # Simulation loop
        t = 0
        dt = self.mj_model.opt.timestep
        
        print(f"Starting simulation for {duration}s with dt={dt}s")
        
        while t < duration:
            # Get current state in Pinocchio format
            q = mj_to_pin_position(self.mj_data, self.pin_model)
            v = mj_to_pin_velocity(self.mj_data, self.pin_model)
            
            # Apply external disturbance if needed
            self.apply_disturbance(t)
            
            # Compute control torques
            tau = controller(
                self.pin_model, 
                self.pin_data, 
                q, 
                v, 
                desired_com, 
                contact_frames, 
                controller_params
            )
            
            # Apply torques to MuJoCo
            self.mj_data.ctrl[:] = pin_to_mj_torque(tau, self.mj_model)
            
            # Step simulation
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            # Record data at appropriate intervals
            if int(t / dt) % 10 == 0:  # Record every 10 timesteps
                self.record_data(t)
            
            # Update viewer if enabled
            if self.viewer is not None:
                self.viewer.sync()
            
            # Update time
            t += dt
            
            # Print progress
            if int(t / dt) % 1000 == 0:
                print(f"Simulation progress: {t:.2f}s / {duration:.2f}s ({t/duration*100:.1f}%)")
        
        # Compile all recorded data
        simulation_data = {
            'time': np.array(self.time_data),
            'com_positions': np.array(self.com_positions),
            'joint_angles': np.array(self.joint_angles),
            'joint_velocities': np.array(self.joint_velocities),
            'angular_momentum': np.array(self.angular_momentum),
            'linear_momentum': np.array(self.linear_momentum),
            'trunk_angles': np.array(self.trunk_angles),
            'contact_forces': self.contact_forces,
            'joint_torques': np.array(self.joint_torques)
        }
        
        print("Simulation completed successfully")
        
        return simulation_data