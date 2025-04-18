# centroidal_dynamics/avg_velocity.py
import numpy as np
import pinocchio as pin
from .cmm import compute_centroidal_momentum, compute_centroidal_inertia

def compute_average_spatial_velocity(robot_model, robot_data, q, v):
    """
    Compute the "average spatial velocity" of the humanoid robot.
    
    This is defined as vG = (IG)^-1 * hG, where IG is the centroidal inertia
    and hG is the centroidal momentum.
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
    v: array-like
        Joint velocity vector
        
    Returns:
    --------
    vG: np.ndarray
        Average spatial velocity (6D vector: [angular_velocity, linear_velocity])
    """
    # Compute centroidal momentum
    hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
    
    # Compute centroidal inertia
    IG = compute_centroidal_inertia(robot_model, robot_data, q)
    
    # Compute average spatial velocity (Equation 24 in the paper)
    vG = np.linalg.solve(IG, hG.vector)
    
    return vG

def compute_kinetic_energy(robot_model, robot_data, q, v):
    """
    Compute the kinetic energy of the robot and its decomposition.
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
    v: array-like
        Joint velocity vector
        
    Returns:
    --------
    T: float
        Total kinetic energy
    T_avg: float
        Kinetic energy due to average spatial velocity
    T_rel: float
        Kinetic energy due to relative motion
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Compute total kinetic energy
    T = pin.computeKineticEnergy(robot_model, robot_data)
    
    # Compute centroidal quantities
    IG = compute_centroidal_inertia(robot_model, robot_data, q)
    vG = compute_average_spatial_velocity(robot_model, robot_data, q, v)
    
    # Compute kinetic energy from average spatial velocity (Equation 30, first term)
    T_avg = 0.5 * vG.dot(IG @ vG)
    
    # Compute kinetic energy from relative motion (Equation 30, second term)
    T_rel = T - T_avg
    
    return T, T_avg, T_rel