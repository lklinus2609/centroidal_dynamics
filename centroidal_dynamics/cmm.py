# centroidal_dynamics/cmm.py
import numpy as np
import pinocchio as pin

def compute_cmm(robot_model, robot_data, q):
    """
    Compute the Centroidal Momentum Matrix (CMM) which maps joint velocities to centroidal momentum.
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
        
    Returns:
    --------
    AG: np.ndarray
        Centroidal Momentum Matrix (6 x n)
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Compute the CMM
    AG = pin.computeCentroidalMap(robot_model, robot_data, q)
    
    return AG

def compute_centroidal_momentum(robot_model, robot_data, q, v):
    """
    Compute the centroidal momentum (angular and linear).
    
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
    hG: pin.Force
        Centroidal momentum (6D spatial force)
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Compute centroidal momentum
    hG = pin.computeCentroidalMomentum(robot_model, robot_data)
    
    return hG

def compute_centroidal_inertia(robot_model, robot_data, q):
    """
    Compute the Centroidal Composite Rigid Body Inertia (CCRBI).
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
        
    Returns:
    --------
    IG: np.ndarray
        Centroidal inertia matrix (6x6)
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Create a zero velocity vector
    v = np.zeros(robot_model.nv)
    
    # Compute the CCRBI
    pin.ccrba(robot_model, robot_data, q, v)
    IG = robot_data.Ig.copy()
    
    return IG

def compute_cmm_derivative(robot_model, robot_data, q, v):
    """
    Compute the time derivative of the Centroidal Momentum Matrix.
    
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
    dAG: np.ndarray
        Time derivative of CMM (6 x n)
    """
    # Compute centroidal dynamics
    AG = compute_cmm(robot_model, robot_data, q)
    hG = compute_centroidal_momentum(robot_model, robot_data, q, v)
    
    # Calculate the bias term (ḣG - AG q̈ = dAG q̇)
    pin.computeCentroidalMapTimeVariation(robot_model, robot_data, q, v)
    dAG = robot_data.dAg
    
    return dAG