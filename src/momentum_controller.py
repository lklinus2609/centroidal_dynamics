#!/usr/bin/env python3
# momentum_controller.py
"""
Momentum-based balance controller implementation.

This module implements the momentum-based balance controller as described in
"A momentum-based balance controller for humanoid robots on non-level and
non-stationary ground" by Lee & Goswami.
"""

import numpy as np
import pinocchio as pin
from .centroidal_dynamics import compute_centroidal_momentum_matrix


def compute_desired_momentum_rate_change(dynamics_data, desired_com, desired_momentum, params):
    """
    Compute the desired centroidal momentum rate change
    
    Args:
        dynamics_data: Dictionary containing current dynamics state
        desired_com: Desired CoM position (3x1)
        desired_momentum: Desired centroidal momentum (6x1)
        params: Controller parameters
    
    Returns:
        h_G_d: Desired centroidal momentum rate change (6x1)
    """
    # Extract current state
    current_momentum = dynamics_data['momentum']
    current_com = dynamics_data['com_pos']
    current_com_vel = dynamics_data['com_vel']
    
    # Extract gains
    Gamma_11 = np.diag(params['Gamma_11'])
    Gamma_12 = np.diag(params['Gamma_12'])
    Gamma_21 = np.diag(params['Gamma_21'])
    
    # Desired CoM velocity (zero for balance)
    desired_com_vel = np.zeros(3)
    
    # Extract momentum components - handling Force objects properly
    if hasattr(current_momentum, 'angular'):
        # It's a Pinocchio Force object
        current_ang_mom = np.array(current_momentum.angular)
        current_lin_mom = np.array(current_momentum.linear)
    else:
        # It's a numpy array
        current_ang_mom = current_momentum[:3]
        current_lin_mom = current_momentum[3:]
    
    if hasattr(desired_momentum, 'angular'):
        # It's a Pinocchio Force object
        desired_ang_mom = np.array(desired_momentum.angular)
        desired_lin_mom = np.array(desired_momentum.linear)
    else:
        # It's a numpy array
        desired_ang_mom = desired_momentum[:3]
        desired_lin_mom = desired_momentum[3:]
    
    # Compute desired momentum rate changes (Equations 57-58)
    k_G_d = Gamma_21 @ (desired_ang_mom - current_ang_mom)
    l_G_d = Gamma_11 @ (desired_com_vel - current_com_vel) + Gamma_12 @ (desired_com - current_com)
    
    # Combine into 6x1 vector
    h_G_d = np.concatenate([k_G_d, l_G_d])
    
    return h_G_d


def create_contact_model(pin_model, pin_data, q, contact_frames):
    """
    Create a contact model for constraint handling
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        q: Joint configuration
        contact_frames: List of frame IDs in contact
    
    Returns:
        contact_model: Contact model with constraint information
    """
    contact_model = {
        'frames': contact_frames,
        'jacobians': [],
        'positions': [],
        'normals': [],
        'friction_coeffs': []
    }
    
    # Compute forward kinematics
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    
    # Compute Jacobians and positions for all contact frames
    for frame_id in contact_frames:
        # Get frame position
        frame_position = pin_data.oMf[frame_id].translation
        contact_model['positions'].append(frame_position)
        
        # Compute frame Jacobian
        pin.computeFrameJacobian(pin_model, pin_data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_frame = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        contact_model['jacobians'].append(J_frame)
        
        # Default normal (assuming horizontal ground)
        contact_model['normals'].append(np.array([0., 0., 1.]))
        
        # Default friction coefficient
        contact_model['friction_coeffs'].append(0.7)
    
    return contact_model


def compute_admissible_linear_momentum_rate(l_G_d, state, contact_model):
    """
    Compute the admissible linear momentum rate change
    
    Args:
        l_G_d: Desired linear momentum rate change (3x1)
        state: Current robot state
        contact_model: Contact constraints
    
    Returns:
        l_G_a: Admissible linear momentum rate change (3x1)
    """
    # In the simplest case (no friction constraints), we can use l_G_d directly
    # This assumes the robot is properly supported
    
    # TODO: Implement more sophisticated constraints if necessary
    # - ZMP/CoP constraints
    # - Friction cone constraints
    
    # For now, just return the desired value
    return l_G_d.copy()


def compute_admissible_angular_momentum_rate(k_G_d, l_G_a, state, contact_model):
    """
    Compute the admissible angular momentum rate change
    
    Args:
        k_G_d: Desired angular momentum rate change (3x1)
        l_G_a: Admissible linear momentum rate change (3x1)
        state: Current robot state
        contact_model: Contact constraints
    
    Returns:
        k_G_a: Admissible angular momentum rate change (3x1)
    """
    # In the simplest case, we can use k_G_d directly
    # This is valid when there are no conflicting constraints
    
    # TODO: Implement more sophisticated constraint handling
    # - CoP constraints
    # - Physical realizability
    
    # For now, just return the desired value
    return k_G_d.copy()


def compute_admissible_momentum_rate_change(h_G_d, robot_state, contact_model):
    """
    Compute the admissible momentum rate change considering physical constraints
    
    Args:
        h_G_d: Desired momentum rate change (6x1)
        robot_state: Current robot state
        contact_model: Model of the contact constraints
    
    Returns:
        h_G_a: Admissible momentum rate change (6x1)
    """
    # Extract components, handling Force objects properly
    if hasattr(h_G_d, 'angular'):
        # It's a Pinocchio Force object
        k_G_d = np.array(h_G_d.angular)
        l_G_d = np.array(h_G_d.linear)
    else:
        # It's a numpy array
        k_G_d = h_G_d[:3]  # Desired angular momentum rate change
        l_G_d = h_G_d[3:]  # Desired linear momentum rate change
    
    # First, try to realize the desired linear momentum rate change
    l_G_a = compute_admissible_linear_momentum_rate(l_G_d, robot_state, contact_model)
    
    # Then, calculate the admissible angular momentum rate given l_G_a
    k_G_a = compute_admissible_angular_momentum_rate(k_G_d, l_G_a, robot_state, contact_model)
    
    # Combine into 6x1 vector
    h_G_a = np.concatenate([k_G_a, l_G_a])
    
    return h_G_a


def compute_desired_posture(q, pin_model):
    """
    Compute desired joint accelerations for posture control
    
    Args:
        q: Current joint configuration
        pin_model: Pinocchio model
    
    Returns:
        q_ddot_posture: Desired joint accelerations for posture
    """
    # Use model's velocity dimension directly
    nv = pin_model.nv
    q_ddot_posture = np.zeros(nv)
    
    # Define reference pose (standing upright)
    q_ref = q.copy()
    
    # Simple PD controller to maintain the reference pose
    kp = 10.0  # Position gain
    
    # Only apply to actuated joints (indices 6 and higher) if floating base
    if nv >= 6:  # If model has floating base
        # Apply only to actuated joints
        q_ddot_posture[6:] = kp * (q_ref[6:nv] - q[6:nv])
    
    return q_ddot_posture


def determine_control_weight(dynamics_data, contact_model, params):
    """
    Determine control weight based on stability margin
    
    Args:
        dynamics_data: Current dynamics state
        contact_model: Contact constraints
        params: Controller parameters
    
    Returns:
        w: Weight between momentum and posture objectives (0 to 1)
    """
    # Default weight
    w = 0.4
    
    # TODO: Implement adaptive weighting based on stability margin
    # For example, increase w when CoP approaches support boundary
    
    return w


def compute_cmm_derivative_qdot(pin_model, pin_data, q, v):
    """
    Compute the product of the time derivative of CMM and joint velocities
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        q: Joint configuration
        v: Joint velocities
    
    Returns:
        dAg_qdot: Product of CMM time derivative and joint velocities (6x1)
    """
    # Calculate centroidal momentum time variation
    pin.computeCentroidalMapTimeVariation(pin_model, pin_data, q, v)
    
    # Return the product of dAg and v (6-dimensional vector)
    # This is more efficient than computing full dAg matrix then multiplying by v
    dAg_v = pin_data.dAg * v  # This should return a 6D vector
    
    return dAg_v


def compute_joint_accelerations(pin_model, pin_data, q, v, h_G_a, A_G, desired_posture, w):
    """
    Compute joint accelerations to achieve admissible momentum rate change
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        q: Joint configuration
        v: Joint velocities
        h_G_a: Admissible momentum rate change (6x1)
        A_G: Centroidal Momentum Matrix
        desired_posture: Desired joint accelerations for posture
        w: Weight between momentum and posture objectives (0 to 1)
    
    Returns:
        q_ddot: Joint accelerations (nx1)
    """
    # Get model dimensions
    nv = pin_model.nv
    
    # Ensure A_G is a numpy array with correct dimensions
    A_G_np = np.array(A_G)
    
    # Get CMM derivative * qdot
    # Use Pinocchio's built-in functions for efficiency
    pin.computeCentroidalMapTimeVariation(pin_model, pin_data, q, v)
    dAg_qdot = np.zeros(6)
    # Compute dAg * v (this should be a 6D vector)
    for i in range(nv):
        dAg_qdot += pin_data.dAg[:, i] * v[i]
    
    # Ensure h_G_a is a 6D vector
    if hasattr(h_G_a, 'angular'):
        h_G_a_np = np.concatenate([
            np.array(h_G_a.angular),
            np.array(h_G_a.linear)
        ])
    else:
        h_G_a_np = np.array(h_G_a).flatten()
    
    # Ensure dimensions are consistent
    if len(desired_posture) != nv:
        print(f"Warning: Reshaping desired_posture from {len(desired_posture)} to {nv}")
        temp = np.zeros(nv)
        min_len = min(len(desired_posture), nv)
        temp[:min_len] = desired_posture[:min_len]
        desired_posture = temp
    
    # Create matrices for least-squares problem
    A_momentum = np.sqrt(w) * A_G_np  # (6, nv)
    A_posture = np.sqrt(1-w) * np.eye(nv)  # (nv, nv)
    A = np.vstack([A_momentum, A_posture])  # (6+nv, nv)
    
    b_momentum = np.sqrt(w) * (h_G_a_np - dAg_qdot)  # (6,)
    b_posture = np.sqrt(1-w) * desired_posture  # (nv,)
    b = np.concatenate([b_momentum, b_posture])  # (6+nv,)
    
    # Solve the least-squares problem
    q_ddot, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return q_ddot


def compute_control_torques(pin_model, pin_data, q, v, a):
    """
    Compute control torques using inverse dynamics
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        q: Joint configuration
        v: Joint velocities
        a: Joint accelerations
    
    Returns:
        tau: Joint torques (n-6 x 1, excluding floating base)
    """
    # Compute joint torques using inverse dynamics
    pin.rnea(pin_model, pin_data, q, v, a)
    tau = pin_data.tau.copy()

    # damping added for bugging
    damping = 2.0
    tau_damping = -damping * v
    tau += tau_damping

    # Return only actuated joint torques (exclude floating base)
    return tau[6:] if len(tau) > 6 else tau


def momentum_balance_controller(pin_model, pin_data, q, v, desired_com, contact_frames, params):
    """
    Main momentum-based balance controller function
    
    Args:
        pin_model: Pinocchio model
        pin_data: Pinocchio data
        q: Joint configuration
        v: Joint velocities
        desired_com: Desired CoM position
        contact_frames: List of frames in contact
        params: Controller parameters
    
    Returns:
        tau: Control torques for actuated joints
    """
    # 1. Update forward kinematics
    pin.forwardKinematics(pin_model, pin_data, q, v)
    pin.updateFramePlacements(pin_model, pin_data)
    
    # 2. Compute centroidal dynamics
    pin.ccrba(pin_model, pin_data, q, v)
    pin.centerOfMass(pin_model, pin_data, q, v, True)
    
    # Ensure we're working with properly formatted data
    dynamics_data = {
        'cmm': pin_data.Ag,  # This is potentially a reference to Pinocchio's data
        'momentum': pin_data.hg,  # This is a Force object in Pinocchio
        'com_pos': pin_data.com[0],  # Position vector
        'com_vel': pin_data.vcom[0],  # Velocity vector
        'ccrbi': pin_data.Ig  # Inertia tensor
    }
    
    # 3. Compute desired centroidal momentum (zero angular momentum for balance)
    desired_momentum = np.zeros(6)
    
    # 4. Compute desired momentum rate change
    h_G_d = compute_desired_momentum_rate_change(
        dynamics_data, 
        desired_com, 
        desired_momentum, 
        params
    )
    
    # 5. Compute admissible momentum rate change (respecting physical constraints)
    contact_model = create_contact_model(pin_model, pin_data, q, contact_frames)
    robot_state = {'q': q, 'v': v}
    h_G_a = compute_admissible_momentum_rate_change(h_G_d, robot_state, contact_model)
    
    # 6. Get CMM (constrained if in contact)
    A_G = dynamics_data['cmm']
    
    # 7. Compute desired posture accelerations
    desired_posture = compute_desired_posture(q, pin_model)
    
    # 8. Determine control weight
    w = determine_control_weight(dynamics_data, contact_model, params)
    
    # 9. Compute joint accelerations
    q_ddot = compute_joint_accelerations(
        pin_model, pin_data, q, v, h_G_a, A_G, desired_posture, w
    )
    
    # 10. Compute control torques
    tau = compute_control_torques(pin_model, pin_data, q, v, q_ddot)
    
    return tau