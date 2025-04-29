#!/usr/bin/env python3
# centroidal_dybamics.py
"""
Centroidal dynamics computations for the momentum-based balance controller.

This module implements the calculations for the Centroidal Momentum Matrix (CMM)
and related quantities, as described in "Centroidal dynamics of a humanoid robot"
by Orin, Goswami & Lee.
"""

import numpy as np
import pinocchio as pin


def compute_centroidal_momentum_matrix(model, data, q):
    """
    Compute the Centroidal Momentum Matrix (CMM) A_G
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint configuration vector (nq x 1)
    
    Returns:
        A_G: Centroidal Momentum Matrix (6 x nv)
    
    Notes:
        The CMM A_G relates joint velocities to centroidal momentum:
            h_G = A_G * q̇
    """
    # Compute the centroidal momentum matrix using Pinocchio
    pin.ccrba(model, data, q)
    
    # Return the CMM
    return data.Ag


def compute_cmm_derivative_qdot(model, data, q, v):
    """
    Compute the product of the time derivative of CMM and joint velocities
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint configuration vector (nq x 1)
        v: Joint velocity vector (nv x 1)
    
    Returns:
        Ȧ_G * q̇: Product of CMM derivative and joint velocities (6 x 1)
    """
    # First compute CCRBA to ensure proper initialization
    pin.ccrba(model, data, q, v)
    
    # Calculate centroidal momentum time variation
    pin.dccrba(model, data, q, v)
    
    # Return the product Ȧ_G * q̇
    return data.dAg * v


def compute_centroidal_dynamics(model, data, q, v):
    """
    Compute full centroidal dynamics (momentum, CoM, etc.)
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint configuration vector (nq x 1)
        v: Joint velocity vector (nv x 1)
    
    Returns:
        dict containing:
            'cmm': Centroidal Momentum Matrix (6 x nv)
            'momentum': Centroidal momentum vector (6 x 1)
            'com_pos': Center of Mass position (3 x 1)
            'com_vel': Center of Mass velocity (3 x 1)
            'ccrbi': Centroidal Composite Rigid Body Inertia (6 x 6)
    """
    # Compute centroidal momentum and matrix
    pin.ccrba(model, data, q, v)
    cmm = data.Ag.copy()
    momentum = data.hg.copy()
    ccrbi = data.Ig.copy()
    
    # Compute CoM position and velocity
    pin.centerOfMass(model, data, q, v, True)
    com_pos = data.com[0].copy()
    com_vel = data.vcom[0].copy()
    
    return {
        'cmm': cmm,
        'momentum': momentum,
        'com_pos': com_pos,
        'com_vel': com_vel,
        'ccrbi': ccrbi
    }


def extract_independent_rows(J, tol=1e-6):
    """
    Extract independent rows from a Jacobian matrix
    
    Args:
        J: Input Jacobian matrix
        tol: Tolerance for rank determination
    
    Returns:
        L: Matrix with independent rows
    """
    # Compute QR decomposition with column pivoting
    Q, R, P = np.linalg.qr(J.T, mode='complete', pivoting=True)
    
    # Determine numerical rank
    abs_diag = np.abs(np.diag(R))
    rank = np.sum(abs_diag > tol * abs_diag[0])
    
    # Extract independent rows
    independent_indices = P[:rank]
    L = J[independent_indices]
    
    return L


def compute_constraint_matrices(L):
    """
    Compute constraint matrices for constrained CMM calculation
    
    Args:
        L: Constraint matrix with independent rows
    
    Returns:
        Q: Permutation matrix separating primary and secondary variables
        L_S: Constraint matrix for secondary variables
        L_P: Constraint matrix for primary variables
    """
    m, n = L.shape  # m: number of constraints, n: total DOFs
    
    # Use QR decomposition to find a good set of primary/secondary variables
    Q, R, P = np.linalg.qr(L.T, mode='complete', pivoting=True)
    
    # Primary variables are the last n-m columns of P
    # Secondary variables are the first m columns of P
    perm = np.zeros((n, n))
    perm[P, np.arange(n)] = 1.0
    
    # Split L into L_S and L_P
    L_perm = L @ perm
    L_S = L_perm[:, :m]
    L_P = L_perm[:, m:]
    
    return perm, L_S, L_P


def compute_constrained_cmm(model, data, q, contact_frames):
    """
    Compute constrained CMM for contacts
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint configuration
        contact_frames: List of frame IDs in contact
    
    Returns:
        A_G^c: Constrained Centroidal Momentum Matrix
    """
    # 1. Compute full CMM
    A_G = compute_centroidal_momentum_matrix(model, data, q)
    
    # 2. No constraints if no contacts
    if not contact_frames:
        return A_G
    
    # 3. Compute contact Jacobians for contact frames
    nc = len(contact_frames)
    J = np.zeros((6 * nc, model.nv))
    
    for i, frame_id in enumerate(contact_frames):
        # Get frame Jacobian (6 x nv matrix)
        pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_frame = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J[6*i:6*(i+1), :] = J_frame
    
    # 4. Extract independent rows to form constraints matrix L
    L = extract_independent_rows(J)
    
    # If there are no independent constraints, return the original CMM
    if L.shape[0] == 0:
        return A_G
    
    # 5. Get permutation matrix Q to separate primary and secondary variables
    Q, L_S, L_P = compute_constraint_matrices(L)
    
    # 6. Split CMM into primary and secondary parts
    m = L.shape[0]  # Number of constraints
    nv = model.nv  # Total DOFs
    
    A_GP = A_G @ Q[:, m:]  # Primary part
    A_GS = A_G @ Q[:, :m]  # Secondary part
    
    # 7. Compute constrained CMM
    # A_G^c = A_GP - A_GS * L_S^-1 * L_P
    L_S_inv = np.linalg.inv(L_S)
    A_G_constrained = A_GP - A_GS @ L_S_inv @ L_P
    
    return A_G_constrained