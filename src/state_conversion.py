#!/usr/bin/env python3
# state_conversion.py
"""
Handles conversion between MuJoCo and Pinocchio states.

This module provides utility functions to convert between MuJoCo and Pinocchio
state representations, accounting for the different conventions used by each library.
"""

import numpy as np


def mj_to_pin_position(mj_data, pin_model):
    """
    Convert MuJoCo position to Pinocchio position
    
    Args:
        mj_data: MuJoCo data
        pin_model: Pinocchio model
    
    Returns:
        q: Pinocchio position vector (nq x 1)
    
    Notes:
        MuJoCo and Pinocchio use different representations for the floating base:
        - MuJoCo: [pos_xyz(3), quat_wxyz(4), joints(...)]
        - Pinocchio: [pos_xyz(3), quat_xyzw(4), joints(...)]
    """
    q = np.zeros(pin_model.nq)
    
    # Copy position
    q[0:3] = mj_data.qpos[0:3]
    
    # Convert quaternion (wxyz -> xyzw)
    q[3] = mj_data.qpos[4]  # x
    q[4] = mj_data.qpos[5]  # y
    q[5] = mj_data.qpos[6]  # z
    q[6] = mj_data.qpos[3]  # w
    
    # Copy joint positions
    q[7:] = mj_data.qpos[7:pin_model.nq]
    
    return q


def mj_to_pin_velocity(mj_data, pin_model):
    """
    Convert MuJoCo velocity to Pinocchio velocity
    
    Args:
        mj_data: MuJoCo data
        pin_model: Pinocchio model
    
    Returns:
        v: Pinocchio velocity vector (nv x 1)
    
    Notes:
        MuJoCo uses [linear_vel(3), angular_vel(3), joint_vel(...)]
        Pinocchio uses [angular_vel(3), linear_vel(3), joint_vel(...)]
    """
    v = np.zeros(pin_model.nv)
    
    # Convert base velocity (swap angular and linear)
    v[0:3] = mj_data.qvel[3:6]  # Angular velocity
    v[3:6] = mj_data.qvel[0:3]  # Linear velocity
    
    # Copy joint velocities
    v[6:] = mj_data.qvel[6:pin_model.nv]
    
    return v


def pin_to_mj_position(q, mj_model):
    """
    Convert Pinocchio position to MuJoCo position
    
    Args:
        q: Pinocchio position vector (nq x 1)
        mj_model: MuJoCo model
    
    Returns:
        mj_qpos: MuJoCo position vector
    """
    mj_qpos = np.zeros(mj_model.nq)
    
    # Copy position
    mj_qpos[0:3] = q[0:3]
    
    # Convert quaternion (xyzw -> wxyz)
    mj_qpos[3] = q[6]  # w
    mj_qpos[4] = q[3]  # x
    mj_qpos[5] = q[4]  # y
    mj_qpos[6] = q[5]  # z
    
    # Copy joint positions
    mj_qpos[7:] = q[7:mj_model.nq]
    
    return mj_qpos


def pin_to_mj_velocity(v, mj_model):
    """
    Convert Pinocchio velocity to MuJoCo velocity
    
    Args:
        v: Pinocchio velocity vector (nv x 1)
        mj_model: MuJoCo model
    
    Returns:
        mj_qvel: MuJoCo velocity vector
    """
    mj_qvel = np.zeros(mj_model.nv)
    
    # Convert base velocity (swap angular and linear)
    mj_qvel[0:3] = v[3:6]  # Linear velocity
    mj_qvel[3:6] = v[0:3]  # Angular velocity
    
    # Copy joint velocities
    mj_qvel[6:] = v[6:mj_model.nv]
    
    return mj_qvel


def pin_to_mj_torque(pin_torques, mj_model):
    """
    Convert Pinocchio torques to MuJoCo control inputs
    
    Args:
        pin_torques: Pinocchio torques (excluding floating base)
        mj_model: MuJoCo model
    
    Returns:
        mj_ctrl: MuJoCo control vector
    """
    # MuJoCo control size is usually actuated joints only
    mj_ctrl = np.zeros(mj_model.nu)
    
    # Map actuated joint torques (exclude floating base)
    mj_ctrl[:] = pin_torques
    
    return mj_ctrl