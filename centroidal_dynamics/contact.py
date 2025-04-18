# centroidal_dynamics/contact.py
import numpy as np
import pinocchio as pin

def compute_contact_jacobian(robot_model, robot_data, q, frame_id):
    """
    Compute the contact Jacobian for a given frame.
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
    frame_id: int
        Frame ID for the contact point
        
    Returns:
    --------
    J: np.ndarray
        Contact Jacobian
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Compute the contact Jacobian
    J = pin.computeFrameJacobian(robot_model, robot_data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    return J

def compute_support_polygon(robot_model, robot_data, q, contact_frames):
    """
    Compute the support polygon for the given contact configuration.
    
    Parameters:
    -----------
    robot_model: pin.Model
        Pinocchio model
    robot_data: pin.Data
        Pinocchio data
    q: array-like
        Joint configuration vector
    contact_frames: list
        List of frame IDs in contact with the ground
        
    Returns:
    --------
    support_polygon: np.ndarray
        Vertices of the support polygon (N x 3 array)
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Get the positions of the contact frames
    vertices = []
    for frame_id in contact_frames:
        position = robot_data.oMf[frame_id].translation
        vertices.append(position)
    
    # For multiple contact points, we would compute the convex hull
    # This is a simplified implementation
    return np.array(vertices)

def compute_cop(robot_model, robot_data, q, v, ddq, contact_frames, external_forces=None):
    """
    Compute the Center of Pressure (CoP) for the given contact configuration.
    
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
    ddq: array-like
        Joint acceleration vector
    contact_frames: list
        List of frame IDs in contact with the ground
    external_forces: list, optional
        External forces applied to the robot
        
    Returns:
    --------
    cop: np.ndarray
        Center of Pressure position
    """
    # Update kinematics
    pin.forwardKinematics(robot_model, robot_data, q, v)
    pin.updateFramePlacements(robot_model, robot_data)
    
    # Compute the ground reaction forces
    # This is a simplified implementation
    # In a real implementation, we'd need to compute the contact forces
    # based on the dynamics and the contact constraints
    
    # For now, we'll use the center of the support polygon as the CoP
    support_polygon = compute_support_polygon(robot_model, robot_data, q, contact_frames)
    cop = np.mean(support_polygon, axis=0)
    
    # TODO: Implement proper CoP calculation based on contact forces
    
    return cop

def check_cop_stability(cop, support_polygon, safety_margin=0.02):
    """
    Check if the CoP is within the support polygon with a safety margin.
    
    Parameters:
    -----------
    cop: array-like
        Center of Pressure position
    support_polygon: array-like
        Vertices of the support polygon
    safety_margin: float
        Safety margin distance from the boundaries
        
    Returns:
    --------
    is_stable: bool
        True if the CoP is within the stable region
    distance: float
        Distance to the nearest boundary (negative if outside)
    """
    # This is a simplified implementation
    # In a real implementation, we'd need to check if the CoP is
    # within the convex hull of the support polygon
    
    # For now, we'll just check if the CoP is within a rectangle
    # defined by the min/max coordinates of the support polygon
    
    min_coords = np.min(support_polygon, axis=0)
    max_coords = np.max(support_polygon, axis=0)
    
    # Apply safety margin
    min_coords += safety_margin
    max_coords -= safety_margin
    
    # Check if CoP is within the rectangle
    is_within_x = min_coords[0] <= cop[0] <= max_coords[0]
    is_within_z = min_coords[2] <= cop[2] <= max_coords[2]
    is_stable = is_within_x and is_within_z
    
    # Compute distance to nearest boundary
    dist_to_min_x = cop[0] - min_coords[0]
    dist_to_max_x = max_coords[0] - cop[0]
    dist_to_min_z = cop[2] - min_coords[2]
    dist_to_max_z = max_coords[2] - cop[2]
    
    distance = min(dist_to_min_x, dist_to_max_x, dist_to_min_z, dist_to_max_z)
    
    return is_stable, distance