# centroidal_dynamics/advanced_balance_control.py
"""
Advanced momentum-based balance controller for humanoid robots.

Implementation based on the paper:
"A momentum-based balance controller for humanoid robots on non-level and non-stationary ground"
by Sung-Hee Lee and Ambarish Goswami
"""

import numpy as np
import pinocchio as pin
import scipy.optimize as opt

# Constants
GRAVITY = np.array([0, -9.81, 0])
MU = 0.5  # Friction coefficient

class BalanceController:
    def __init__(self, robot_model, contact_frame_ids=None, cop_safety_margin=0.02):
        """
        Initialize the balance controller.
        
        Parameters:
        -----------
        robot_model: pin.Model
            Pinocchio model of the robot
        contact_frame_ids: list, optional
            IDs of the contact frames (feet)
        cop_safety_margin: float, optional
            Safety margin for CoP constraints
        """
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        
        # Set default contact frames if not provided
        if contact_frame_ids is None:
            # Try to find foot frames - this is a heuristic
            foot_names = ["right_foot", "left_foot", "r_foot", "l_foot", "RF", "LF"]
            contact_frame_ids = []
            for name in foot_names:
                if name in self.robot_model.frames:
                    contact_frame_ids.append(self.robot_model.getFrameId(name))
            
            if not contact_frame_ids:
                raise ValueError("No contact frames provided and no default frames found")
        
        self.contact_frame_ids = contact_frame_ids
        self.cop_safety_margin = cop_safety_margin
        
        # Foot dimensions - these should be adjusted for your robot
        self.foot_length = 0.2  # Length in x direction
        self.foot_width = 0.1   # Width in y direction
        
        # Store previous momentum for damping
        self.prev_angular_momentum = np.zeros(3)
        self.prev_linear_momentum = np.zeros(3)
        self.dt = 0.01
        
    def detect_contacts(self, q, threshold=0.01):
        """
        Detect which feet are in contact with the ground.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        threshold: float
            Height threshold for contact detection
            
        Returns:
        --------
        contact_ids: list
            IDs of the contact frames in contact
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        contact_ids = []
        for frame_id in self.contact_frame_ids:
            frame_pos = self.robot_data.oMf[frame_id].translation
            if frame_pos[1] < threshold:  # Assuming y is up
                contact_ids.append(frame_id)
                
        return contact_ids
    
    def get_support_polygon(self, q, contact_ids):
        """
        Get the support polygon vertices for the given contacts.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        contact_ids: list
            IDs of the contact frames in contact
            
        Returns:
        --------
        vertices: np.ndarray
            Vertices of the support polygon in world frame
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        all_vertices = []
        
        for frame_id in contact_ids:
            # Get the foot placement
            oMf = self.robot_data.oMf[frame_id]
            
            # Define corners of the foot in the foot frame
            half_length = self.foot_length / 2
            half_width = self.foot_width / 2
            
            foot_corners = np.array([
                [half_length, 0, half_width],
                [half_length, 0, -half_width],
                [-half_length, 0, -half_width],
                [-half_length, 0, half_width]
            ])
            
            # Transform corners to world frame
            for corner in foot_corners:
                all_vertices.append(oMf.act(corner))
                
        # For single support, just return the foot vertices
        if len(contact_ids) == 1:
            return np.array(all_vertices)
        
        # For double support, compute convex hull
        # For simplicity, we'll just use all vertices - 
        # in practice you'd compute the actual convex hull
        return np.array(all_vertices)
    
    def is_cop_inside_support(self, cop, support_polygon):
        """
        Check if the CoP is inside the support polygon.
        
        This is a simplified version that just checks if the CoP is inside 
        the bounding box of the support polygon with a safety margin.
        
        Parameters:
        -----------
        cop: array-like
            Center of Pressure in world frame
        support_polygon: np.ndarray
            Vertices of the support polygon
            
        Returns:
        --------
        is_inside: bool
            True if CoP is inside the support polygon
        """
        # Compute bounding box of support polygon
        min_x = np.min(support_polygon[:, 0]) + self.cop_safety_margin
        max_x = np.max(support_polygon[:, 0]) - self.cop_safety_margin
        min_z = np.min(support_polygon[:, 2]) + self.cop_safety_margin
        max_z = np.max(support_polygon[:, 2]) - self.cop_safety_margin
        
        # Check if CoP is inside
        return (min_x <= cop[0] <= max_x) and (min_z <= cop[2] <= max_z)
    
    def compute_cmm(self, q):
        """
        Compute the Centroidal Momentum Matrix.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
            
        Returns:
        --------
        AG: np.ndarray
            Centroidal Momentum Matrix
        """
        pin.ccrba(self.robot_model, self.robot_data, q, np.zeros(self.robot_model.nv))
        return self.robot_data.Ag
    
    def compute_cmm_derivative(self, q, v):
        """
        Compute the time derivative of the CMM multiplied by joint velocities.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
            
        Returns:
        --------
        dAG_v: np.ndarray
            Time derivative of CMM multiplied by velocities
        """
        # Compute using finite differences (simplified)
        delta = 1e-6
        AG = self.compute_cmm(q)
        
        # Approximate dAG_v with finite differences
        dAG_v = np.zeros(6)
        for i in range(v.shape[0]):
            if abs(v[i]) > 1e-10:
                q_delta = np.zeros_like(v)
                q_delta[i] = delta
                q_plus = pin.integrate(self.robot_model, q, q_delta)
                AG_plus = self.compute_cmm(q_plus)
                dAG_i = (AG_plus - AG) / delta
                dAG_v += dAG_i[:, i] * v[i]
        
        return dAG_v
    
    def compute_centroidal_momentum(self, q, v):
        """
        Compute the centroidal momentum.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
            
        Returns:
        --------
        kG: np.ndarray
            Angular momentum around CoM
        lG: np.ndarray
            Linear momentum
        """
        pin.ccrba(self.robot_model, self.robot_data, q, v)
        
        # Use the angular and linear attributes instead of slicing
        return np.array(self.robot_data.hg.angular), np.array(self.robot_data.hg.linear)
    
    def compute_desired_momentum_rate(self, q, v, com_des, vcom_des, ang_mom_des,
                                     Gamma_11, Gamma_12, Gamma_21, Gamma_22, Gamma_23):
        """
        Compute desired momentum rate with damping terms.
        
        Parameters:
        -----------
        q, v: array-like
            Current joint position and velocity
        com_des, vcom_des: array-like
            Desired CoM position and velocity
        ang_mom_des: array-like
            Desired angular momentum
        Gamma_11, Gamma_12, Gamma_21, Gamma_22, Gamma_23: array-like
            Gain matrices
            
        Returns:
        --------
        hG_dot_des: np.ndarray
            Desired centroidal momentum rate change
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q, v)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        # Get current states
        com = pin.centerOfMass(self.robot_model, self.robot_data, q)
        kG, lG = self.compute_centroidal_momentum(q, v)
        
        # Compute CoM velocity
        J_com = pin.jacobianCenterOfMass(self.robot_model, self.robot_data, q)
        vcom = J_com @ v
        
        # Total mass
        mass = sum(self.robot_model.inertias[i].mass for i in range(self.robot_model.njoints))
        
        # Compute desired linear momentum rate with damping
        l_dot_des = mass * (
            Gamma_11 @ (vcom_des - vcom) + 
            Gamma_12 @ (com_des - com) - 
            Gamma_23 @ ((lG - self.prev_linear_momentum) / self.dt)
        )
        
        # Compute desired angular momentum rate with damping
        k_dot_des = (
            Gamma_21 @ (ang_mom_des - kG) - 
            Gamma_22 @ ((kG - self.prev_angular_momentum) / self.dt)
        )
        
        # Update previous values
        self.prev_angular_momentum = kG.copy()
        self.prev_linear_momentum = lG.copy()
        
        # Combine
        hG_dot_des = np.concatenate([k_dot_des, l_dot_des])
        
        return hG_dot_des
    
    def compute_grf_cop_single_support(self, q, hG_dot_des, contact_id):
        """
        Compute the GRF and CoP for single support.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        hG_dot_des: array-like
            Desired momentum rate change
        contact_id: int
            ID of the contact frame
            
        Returns:
        --------
        grf: np.ndarray
            Ground reaction force
        cop: np.ndarray
            Center of pressure
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        # Get necessary physical parameters
        mass = sum(self.robot_model.inertias[i].mass for i in range(self.robot_model.njoints))
        com = pin.centerOfMass(self.robot_model, self.robot_data, q)
        
        # Split desired momentum rate
        k_dot_des = hG_dot_des[:3]
        l_dot_des = hG_dot_des[3:]
        
        # Compute GRF (Equation 12)
        grf = l_dot_des - mass * GRAVITY
        
        # Compute CoP (Equations 13-14)
        # Note: These equations are adapted for our coordinate system
        # where y is up, x is forward, and z is lateral
        # In the paper, Y is up, X is forward, Z is lateral
        # p_d,X = r_G,X - (f_d,X * r_G,Y - k_dot_d,Z) / (l_dot_d,Y - mg)
        # p_d,Z = r_G,Z - (f_d,Z * r_G,Y + k_dot_d,X) / (l_dot_d,Y - mg)
        
        denominator = l_dot_des[1] - mass * GRAVITY[1]
        
        # Avoid division by zero
        if abs(denominator) < 1e-10:
            denominator = 1e-10 * np.sign(denominator)
        
        cop_x = com[0] - (grf[0] * com[1] - k_dot_des[2]) / denominator
        cop_z = com[2] - (grf[2] * com[1] + k_dot_des[0]) / denominator
        
        # Construct CoP in world frame (y coordinate is on the ground)
        foot_pos = self.robot_data.oMf[contact_id].translation
        cop = np.array([cop_x, foot_pos[1], cop_z])
        
        return grf, cop
    
    def compute_grf_cop_double_support(self, q, hG_dot_des, contact_ids):
        """
        Compute GRFs and CoPs for double support.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        hG_dot_des: array-like
            Desired momentum rate change
        contact_ids: list
            IDs of the contact frames
            
        Returns:
        --------
        grfs: list
            Ground reaction forces for each foot
        cops: list
            Centers of pressure for each foot
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        # This is a simplified version that just splits the desired GRF equally
        # between the two feet. In practice, you would solve the optimization
        # problem described in Section 3.4.2 of the paper.
        
        # Get necessary physical parameters
        mass = sum(self.robot_model.inertias[i].mass for i in range(self.robot_model.njoints))
        com = pin.centerOfMass(self.robot_model, self.robot_data, q)
        
        # Split desired momentum rate
        k_dot_des = hG_dot_des[:3]
        l_dot_des = hG_dot_des[3:]
        
        # Compute total GRF
        total_grf = l_dot_des - mass * GRAVITY
        
        # Split GRF equally between feet
        grfs = [total_grf / len(contact_ids) for _ in contact_ids]
        
        # Compute CoP for each foot - for simplicity, place at center of foot
        cops = []
        for frame_id in contact_ids:
            foot_pos = self.robot_data.oMf[frame_id].translation
            cops.append(foot_pos)
            
        return grfs, cops
    
    def compute_momentum_from_grf_cop(self, q, grfs, cops, contact_ids):
        """
        Compute the momentum rate change from GRFs and CoPs.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        grfs: list
            Ground reaction forces
        cops: list
            Centers of pressure
        contact_ids: list
            IDs of the contact frames
            
        Returns:
        --------
        hG_dot: np.ndarray
            Momentum rate change
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        com = pin.centerOfMass(self.robot_model, self.robot_data, q)
        mass = sum(self.robot_model.inertias[i].mass for i in range(self.robot_model.njoints))
        
        # Compute linear momentum rate change (sum of all GRFs plus gravity)
        l_dot = mass * GRAVITY
        for grf in grfs:
            l_dot += grf
            
        # Compute angular momentum rate change
        k_dot = np.zeros(3)
        for grf, cop in zip(grfs, cops):
            # r × F contribution
            k_dot += np.cross(cop - com, grf)
            
        return np.concatenate([k_dot, l_dot])
    
    def compute_admissible_momentum_rate(self, q, v, hG_dot_des):
        """
        Compute admissible momentum rate change.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
        hG_dot_des: array-like
            Desired momentum rate change
            
        Returns:
        --------
        hG_dot_adm: np.ndarray
            Admissible momentum rate change
        """
        # Detect contacts
        contact_ids = self.detect_contacts(q)
        
        if not contact_ids:
            # No contacts - can't generate any reaction forces
            return np.zeros(6)
        
        # Compute support polygon
        support_polygon = self.get_support_polygon(q, contact_ids)
        
        # Compute GRF and CoP
        if len(contact_ids) == 1:
            # Single support
            grf, cop = self.compute_grf_cop_single_support(q, hG_dot_des, contact_ids[0])
            grfs, cops = [grf], [cop]
        else:
            # Double support
            grfs, cops = self.compute_grf_cop_double_support(q, hG_dot_des, contact_ids)
        
        # Check if CoP is inside support polygon
        is_admissible = True
        for cop in cops:
            if not self.is_cop_inside_support(cop, support_polygon):
                is_admissible = False
                break
        
        if is_admissible:
            # If admissible, return the desired momentum rate
            return hG_dot_des
        else:
            # If not admissible, prioritize linear momentum over angular momentum
            # This is a simplified version - in practice, you would adjust the
            # CoP to be inside the support polygon while maintaining the GRF
            
            # Preserve linear momentum rate
            l_dot_des = hG_dot_des[3:]
            
            # Compute modified CoPs that are inside the support polygon
            modified_cops = []
            for cop in cops:
                # Project CoP to the nearest point on the boundary of the support polygon
                # (simplified to just clamp to the bounding box)
                min_x = np.min(support_polygon[:, 0]) + self.cop_safety_margin
                max_x = np.max(support_polygon[:, 0]) - self.cop_safety_margin
                min_z = np.min(support_polygon[:, 2]) + self.cop_safety_margin
                max_z = np.max(support_polygon[:, 2]) - self.cop_safety_margin
                
                modified_cop = cop.copy()
                modified_cop[0] = np.clip(cop[0], min_x, max_x)
                modified_cop[2] = np.clip(cop[2], min_z, max_z)
                modified_cops.append(modified_cop)
            
            # Compute resulting momentum rate with modified CoPs
            hG_dot_adm = self.compute_momentum_from_grf_cop(q, grfs, modified_cops, contact_ids)
            
            # Ensure the linear momentum rate is preserved
            hG_dot_adm[3:] = l_dot_des
            
            return hG_dot_adm
        
    def compute_admissible_momentum_rate_double_support(self, q, v, hG_dot_des):
        """
        Compute admissible momentum rate change for double support based on 
        Lee & Goswami (2012), Section 3.4.2.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
        hG_dot_des: array-like
            Desired momentum rate change
            
        Returns:
        --------
        hG_dot_adm: np.ndarray
            Admissible momentum rate change
        """
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        
        # Get necessary physical parameters
        mass = sum(self.robot_model.inertias[i].mass for i in range(self.robot_model.njoints))
        com = pin.centerOfMass(self.robot_model, self.robot_data, q)
        g = np.array([0, -9.81, 0])  # Gravity vector
        
        # Get both foot positions
        r_r = self.robot_data.oMf[self.contact_frame_ids[0]].translation
        r_l = self.robot_data.oMf[self.contact_frame_ids[1]].translation
        
        # Split desired momentum rate
        k_dot_des = hG_dot_des[:3]
        l_dot_des = hG_dot_des[3:]
        
        # Compute total GRF
        total_grf = l_dot_des - mass * g
        
        # Implement equations 21-26: Determine foot GRFs
        # We'll set up the linear least-squares problem with non-negativity constraints
        
        # Each foot GRF is modeled using 4 basis vectors (approximating friction cone)
        # For simplicity, we'll define them as unit vectors along cardinal directions
        basis_vectors = np.array([
            [1, 0, 0],  # Forward
            [-1, 0, 0], # Backward
            [0, 0, 1],  # Left
            [0, 0, -1]  # Right
        ])
        
        # Add an upward component to each basis vector to create the friction cone
        friction_coef = 0.5  # Typical friction coefficient
        for i in range(4):
            # Normalize horizontal component
            horiz_norm = np.linalg.norm(basis_vectors[i])
            if horiz_norm > 0:
                basis_vectors[i] = basis_vectors[i] / horiz_norm
                # Add vertical component based on friction coefficient
                basis_vectors[i] = np.array([
                    basis_vectors[i][0],
                    friction_coef,  # Vertical component (y-axis is up)
                    basis_vectors[i][2]
                ])
                # Normalize again
                basis_vectors[i] = basis_vectors[i] / np.linalg.norm(basis_vectors[i])
        
        # Create the matrices for the least-squares problem (equation 23)
        # Matrix for linear momentum constraint: β_r β_l
        A_lin = np.zeros((3, 8))
        for i in range(4):
            A_lin[:, i] = basis_vectors[i]      # Right foot basis vectors
            A_lin[:, i+4] = basis_vectors[i]    # Left foot basis vectors
        
        # Matrix for angular momentum constraint: [r_r - r_G]× β_r, [r_l - r_G]× β_l
        A_ang = np.zeros((3, 8))
        for i in range(4):
            A_ang[:, i] = np.cross(r_r - com, basis_vectors[i])     # Right foot
            A_ang[:, i+4] = np.cross(r_l - com, basis_vectors[i])   # Left foot
        
        # Regularization term to minimize foot GRF magnitudes
        reg_param = 0.01
        A_reg = reg_param * np.eye(8)
        
        # Combine constraints (equation 24)
        A = np.vstack([A_lin, A_ang, A_reg])
        
        # Right-hand side vector (equation 25)
        b = np.concatenate([
            l_dot_des - mass * g,  # Linear momentum constraint
            k_dot_des,            # Angular momentum constraint
            np.zeros(8)           # Regularization
        ])
        
        # Solve the non-negative least-squares problem (equation 23)
        from scipy.optimize import nnls
        rho, _ = nnls(A, b)
        
        # Compute foot GRFs using the basis vectors
        f_r = np.zeros(3)
        f_l = np.zeros(3)
        
        for i in range(4):
            f_r += basis_vectors[i] * rho[i]
            f_l += basis_vectors[i] * rho[i+4]
        
        # Now implement equations 27-30: Determine foot CoPs
        # Compute required ankle torques
        k_dot_f = np.cross(r_r - com, f_r) + np.cross(r_l - com, f_l)
        k_dot_tau = k_dot_des - k_dot_f
        
        # Foot coordinate frames
        R_r = self.robot_data.oMf[self.contact_frame_ids[0]].rotation
        R_l = self.robot_data.oMf[self.contact_frame_ids[1]].rotation
        
        # Transform GRFs to foot frames
        f_r_local = R_r.T @ f_r
        f_l_local = R_l.T @ f_l
        
        # Foot dimensions
        foot_half_length = self.foot_length / 2
        foot_half_width = self.foot_width / 2
        
        # Compute foot CoPs
        # For a simple implementation, we'll compute CoPs that minimize ankle torques
        # In the foot frame, assuming x-forward, z-lateral, y-up
        d_r = np.zeros(3)
        d_l = np.zeros(3)
        
        # Set CoP at center of feet (simplified)
        # This could be optimized to better distribute the torques
        d_r[0] = 0  # x-direction (front/back)
        d_r[2] = 0  # z-direction (left/right)
        
        d_l[0] = 0
        d_l[2] = 0
        
        # Convert to world frame
        p_r = r_r + R_r @ d_r
        p_l = r_l + R_l @ d_l
        
        # Compute resulting ankle torques
        tau_r = np.cross(R_r @ d_r, f_r)
        tau_l = np.cross(R_l @ d_l, f_l)
        
        # Compute actual momentum rate from the computed GRFs and CoPs
        l_dot_actual = f_r + f_l + mass * g
        k_dot_actual = np.cross(p_r - com, f_r) + np.cross(p_l - com, f_l)
        
        # Return the admissible momentum rate change
        hG_dot_adm = np.concatenate([k_dot_actual, l_dot_actual])
        
        return hG_dot_adm
    
    def compute_joint_accelerations(self, q, v, hG_dot_adm):
        """
        Compute joint accelerations to realize the admissible momentum rate change.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
        hG_dot_adm: array-like
            Admissible momentum rate change
            
        Returns:
        --------
        ddq: np.ndarray
            Joint accelerations
        """
        # Compute the CMM
        AG = self.compute_cmm(q)
        
        # Compute the derivative term
        dAG_v = self.compute_cmm_derivative(q, v)
        
        # Solve the least-squares problem with regularization
        lambda_reg = 1e-4
        AG_reg = np.vstack([AG, lambda_reg * np.eye(self.robot_model.nv)])
        b_reg = np.hstack([hG_dot_adm - dAG_v, np.zeros(self.robot_model.nv)])
        
        ddq = np.linalg.lstsq(AG_reg, b_reg, rcond=None)[0]
        
        # Apply acceleration limits
        ddq_max = 50.0
        ddq = np.clip(ddq, -ddq_max, ddq_max)
        
        return ddq
    
    def update_state(self, q, v, ddq, dt):
        """
        Update the robot state using a more stable integration scheme.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
        ddq: array-like
            Joint accelerations
        dt: float
            Time step
            
        Returns:
        --------
        q_new: np.ndarray
            Updated joint configuration
        v_new: np.ndarray
            Updated joint velocities
        """
        # Use semi-implicit Euler integration
        v_new = np.clip(v + ddq * dt, -5, 5)
        q_new = pin.integrate(self.robot_model, q, v_new * dt)
        
        return q_new, v_new
    
    def compute_control_torques(self, q, v, ddq):
        """
        Compute joint torques to realize the desired joint accelerations.
        
        Parameters:
        -----------
        q: array-like
            Joint configuration vector
        v: array-like
            Joint velocity vector
        ddq: array-like
            Desired joint accelerations
            
        Returns:
        --------
        tau: np.ndarray
            Joint torques
        """
        return pin.rnea(self.robot_model, self.robot_data, q, v, ddq)