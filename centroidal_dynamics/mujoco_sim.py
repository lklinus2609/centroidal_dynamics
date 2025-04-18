# centroidal_dynamics/mujoco_sim.py
import numpy as np
import mujoco
from mujoco import viewer
import pinocchio as pin
import time

class MujocoSimulator:
    def __init__(self, mujoco_model_path, urdf_model_path=None):
        """
        Initialize MuJoCo simulator
        
        Parameters:
        -----------
        mujoco_model_path: str
            Path to the MuJoCo XML model file
        urdf_model_path: str, optional
            Path to the URDF model file for Pinocchio
            If None, will try to use the mujoco_model_path for Pinocchio if it's a URDF
        """
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize the viewer
        self.viewer = None
        
        # Load the Pinocchio model (for our balance controller)
        if urdf_model_path is None:
            # Try to use the same file if it's a URDF
            try:
                self.pin_model = pin.buildModelFromUrdf(mujoco_model_path)
                self.pin_data = self.pin_model.createData()
            except ValueError:
                print("Warning: Could not load MuJoCo XML file as URDF for Pinocchio.")
                print("Please specify a separate URDF file path for Pinocchio.")
                raise
        else:
            # Use the provided URDF file
            self.pin_model = pin.buildModelFromUrdf(urdf_model_path)
            self.pin_data = self.pin_model.createData()
        
        # Initialize state
        self.reset()
        
        # Initialize foot and ankle IDs
        self.init_foot_ankle_ids()
    
    def init_foot_ankle_ids(self):
        """
        Initialize foot and ankle IDs by searching for relevant names in the model
        """
        # Initialize IDs
        self.right_foot_id = None
        self.left_foot_id = None
        self.right_ankle_joints = []
        self.left_ankle_joints = []
        
        # Search for foot bodies (ankle roll links are the foot links in this model)
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            if body_name == "right_ankle_roll_link":
                self.right_foot_id = i
                print(f"Found right foot body: {body_name} (ID: {i})")
            elif body_name == "left_ankle_roll_link":
                self.left_foot_id = i
                print(f"Found left foot body: {body_name} (ID: {i})")
        
        # Search for ankle joints
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            if joint_name == "right_ankle_pitch_joint" or joint_name == "right_ankle_roll_joint":
                self.right_ankle_joints.append(i)
                print(f"Found right ankle joint: {joint_name} (ID: {i})")
            elif joint_name == "left_ankle_pitch_joint" or joint_name == "left_ankle_roll_joint":
                self.left_ankle_joints.append(i)
                print(f"Found left ankle joint: {joint_name} (ID: {i})")
        
        # If we couldn't find the specific IDs, provide a warning
        if self.right_foot_id is None or self.left_foot_id is None:
            print("WARNING: Could not find foot bodies. GRF and CoP data will not be accurate.")
        
        if not self.right_ankle_joints or not self.left_ankle_joints:
            print("WARNING: Could not find ankle joints. Ankle torque data will not be accurate.")
    
    def reset(self):
        """Reset the simulation to the initial state"""
        mujoco.mj_resetData(self.model, self.data)
        
    def get_state(self):
        """
        Get the current state of the robot in MuJoCo format
        
        Returns:
        --------
        q: np.ndarray
            Joint positions
        v: np.ndarray
            Joint velocities
        """
        q = self.data.qpos.copy()
        v = self.data.qvel.copy()
        
        return q, v
    
    def set_state(self, q, v):
        """
        Set the state of the robot in MuJoCo format
        
        Parameters:
        -----------
        q: np.ndarray
            Joint positions
        v: np.ndarray
            Joint velocities
        """
        # Set joint positions and velocities
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)

    def get_pin_state(self):
        """Get current state in Pinocchio format"""
        # Get MuJoCo state
        mj_q, mj_v = self.get_state()
        
        # For debug purposes, print dimensions
        print(f"MuJoCo state dimensions: q({len(mj_q)}), v({len(mj_v)})")
        print(f"Pinocchio state dimensions: nq({self.pin_model.nq}), nv({self.pin_model.nv})")
        
        # For now, create a simple mapping - this needs to be customized based on your specific models
        q_pin = np.zeros(self.pin_model.nq)
        v_pin = np.zeros(self.pin_model.nv)
        
        # Map base pose (first 7 elements in Pinocchio format)
        if self.pin_model.nq >= 7 and len(mj_q) >= 7:
            # Position - first 3 elements are typically the same
            q_pin[0:3] = mj_q[0:3]
            
            # Orientation - might need conversion between quaternion formats
            if len(mj_q) >= 7:  # If MuJoCo has quaternion
                q_pin[3:7] = mj_q[3:7]  # Simple direct copy, might need reordering
        
        # Map joint positions - depends on your specific model structure
        min_joint_dofs = min(self.pin_model.nq - 7, len(mj_q) - 7)
        if min_joint_dofs > 0:
            q_pin[7:7+min_joint_dofs] = mj_q[7:7+min_joint_dofs]
        
        # Map velocities similarly
        if self.pin_model.nv >= 6 and len(mj_v) >= 6:
            v_pin[0:6] = mj_v[0:6]  # Base velocity
        
        min_joint_vels = min(self.pin_model.nv - 6, len(mj_v) - 6)
        if min_joint_vels > 0:
            v_pin[6:6+min_joint_vels] = mj_v[6:6+min_joint_vels]
        
        return q_pin, v_pin

    def apply_pin_control(self, pin_tau):
        """
        Apply Pinocchio-computed torques to MuJoCo
        
        Parameters:
        -----------
        pin_tau: np.ndarray
            Joint torques computed by Pinocchio
        """
        # For now, just create zero controls for MuJoCo
        # In a real implementation, you would map between Pinocchio and MuJoCo control
        mj_ctrl = np.zeros(self.model.nu)
        
        # Apply control
        self.data.ctrl[:] = mj_ctrl
    
    def get_contact_data(self):
        """
        Get ground reaction forces and center of pressure for both feet
        
        Returns:
        --------
        foot_forces: np.ndarray (2, 3)
            Ground reaction forces for [right_foot, left_foot] in [x, y, z]
        foot_cops: np.ndarray (2, 2)
            Center of pressure for [right_foot, left_foot] in [x, z]
        """
        # Initialize output arrays
        foot_forces = np.zeros((2, 3))  # [right/left, xyz]
        foot_cops = np.zeros((2, 2))    # [right/left, xz]
        foot_contact_count = np.zeros(2, dtype=int)  # Count of contacts for each foot
        
        # Get contact forces from MuJoCo
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if this contact involves one of the feet
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            
            foot_index = -1
            if body1_id == self.right_foot_id or body2_id == self.right_foot_id:
                foot_index = 0  # Right foot
            elif body1_id == self.left_foot_id or body2_id == self.left_foot_id:
                foot_index = 1  # Left foot
                
            if foot_index >= 0:
                # Get contact force in world frame
                contact_force = np.zeros(6)  # [fx, fy, fz, mx, my, mz]
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                
                # Add to total force for this foot
                foot_forces[foot_index] += contact_force[0:3]
                
                # Calculate contribution to CoP
                if abs(contact_force[2]) > 1e-6:  # Only if there's vertical force
                    foot_contact_count[foot_index] += 1
                    
                    # Get contact position
                    contact_pos = contact.pos
                    
                    # Update CoP (weighted by vertical force)
                    weight = abs(contact_force[2])
                    cop_contribution = np.array([contact_pos[0], contact_pos[2]]) * weight
                    foot_cops[foot_index] += cop_contribution
        
        # Normalize CoP by total vertical force
        for i in range(2):
            if foot_contact_count[i] > 0 and abs(foot_forces[i, 2]) > 1e-6:
                foot_cops[i] /= abs(foot_forces[i, 2])
                
        return foot_forces, foot_cops

    def get_ankle_torques(self):
        """
        Get ankle joint torques
        
        Returns:
        --------
        ankle_torques: np.ndarray (2, 3)
            Ankle torques for [right_ankle, left_ankle] in [x, y, z]
            For this robot, only the first two dimensions are used (pitch, roll)
        """
        # Initialize output array
        ankle_torques = np.zeros((2, 3))  # [right/left, xyz]
        
        # Get torques from MuJoCo actuator forces
        for i, joint_id in enumerate(self.right_ankle_joints):
            if i < 2:  # We only have pitch and roll
                ankle_torques[0, i] = self.data.qfrc_actuator[joint_id]
                
        for i, joint_id in enumerate(self.left_ankle_joints):
            if i < 2:  # We only have pitch and roll
                ankle_torques[1, i] = self.data.qfrc_actuator[joint_id]
                
        return ankle_torques
    
    def step(self, tau=None, external_force=None, dt=None):
        """
        Step the simulation forward
        
        Parameters:
        -----------
        tau: np.ndarray, optional
            Joint torques
        external_force: dict, optional
            External forces to apply
            Format: {body_id: [fx, fy, fz, mx, my, mz]}
        dt: float, optional
            Time step
            
        Returns:
        --------
        q: np.ndarray
            New joint positions
        v: np.ndarray
            New joint velocities
        """
        # Apply control
        if tau is not None:
            self.data.ctrl[:] = tau
        
        # Apply external forces
        if external_force is not None:
            for body_id, force in external_force.items():
                self.data.xfrc_applied[body_id, :] = force
        
        # Step the simulation
        if dt is not None:
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data)
        
        # Return the new state
        return self.get_state()
    
    def render(self):
        """Render the current state of the simulation"""
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
    
    def close(self):
        """Close the viewer"""
        if self.viewer is not None:
            self.viewer.close()

    def print_model_info(self):
        """Print information about the MuJoCo model"""
        print("=== MuJoCo Model Information ===")
        print(f"Number of bodies: {self.model.nbody}")
        print("Body IDs and names:")
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            print(f"  {i}: {body_name}")
        
        print("\nNumber of joints: {self.model.njnt}")
        print("Joint IDs and names:")
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            print(f"  {i}: {joint_name}")