#!/usr/bin/env python3
# knee_joint_finder.py

import os
import mujoco
import numpy as np

def main():
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "models", "unitree_g1.xml")
    print(f"Attempting to load model from: {model_path}")
    
    try:
        mj_model = mujoco.MjModel.from_xml_path(model_path)
        mj_data = mujoco.MjData(mj_model)
        print(f"Successfully loaded model with {mj_model.nq} position DOFs and {mj_model.nv} velocity DOFs")
        
        # Print basic joint information
        print("\nJoint information:")
        for i in range(mj_model.njnt):
            name = mj_model.joint(i).name
            qpos_idx = mj_model.jnt(i).qposadr
            type_int = mj_model.jnt(i).type
            
            print(f"{i}: {name:25s} | qpos index: {qpos_idx} | type: {type_int}")
        
        # Find knee, hip, and ankle joints
        knee_indices = []
        hip_pitch_indices = []
        ankle_pitch_indices = []
        
        for i in range(mj_model.njnt):
            name = mj_model.joint(i).name
            qpos_idx = mj_model.jnt(i).qposadr
            
            if "knee" in name.lower():
                knee_indices.append((name, qpos_idx))
            elif "hip_pitch" in name.lower():
                hip_pitch_indices.append((name, qpos_idx))
            elif "ankle_pitch" in name.lower():
                ankle_pitch_indices.append((name, qpos_idx))
        
        print("\nKnee joints:")
        for name, idx in knee_indices:
            print(f"{name:25s} | qpos index: {idx}")
        
        print("\nHip pitch joints:")
        for name, idx in hip_pitch_indices:
            print(f"{name:25s} | qpos index: {idx}")
        
        print("\nAnkle pitch joints:")
        for name, idx in ankle_pitch_indices:
            print(f"{name:25s} | qpos index: {idx}")
            
        # Test setting knee angles to 45 degrees
        print("\nSetting knee angles to 45 degrees...")
        angle_rad = 45 * np.pi / 180.0
        
        for name, idx in knee_indices:
            # Save original value
            original_angle = mj_data.qpos[idx]
            print(f"Original {name} angle: {original_angle * 180.0 / np.pi:.2f} degrees")
            
            # Set new value
            mj_data.qpos[idx] = angle_rad
        
        # Forward kinematics to update positions
        mujoco.mj_forward(mj_model, mj_data)
        
        # Verify angles were set correctly
        for name, idx in knee_indices:
            print(f"New {name} angle: {mj_data.qpos[idx] * 180.0 / np.pi:.2f} degrees")
        
        # Also print actuator indices
        print("\nActuator information:")
        for i in range(mj_model.nu):
            name = mj_model.actuator(i).name
            print(f"Actuator {i}: {name}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()