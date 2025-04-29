#!/usr/bin/env python3
import os
import time
import mujoco
import mujoco.viewer

def main():
    # Load model path
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "unitree_g1.xml"
    )
    print(f"Loading model from: {model_path}")

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Raise the robot 2 meters off the ground (identity quaternion: [1, 0, 0, 0])
        data.qpos[2] = 2.0
        data.qpos[3] = 1.0  # w
        data.qpos[4] = 0.0  # x
        data.qpos[5] = 0.0  # y
        data.qpos[6] = 0.0  # z

        print(f"Model loaded: {model.njnt} joints, {model.nu} actuators")

        # Launch viewer with sliders
        with mujoco.viewer.launch(model, data) as viewer:
            print("Use the sliders in the viewer to test joint actuators.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)

    except Exception as e:
        print(f"Error: {e}")
        print("\nSearching for model file...")
        for root, _, files in os.walk(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
            for file in files:
                if file.endswith('.xml') and 'unitree' in file.lower():
                    print(f"Found: {os.path.join(root, file)}")

if __name__ == "__main__":
    main()
