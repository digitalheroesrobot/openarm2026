"""
OpenArm V1 MuJoCo viewer.

Purpose:
1. Load the official OpenArm V1 bimanual model.
2. Initialize the simulation.
3. Display the robot in the interactive MuJoCo viewer.
4. Verify that the simulation advances correctly.

Controls:
- Close the viewer window to exit.
"""

from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root
        / "simulation"
        / "vendor"
        / "openarm_mujoco"
        / "v1"
        / "scene_with_cup.xml"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    print(f"Loading model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("\n========== OpenArm V1 ==========")
    print(f"Bodies:     {model.nbody}")
    print(f"Joints:     {model.njnt}")
    print(f"Actuators:  {model.nu}")
    print(f"Timestep:   {model.opt.timestep}")
    print("================================")
    print("\nLaunching MuJoCo viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = data.time

            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep simulation close to real-time.
            elapsed = data.time - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                import time
                time.sleep(remaining)

    print("MuJoCo viewer closed.")


if __name__ == "__main__":
    main()
