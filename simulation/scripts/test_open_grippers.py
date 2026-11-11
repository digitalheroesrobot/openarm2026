"""
OpenArm V1 - Open Gripper and Home Pose Validation

Objectives:
1. Load the OpenArm V1 scene with table and paper cup.
2. Restore the previously validated home arm configuration.
3. Open both grippers to 44 mm.
4. Check finger self-collisions.
5. Check robot/table/cup collisions.
6. Display the static configuration in MuJoCo.

Important:
- This script does not advance simulation physics.
- Joint positions are initialized directly for static inspection.
- Actuator targets are configured for the next dynamic test.
- No existing XML or robot model files are modified.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "vendor"
    / "openarm_mujoco"
    / "v1"
    / "scene_with_cup.xml"
)

# Validated arm home configuration from the previous IK test.
HOME = {
    "openarm_left_joint1": 0.064370,
    "openarm_left_joint2": -0.214334,
    "openarm_left_joint3": 0.015058,
    "openarm_left_joint4": 1.486495,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,

    "openarm_right_joint1": -0.064370,
    "openarm_right_joint2": 0.214334,
    "openarm_right_joint3": -0.015058,
    "openarm_right_joint4": 1.486495,
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
}

FINGER_JOINTS = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]

FINGER_ACTUATORS = [
    "left_finger1_ctrl",
    "right_finger1_ctrl",
]

# Joint coordinates are meters.
OPEN_POSITION = 0.044

# Viewer refresh rate.
VIEWER_FPS = 60


# =========================================================
# MODEL HELPERS
# =========================================================

def get_id(model, object_type, name):
    """Return the MuJoCo object ID or raise an error."""

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")

    return object_id


def get_name(model, object_type, object_id):
    """Return a readable MuJoCo object name."""

    name = mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )

    return name if name else f"unnamed_{object_id}"


# =========================================================
# CONFIGURATION INITIALIZATION
# =========================================================

def initialize_home_pose(model, data):

    print("\n========== ARM HOME CONFIGURATION ==========")

    for name, value in HOME.items():

        joint_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_id = model.jnt_qposadr[joint_id]

        lower, upper = model.jnt_range[joint_id]

        if not lower <= value <= upper:
            raise ValueError(
                f"{name}: {value} outside [{lower}, {upper}]"
            )

        data.qpos[qpos_id] = value

        print(f"{name}: {value:.6f} rad")


def initialize_open_grippers(model, data):

    print("\n========== GRIPPER CONFIGURATION ==========")

    for name in FINGER_JOINTS:

        joint_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_id = model.jnt_qposadr[joint_id]

        lower, upper = model.jnt_range[joint_id]

        if not lower <= OPEN_POSITION <= upper:
            raise ValueError(
                f"{name}: opening position outside joint limits"
            )

        data.qpos[qpos_id] = OPEN_POSITION

        print(
            f"{name}: "
            f"{data.qpos[qpos_id]:.4f} m"
        )

    # Configure position actuator targets.
    for name in FINGER_ACTUATORS:

        actuator_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if model.actuator_ctrllimited[actuator_id]:

            lower, upper = model.actuator_ctrlrange[actuator_id]

            if not lower <= OPEN_POSITION <= upper:
                raise ValueError(
                    f"{name}: target outside actuator limits"
                )

        data.ctrl[actuator_id] = OPEN_POSITION

        print(
            f"{name} target: "
            f"{data.ctrl[actuator_id]:.4f} m"
        )


# =========================================================
# END-EFFECTOR DIAGNOSTICS
# =========================================================

def print_end_effector_positions(model, data):

    print("\n========== END-EFFECTOR POSITIONS ==========")

    for side in ("left", "right"):

        site_name = f"{side}_ee_control_point"

        site_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        position = data.site_xpos[site_id]

        print(
            f"{side.upper()}: "
            f"{np.array2string(position, precision=5)}"
        )


# =========================================================
# COLLISION DIAGNOSTICS
# =========================================================

def inspect_contacts(model, data):

    print("\n========== COLLISION CONTACTS ==========")

    finger_contacts = 0
    table_cup_contacts = 0
    negative_distance_contacts = 0

    for index in range(data.ncon):

        contact = data.contact[index]

        name1 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]

        body1 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id,
        )

        body2 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id,
        )

        names = (
            name1.lower(),
            name2.lower(),
            body1.lower(),
            body2.lower(),
        )

        is_finger_pair = (
            "finger" in name1.lower()
            and "finger" in name2.lower()
        )

        involves_table_or_cup = any(
            "table" in name or "cup" in name
            for name in names
        )

        if is_finger_pair:
            finger_contacts += 1

        if involves_table_or_cup:
            table_cup_contacts += 1

        if contact.dist < 0:
            negative_distance_contacts += 1

        print(
            f"{name1} ({body1}) <-> "
            f"{name2} ({body2}) | "
            f"distance: {contact.dist * 1000:.3f} mm"
        )

    print("\n========== COLLISION SUMMARY ==========")

    print("Total contacts:", data.ncon)
    print("Finger-to-finger contacts:", finger_contacts)
    print("Table/cup-related contacts:", table_cup_contacts)
    print("Negative-distance contacts:", negative_distance_contacts)

    if finger_contacts == 0 and table_cup_contacts == 0:
        print("\nSTATIC GEOMETRY TEST: PASS")
    else:
        print("\nSTATIC GEOMETRY TEST: CHECK REQUIRED")

    if negative_distance_contacts:
        print("WARNING: Negative contact distances detected.")

    print("=======================================")


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n==========================================")
    print("       OPENARM V1 GRIPPER TEST")
    print("==========================================")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Scene XML not found: {MODEL_PATH}"
        )

    print("\nLoading:", MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    initialize_home_pose(model, data)

    initialize_open_grippers(model, data)

    # Forward kinematics and collision detection.
    mujoco.mj_forward(model, data)

    print_end_effector_positions(model, data)

    inspect_contacts(model, data)

    print("\n========== MODEL INFORMATION ==========")
    print("Bodies:", model.nbody)
    print("Joints:", model.njnt)
    print("Actuators:", model.nu)
    print("Simulation timestep:", model.opt.timestep)

    print("\nOpening MuJoCo viewer...")
    print("The robot will remain in a static configuration.")
    print("Close the MuJoCo window to exit.")

    # Passive viewer: no simulation stepping.
    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("Viewer running successfully.")

        while viewer.is_running():

            viewer.sync()

            time.sleep(1.0 / VIEWER_FPS)

    print("\nMuJoCo viewer closed.")
    print("Gripper test finished.")


if __name__ == "__main__":
    main()
