
#!/usr/bin/env python3

"""
OPENARM V1 — HOME GRASP GEOMETRY INSPECTION

Purpose:
    Inspect actual gripper geometry at the validated HOME pose.

Checks:
    1. Set both arms to the validated HOME configuration.
    2. Open both grippers to 44 mm.
    3. Inspect left/right hand TCP positions and orientations.
    4. Inspect end-effector control-point positions.
    5. Inspect finger collision geometry positions.
    6. Calculate world-space finger sliding axes.
    7. Inspect cup position and tabletop height.
    8. Check static collisions.

This is a geometry diagnostic only.
No trajectory execution or XML modification.
"""

from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

XML_PATH = (
    ROOT
    / "simulation/vendor/openarm_mujoco/v1/scene_with_cup.xml"
)

FINGER_OPEN = 0.044

JOINT_NAMES = (
    [f"openarm_left_joint{i}" for i in range(1, 8)]
    + [f"openarm_right_joint{i}" for i in range(1, 8)]
)

HOME = np.array([
     0.064370, -0.214334,  0.015058, 1.486495, 0, 0, 0,
    -0.064370,  0.214334, -0.015058, 1.486495, 0, 0, 0,
])

FINGER_NAMES = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]

TCP_BODY_NAMES = [
    "openarm_left_hand_tcp",
    "openarm_right_hand_tcp",
]

EE_SITE_NAMES = [
    "left_ee_control_point",
    "right_ee_control_point",
]

FINGER_GEOM_NAMES = [
    "openarm_left_right_finger_collision",
    "openarm_left_left_finger_collision",
    "openarm_right_right_finger_collision",
    "openarm_right_left_finger_collision",
]


# ============================================================
# HELPERS
# ============================================================

def object_id(model, object_type, name):

    result = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if result < 0:
        raise RuntimeError(
            f"MuJoCo object not found: {name}"
        )

    return result


def section(title):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_vector(label, value):

    print(
        f"{label}: "
        + np.array2string(
            np.asarray(value),
            precision=6,
            suppress_small=True,
        )
    )


def normalize(vector):

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector

    return vector / norm


# ============================================================
# MAIN
# ============================================================

def main():

    section("OPENARM V1 — HOME GRASP GEOMETRY")

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    print("Scene:", XML_PATH)

    # --------------------------------------------------------
    # 1. Set validated HOME configuration
    # --------------------------------------------------------

    data.qpos[:] = model.qpos0

    arm_qpos = []

    for name in JOINT_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        arm_qpos.append(
            int(model.jnt_qposadr[jid])
        )

    for address, value in zip(
        arm_qpos,
        HOME,
    ):

        data.qpos[address] = value

    # --------------------------------------------------------
    # 2. Open both grippers
    # --------------------------------------------------------

    finger_qpos = []

    for name in FINGER_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        address = int(model.jnt_qposadr[jid])

        finger_qpos.append(address)

        data.qpos[address] = FINGER_OPEN

    finger_actuator_names = [
        "left_finger1_ctrl",
        "right_finger1_ctrl",
    ]

    for name in finger_actuator_names:

        aid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        data.ctrl[aid] = FINGER_OPEN

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # 3. TCP positions and orientations
    # --------------------------------------------------------

    section("1. HAND TCP FRAMES")

    for name in TCP_BODY_NAMES:

        bid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        position = data.xpos[bid]

        rotation = data.xmat[bid].reshape(3, 3)

        print(f"\n{name}")

        print_vector("World position (m)", position)

        print("World rotation matrix:")

        print(
            np.array2string(
                rotation,
                precision=5,
                suppress_small=True,
            )
        )

        print_vector(
            "Local X axis in world",
            rotation[:, 0],
        )

        print_vector(
            "Local Y axis in world",
            rotation[:, 1],
        )

        print_vector(
            "Local Z axis in world",
            rotation[:, 2],
        )

    # --------------------------------------------------------
    # 4. End-effector control points
    # --------------------------------------------------------

    section("2. END-EFFECTOR CONTROL POINTS")

    for name in EE_SITE_NAMES:

        sid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        print(f"\n{name}")

        print_vector(
            "World position (m)",
            data.site_xpos[sid],
        )

        rotation = data.site_xmat[sid].reshape(3, 3)

        print("World rotation matrix:")

        print(
            np.array2string(
                rotation,
                precision=5,
                suppress_small=True,
            )
        )

    # --------------------------------------------------------
    # 5. Finger positions and sliding axes
    # --------------------------------------------------------

    section("3. FINGER JOINTS AND SLIDING AXES")

    for name in FINGER_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        body_id = int(model.jnt_bodyid[jid])

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        body_rotation = data.xmat[
            body_id
        ].reshape(3, 3)

        local_axis = model.jnt_axis[jid]

        world_axis = normalize(
            body_rotation @ local_axis
        )

        print(f"\n{name}")

        print("Parent body:", body_name)

        print(
            "Joint position:",
            f"{data.qpos[model.jnt_qposadr[jid]] * 1000:.2f} mm",
        )

        print_vector(
            "Local sliding axis",
            local_axis,
        )

        print_vector(
            "World sliding axis",
            world_axis,
        )

    # --------------------------------------------------------
    # 6. Finger collision geometry
    # --------------------------------------------------------

    section("4. FINGER COLLISION GEOMETRY")

    for name in FINGER_GEOM_NAMES:

        gid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        print(f"\n{name}")

        print_vector(
            "World position (m)",
            data.geom_xpos[gid],
        )

        print_vector(
            "Geom size",
            model.geom_size[gid],
        )

        print(
            "World rotation matrix:"
        )

        print(
            np.array2string(
                data.geom_xmat[gid].reshape(3, 3),
                precision=5,
                suppress_small=True,
            )
        )

    # --------------------------------------------------------
    # 7. Cup position and table height
    # --------------------------------------------------------

    section("5. CUP AND TABLE REFERENCE")

    cup_body = object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cup",
    )

    print_vector(
        "Cup body position (m)",
        data.xpos[cup_body],
    )

    cup_geom = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "cup",
    )

    print_vector(
        "Cup geometry center (m)",
        data.geom_xpos[cup_geom],
    )

    table_geom = object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    table_top_z = (
        data.geom_xpos[table_geom][2]
        + model.geom_size[table_geom][2]
    )

    print(
        f"Tabletop surface height: {table_top_z:.6f} m"
    )

    print(
        f"Cup center XY: "
        f"{data.xpos[cup_body][0]:.6f}, "
        f"{data.xpos[cup_body][1]:.6f}"
    )

    # --------------------------------------------------------
    # 8. Static collision inspection
    # --------------------------------------------------------

    section("6. STATIC COLLISION INSPECTION")

    print("Total contacts:", data.ncon)

    for i in range(data.ncon):

        contact = data.contact[i]

        name1 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        print(
            f"  {name1} <-> {name2} | "
            f"{contact.dist * 1000:.3f} mm"
        )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    section("7. SUMMARY")

    print("Arm configuration: VALIDATED HOME")
    print("Gripper command: 44 mm")
    print("Cup position:", np.round(data.xpos[cup_body], 6))

    print("\nGeometry inspection completed.")
    print("No robot movement or XML modification performed.")


if __name__ == "__main__":
    main()
