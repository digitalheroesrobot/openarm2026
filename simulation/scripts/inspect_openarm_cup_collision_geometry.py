#!/usr/bin/env python3
"""
OPENARM V1 — CUP PICKUP COLLISION GEOMETRY INSPECTION

Purpose
-------
Inspect the actual MuJoCo collision geometry responsible for the failed
cup grasp orientation scan.

Configurations:
    1. Validated HOME
    2. Candidate A — original HOME orientation, 355 mm grasp center

Reports:
    - Geometry type and dimensions
    - Parent body
    - Local position and orientation
    - World position and orientation
    - Mesh asset information, where applicable
    - Static contact pairs and penetration distances

No robot movement.
No XML modification.
No trajectory execution.

Run:
    .venv/bin/python simulation/scripts/inspect_openarm_cup_collision_geometry.py
"""

from pathlib import Path

import mujoco
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENE_XML = (
    PROJECT_ROOT
    / "simulation/vendor/openarm_mujoco/v1/scene_with_cup.xml"
)

LEFT_JOINTS = [
    f"openarm_left_joint{i}" for i in range(1, 8)
]

RIGHT_JOINTS = [
    f"openarm_right_joint{i}" for i in range(1, 8)
]

FINGER_JOINTS = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]

HOME = np.array([
     0.064370, -0.214334,  0.015058,
     1.486495,  0.000000,  0.000000,  0.000000,

    -0.064370,  0.214334, -0.015058,
     1.486495,  0.000000,  0.000000,  0.000000,
])

# Candidate A solution from the previous IK test, in degrees.
CANDIDATE_A_DEGREES = np.array([
    20.440,
    10.000,
    30.404,
    67.853,
    11.528,
    37.824,
   -24.119,
])

GRIPPER_OPENING = 0.044

TARGET_GEOMS = [
    "openarm_body_link0_collision",
    "openarm_left_link5_collision",
    "openarm_left_hand_collision",
    "openarm_left_right_finger_collision",
    "openarm_left_left_finger_collision",
    "cup",
]


# ============================================================================
# HELPERS
# ============================================================================

def get_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")

    return object_id


def joint_qpos_address(model, name):
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def object_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name else f"unnamed_{object_id}"


def geom_type_name(geom_type):
    try:
        return mujoco.mjtGeom(geom_type).name
    except ValueError:
        return str(geom_type)


def rotation_matrix_string(matrix):
    return np.array2string(
        matrix,
        precision=5,
        suppress_small=True,
    )


# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

def set_home(model, data):

    data.qpos[:] = model.qpos0

    for i, name in enumerate(LEFT_JOINTS):
        data.qpos[joint_qpos_address(model, name)] = HOME[i]

    for i, name in enumerate(RIGHT_JOINTS):
        data.qpos[joint_qpos_address(model, name)] = HOME[i + 7]

    for name in FINGER_JOINTS:
        data.qpos[joint_qpos_address(model, name)] = GRIPPER_OPENING

    mujoco.mj_forward(model, data)


def set_candidate_a(model, data):

    set_home(model, data)

    for i, name in enumerate(LEFT_JOINTS):
        address = joint_qpos_address(model, name)
        data.qpos[address] = np.radians(CANDIDATE_A_DEGREES[i])

    mujoco.mj_forward(model, data)


# ============================================================================
# GEOMETRY INSPECTION
# ============================================================================

def inspect_geom(model, data, geom_name):

    gid = get_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    body_id = int(model.geom_bodyid[gid])

    body_name = object_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    geom_type = geom_type_name(model.geom_type[gid])

    local_position = model.geom_pos[gid].copy()
    local_quaternion = model.geom_quat[gid].copy()

    world_position = data.geom_xpos[gid].copy()
    world_rotation = data.geom_xmat[gid].reshape(3, 3).copy()

    print(f"\nGEOMETRY: {geom_name}")
    print("-" * 72)

    print(f"Geometry ID: {gid}")
    print(f"Type: {geom_type}")
    print(f"Parent body: {body_name}")

    print(
        "Local position (m):",
        np.round(local_position, 6),
    )

    print(
        "Local quaternion (wxyz):",
        np.round(local_quaternion, 6),
    )

    print(
        "Geometry size:",
        np.round(model.geom_size[gid], 6),
    )

    print(
        "World position (m):",
        np.round(world_position, 6),
    )

    print("World rotation:")
    print(rotation_matrix_string(world_rotation))

    print(
        "Collision masks:",
        f"contype={model.geom_contype[gid]},",
        f"conaffinity={model.geom_conaffinity[gid]}",
    )

    print(
        "Friction:",
        np.round(model.geom_friction[gid], 6),
    )

    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:

        mesh_id = int(model.geom_dataid[gid])

        mesh_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_MESH,
            mesh_id,
        )

        print(f"Mesh asset: {mesh_name}")

        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])

        vertices = model.mesh_vert[
            vertex_start:vertex_start + vertex_count
        ]

        if len(vertices):

            print(
                "Compiled mesh vertex minimum:",
                np.round(vertices.min(axis=0), 6),
            )

            print(
                "Compiled mesh vertex maximum:",
                np.round(vertices.max(axis=0), 6),
            )


# ============================================================================
# CONTACT INSPECTION
# ============================================================================

def inspect_contacts(model, data):

    print("\nSTATIC CONTACTS")
    print("-" * 72)

    if data.ncon == 0:
        print("No contacts.")
        return

    for i in range(data.ncon):

        contact = data.contact[i]

        name1 = object_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = object_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        print(
            f"{i + 1:02d}. {name1} <-> {name2}"
        )

        print(
            f"    Distance: {contact.dist * 1000:.3f} mm"
        )

        print(
            f"    Contact position: {np.round(contact.pos, 6)}"
        )


# ============================================================================
# CONFIGURATION REPORT
# ============================================================================

def inspect_configuration(model, data, label):

    print("\n")
    print("=" * 80)
    print(label)
    print("=" * 80)

    mujoco.mj_forward(model, data)

    for name in TARGET_GEOMS:
        inspect_geom(model, data, name)

    inspect_contacts(model, data)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("OPENARM V1 — CUP PICKUP COLLISION GEOMETRY INSPECTION")
    print("=" * 80)

    if not SCENE_XML.exists():
        raise FileNotFoundError(SCENE_XML)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    print(f"\nScene: {SCENE_XML}")

    set_home(model, data)

    inspect_configuration(
        model,
        data,
        "CONFIGURATION 1 — VALIDATED HOME",
    )

    set_candidate_a(model, data)

    inspect_configuration(
        model,
        data,
        "CONFIGURATION 2 — CANDIDATE A",
    )

    print("\n")
    print("=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)

    print("No robot trajectory executed.")
    print("No XML files modified.")


if __name__ == "__main__":
    main()