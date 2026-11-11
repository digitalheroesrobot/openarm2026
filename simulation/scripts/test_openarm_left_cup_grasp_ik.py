#!/usr/bin/env python3
"""
OPENARM V1 — CUP GRASP WRIST ORIENTATION SCAN

Purpose
-------
Search for a collision-free left-arm grasp orientation by rotating
the validated HOME TCP orientation around its local X axis.

For each angle:
1. Calculate the required TCP target from the finger-center offset.
2. Solve IK using the seven left-arm joints.
3. Check joint limits.
4. Check all MuJoCo contacts.
5. Classify robot self-collision, cup collision, and table collision.

This is a static geometry/IK diagnostic.
No trajectory is executed.
No XML is modified.

Run:
    .venv/bin/python simulation/scripts/test_openarm_left_cup_grasp_ik.py
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

TCP_NAME = "openarm_left_hand_tcp"

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

LEFT_FINGER_GEOMS = [
    "openarm_left_right_finger_collision",
    "openarm_left_left_finger_collision",
]

GRIPPER_OPENING = 0.044

# Desired average finger collision-geometry center.
CUP_GRASP_CENTER = np.array([0.200, 0.000, 0.355])

# Scan range.
ORIENTATION_ANGLES_DEG = np.arange(0.0, 181.0, 15.0)

HOME = np.array([
     0.064370, -0.214334,  0.015058,
     1.486495,  0.000000,  0.000000,  0.000000,

    -0.064370,  0.214334, -0.015058,
     1.486495,  0.000000,  0.000000,  0.000000,
])

IK_MAX_ITERATIONS = 700
POSITION_TOLERANCE = 0.001
ORIENTATION_TOLERANCE = 0.015

DAMPING = 0.03
MAX_JOINT_STEP = 0.08
ORIENTATION_WEIGHT = 0.35

# Report a candidate as collision-free only if no MuJoCo contacts exist.
# This is intentionally conservative for this diagnostic.


# ============================================================================
# MUJOCO HELPERS
# ============================================================================

def get_id(model, object_type, name):
    result = mujoco.mj_name2id(model, object_type, name)

    if result < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")

    return result


def qpos_address(model, name):
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def dof_address(model, name):
    jid = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_dofadr[jid])


def body_pose(model, data, name):
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    return (
        data.xpos[bid].copy(),
        data.xmat[bid].reshape(3, 3).copy(),
    )


def rotation_error(current, target):
    error = np.zeros(3)

    for axis in range(3):
        error += np.cross(
            current[:, axis],
            target[:, axis],
        )

    return 0.5 * error


def rotation_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_home(model, data):

    data.qpos[:] = model.qpos0

    for i, name in enumerate(LEFT_JOINTS):
        data.qpos[qpos_address(model, name)] = HOME[i]

    for i, name in enumerate(RIGHT_JOINTS):
        data.qpos[qpos_address(model, name)] = HOME[i + 7]

    for name in FINGER_JOINTS:
        data.qpos[qpos_address(model, name)] = GRIPPER_OPENING

    mujoco.mj_forward(model, data)

    return data.qpos.copy()


# ============================================================================
# TCP-TO-FINGER OFFSET
# ============================================================================

def calculate_finger_offset(model, data):

    tcp_position, tcp_rotation = body_pose(
        model, data, TCP_NAME
    )

    finger_positions = []

    for name in LEFT_FINGER_GEOMS:
        gid = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        finger_positions.append(data.geom_xpos[gid].copy())

    finger_center = np.mean(finger_positions, axis=0)

    offset_world = finger_center - tcp_position
    offset_local = tcp_rotation.T @ offset_world

    return tcp_rotation, offset_local


# ============================================================================
# IK SOLVER
# ============================================================================

def solve_ik(
    model,
    data,
    initial_qpos,
    target_position,
    target_rotation,
):

    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)

    tcp_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        TCP_NAME,
    )

    q_addresses = [
        qpos_address(model, name)
        for name in LEFT_JOINTS
    ]

    dof_addresses = [
        dof_address(model, name)
        for name in LEFT_JOINTS
    ]

    joint_ids = [
        get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in LEFT_JOINTS
    ]

    joint_ranges = np.array([
        model.jnt_range[jid]
        for jid in joint_ids
    ])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for iteration in range(IK_MAX_ITERATIONS):

        mujoco.mj_forward(model, data)

        current_position, current_rotation = body_pose(
            model, data, TCP_NAME
        )

        ep = target_position - current_position
        er = rotation_error(current_rotation, target_rotation)

        position_error = np.linalg.norm(ep)
        orientation_error = np.linalg.norm(er)

        if (
            position_error <= POSITION_TOLERANCE
            and orientation_error <= ORIENTATION_TOLERANCE
        ):
            break

        mujoco.mj_jacBody(
            model,
            data,
            jacp,
            jacr,
            tcp_id,
        )

        J = np.zeros((6, 7))

        for i, dof in enumerate(dof_addresses):
            J[:3, i] = jacp[:, dof]
            J[3:, i] = jacr[:, dof]

        J[3:, :] *= ORIENTATION_WEIGHT

        error = np.concatenate([
            ep,
            er * ORIENTATION_WEIGHT,
        ])

        system = J @ J.T + DAMPING**2 * np.eye(6)

        try:
            dq = J.T @ np.linalg.solve(system, error)
        except np.linalg.LinAlgError:
            break

        dq = np.clip(
            dq,
            -MAX_JOINT_STEP,
            MAX_JOINT_STEP,
        )

        for i, address in enumerate(q_addresses):

            data.qpos[address] += dq[i]

            lower, upper = joint_ranges[i]

            data.qpos[address] = np.clip(
                data.qpos[address],
                lower,
                upper,
            )

    mujoco.mj_forward(model, data)

    final_position, final_rotation = body_pose(
        model, data, TCP_NAME
    )

    final_position_error = np.linalg.norm(
        target_position - final_position
    )

    final_orientation_error = np.linalg.norm(
        rotation_error(final_rotation, target_rotation)
    )

    converged = (
        final_position_error <= POSITION_TOLERANCE
        and final_orientation_error <= ORIENTATION_TOLERANCE
    )

    joint_values = np.array([
        data.qpos[address]
        for address in q_addresses
    ])

    # Distance to the closest joint limit, in radians.
    joint_margins = np.minimum(
        joint_values - joint_ranges[:, 0],
        joint_ranges[:, 1] - joint_values,
    )

    return {
        "converged": bool(converged),
        "iterations": iteration + 1,
        "position_error": float(final_position_error),
        "orientation_error": float(final_orientation_error),
        "joint_values": joint_values,
        "joint_margins": joint_margins,
        "joint_ranges": joint_ranges,
        "tcp_position": final_position,
    }


# ============================================================================
# CONTACT CLASSIFICATION
# ============================================================================

def get_geom_name(model, gid):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        gid,
    )

    return name if name else f"geom_{gid}"


def classify_contact(name1, name2):

    names = (name1, name2)

    if "cup" in names:
        return "CUP"

    if "table" in name1 or "table" in name2:
        return "TABLE"

    if "openarm" in name1 and "openarm" in name2:
        return "SELF"

    return "OTHER"


def inspect_contacts(model, data):

    contacts = []

    for i in range(data.ncon):

        contact = data.contact[i]

        name1 = get_geom_name(model, contact.geom1)
        name2 = get_geom_name(model, contact.geom2)

        contacts.append({
            "name1": name1,
            "name2": name2,
            "category": classify_contact(name1, name2),
            "distance": float(contact.dist),
        })

    return contacts


# ============================================================================
# ORIENTATION SCAN
# ============================================================================

def test_orientation(
    model,
    data,
    initial_qpos,
    home_rotation,
    offset_local,
    angle_deg,
):

    angle_rad = np.radians(angle_deg)

    target_rotation = (
        home_rotation @ rotation_x(angle_rad)
    )

    target_tcp_position = (
        CUP_GRASP_CENTER
        - target_rotation @ offset_local
    )

    result = solve_ik(
        model,
        data,
        initial_qpos,
        target_tcp_position,
        target_rotation,
    )

    contacts = inspect_contacts(model, data)

    contact_counts = {
        category: sum(
            c["category"] == category
            for c in contacts
        )
        for category in ["SELF", "CUP", "TABLE", "OTHER"]
    }

    collision_free = len(contacts) == 0

    # Require both accurate IK and no detected contacts.
    feasible = result["converged"] and collision_free

    return {
        "angle": angle_deg,
        "result": result,
        "contacts": contacts,
        "contact_counts": contact_counts,
        "collision_free": collision_free,
        "feasible": feasible,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 84)
    print("OPENARM V1 — CUP GRASP WRIST ORIENTATION SCAN")
    print("=" * 84)

    if not SCENE_XML.exists():
        raise FileNotFoundError(SCENE_XML)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    initial_qpos = initialize_home(model, data)

    home_tcp_position, home_rotation = body_pose(
        model, data, TCP_NAME
    )

    measured_rotation, offset_local = calculate_finger_offset(
        model, data
    )

    print("\nInitial HOME:")
    print("TCP position:", np.round(home_tcp_position, 6))
    print("Finger-center offset in TCP frame:", np.round(offset_local, 6))

    print("\nGrasp center:", CUP_GRASP_CENTER)
    print("Gripper opening: 44 mm")
    print("Rotation axis: TCP-local X")
    print("Scan angles:", ORIENTATION_ANGLES_DEG)

    results = []

    print("\n" + "=" * 84)
    print("SCAN RESULTS")
    print("=" * 84)

    for angle in ORIENTATION_ANGLES_DEG:

        item = test_orientation(
            model,
            data,
            initial_qpos,
            home_rotation,
            offset_local,
            float(angle),
        )

        results.append(item)

        r = item["result"]
        c = item["contact_counts"]

        print(
            f"\nAngle {angle:6.1f} deg | "
            f"IK={'PASS' if r['converged'] else 'FAIL'} | "
            f"Pos={r['position_error'] * 1000:7.2f} mm | "
            f"Ori={np.degrees(r['orientation_error']):6.2f} deg | "
            f"Self={c['SELF']:2d} | "
            f"Cup={c['CUP']:2d} | "
            f"Table={c['TABLE']:2d} | "
            f"Other={c['OTHER']:2d} | "
            f"{'FEASIBLE' if item['feasible'] else 'REJECT'}"
        )

    # ------------------------------------------------------------------------
    # Detailed report for the best candidate.
    # ------------------------------------------------------------------------

    feasible = [
        item for item in results
        if item["feasible"]
    ]

    print("\n" + "=" * 84)
    print("FINAL SUMMARY")
    print("=" * 84)

    print(f"Total orientations tested: {len(results)}")
    print(f"Feasible orientations: {len(feasible)}")

    if feasible:

        # Prefer the solution with the greatest minimum joint-limit margin.
        best = max(
            feasible,
            key=lambda item: np.min(
                item["result"]["joint_margins"]
            ),
        )

        r = best["result"]

        print(f"\nBest angle: {best['angle']:.1f} deg")
        print("TCP position:", np.round(r["tcp_position"], 6))

        print(
            "Position error:",
            f"{r['position_error'] * 1000:.3f} mm",
        )

        print(
            "Orientation error:",
            f"{np.degrees(r['orientation_error']):.3f} deg",
        )

        print("\nLeft arm joints:")

        for i, value in enumerate(r["joint_values"]):
            print(
                f"  Joint {i + 1}: "
                f"{np.degrees(value):.3f} deg"
            )

        print(
            "\nMinimum joint-limit margin:",
            f"{np.degrees(np.min(r['joint_margins'])):.2f} deg",
        )

    else:
        print("\nNo collision-free IK candidate was found.")
        print(
            "The next search must change the grasp-center position, "
            "orientation family, or both."
        )

    print("\nDiagnostic completed.")
    print("No trajectory executed.")
    print("No XML modified.")


if __name__ == "__main__":
    main()