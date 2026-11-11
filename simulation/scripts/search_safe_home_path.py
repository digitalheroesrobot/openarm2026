#!/usr/bin/env python3

"""
OpenArm V1 — Collision-Free Home Path Search

Purpose:
    Find a safe three-stage path from the initial configuration
    to the custom home configuration.

Path:
    1. Change joints 1-3 while joint 4 stays at zero.
    2. Move joint 4 to its home position.
    3. Adjust joints 1-3 to their final home positions.

Search:
    Grid search over temporary symmetric joint 1-3 configurations.

Safety:
    - Original MuJoCo XML remains unchanged.
    - All four fingers remain open at 44 mm.
    - Every interpolated configuration is collision-checked.
    - Only robot-table penetrations are considered obstacles.
"""

from pathlib import Path
from itertools import product

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

APPROACH_STEPS = 31
JOINT4_STEPS = 101
FINAL_STEPS = 31

# Search offsets around the final left-arm joint values.
Q1_OFFSETS = np.linspace(-0.4, 0.4, 9)
Q2_OFFSETS = np.linspace(-0.6, 0.6, 9)
Q3_OFFSETS = np.linspace(-0.6, 0.6, 9)

# Collision tolerance:
# Penetrations deeper than 0.001 mm are rejected.
COLLISION_TOLERANCE = -1e-6


# Joint order:
# Left joints 1-7, followed by right joints 1-7.

HOME = np.array([
     0.064370, -0.214334,  0.015058, 1.486495,
     0.0,      0.0,       0.0,

    -0.064370,  0.214334, -0.015058, 1.486495,
     0.0,       0.0,       0.0,
])

ZERO = np.zeros(14)

JOINT_NAMES = [
    f"openarm_left_joint{i}" for i in range(1, 8)
] + [
    f"openarm_right_joint{i}" for i in range(1, 8)
]

FINGER_NAMES = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]


# ============================================================
# MODEL SETUP
# ============================================================

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

joint_addresses = {}

for name in JOINT_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")

    joint_addresses[name] = model.jnt_qposadr[jid]


finger_addresses = []

for name in FINGER_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(f"Finger joint not found: {name}")

    finger_addresses.append(model.jnt_qposadr[jid])


table_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "table_top",
)

if table_id < 0:
    raise RuntimeError("table_top geometry not found")


# ============================================================
# COLLISION CHECK
# ============================================================

def set_configuration(q):

    for i, name in enumerate(JOINT_NAMES):
        data.qpos[joint_addresses[name]] = q[i]

    for address in finger_addresses:
        data.qpos[address] = FINGER_OPEN

    mujoco.mj_forward(model, data)


def check_table_collision(q):

    set_configuration(q)

    collisions = []

    for i in range(data.ncon):

        contact = data.contact[i]

        if (
            contact.geom1 != table_id
            and contact.geom2 != table_id
        ):
            continue

        if contact.dist < COLLISION_TOLERANCE:

            geom1 = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1,
            )

            geom2 = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2,
            )

            collisions.append({
                "geom1": geom1,
                "geom2": geom2,
                "depth_mm": -contact.dist * 1000.0,
            })

    return collisions


# ============================================================
# TRAJECTORY CHECK
# ============================================================

def check_segment(start, end, steps):

    for alpha in np.linspace(0.0, 1.0, steps):

        q = start + alpha * (end - start)

        collisions = check_table_collision(q)

        if collisions:
            return False, alpha, q.copy(), collisions

    return True, None, None, []


def build_candidate(q1, q2, q3):

    """
    Build a symmetric temporary posture.

    Left:
        q1, q2, q3

    Right:
        -q1, -q2, -q3

    Joint 4 remains zero.
    """

    q = ZERO.copy()

    q[0] = q1
    q[1] = q2
    q[2] = q3

    q[7] = -q1
    q[8] = -q2
    q[9] = -q3

    return q


def validate_candidate(candidate):

    # Stage 1:
    # Initial pose -> temporary posture.

    ok, alpha, q, collisions = check_segment(
        ZERO,
        candidate,
        APPROACH_STEPS,
    )

    if not ok:
        return False, "approach", alpha, q, collisions

    # Stage 2:
    # Temporary posture -> final joint 4 values.

    stage2_end = candidate.copy()

    stage2_end[3] = HOME[3]
    stage2_end[10] = HOME[10]

    ok, alpha, q, collisions = check_segment(
        candidate,
        stage2_end,
        JOINT4_STEPS,
    )

    if not ok:
        return False, "joint4", alpha, q, collisions

    # Stage 3:
    # Final joint 4 values are fixed.
    # Adjust joints 1-3 to the final home values.

    ok, alpha, q, collisions = check_segment(
        stage2_end,
        HOME,
        FINAL_STEPS,
    )

    if not ok:
        return False, "final_adjustment", alpha, q, collisions

    return True, "safe", None, None, []


# ============================================================
# SEARCH
# ============================================================

def main():

    print("=" * 68)
    print("OPENARM V1 — SAFE HOME PATH SEARCH")
    print("=" * 68)

    print("\nModel:", XML_PATH)
    print("Candidate grid:")
    print("  q1:", len(Q1_OFFSETS))
    print("  q2:", len(Q2_OFFSETS))
    print("  q3:", len(Q3_OFFSETS))

    total = (
        len(Q1_OFFSETS)
        * len(Q2_OFFSETS)
        * len(Q3_OFFSETS)
    )

    print("Total candidates:", total)

    # Confirm known endpoints.
    print("\nInitial pose:", "SAFE" if not check_table_collision(ZERO) else "COLLIDING")
    print("Home pose:", "SAFE" if not check_table_collision(HOME) else "COLLIDING")

    if check_table_collision(ZERO) or check_table_collision(HOME):
        raise RuntimeError("Initial or final configuration is not collision-free")

    # Generate candidates around home q1-q3.
    candidates = []

    for d1, d2, d3 in product(
        Q1_OFFSETS,
        Q2_OFFSETS,
        Q3_OFFSETS,
    ):

        q1 = HOME[0] + d1
        q2 = HOME[1] + d2
        q3 = HOME[2] + d3

        candidates.append((
            np.linalg.norm([d1, d2, d3]),
            q1,
            q2,
            q3,
        ))

    # Try candidates closest to home first.
    candidates.sort(key=lambda item: item[0])

    safe_paths = []

    failure_counts = {
        "approach": 0,
        "joint4": 0,
        "final_adjustment": 0,
    }

    print("\nSearching...")

    for index, (distance, q1, q2, q3) in enumerate(
        candidates,
        start=1,
    ):

        candidate = build_candidate(q1, q2, q3)

        valid, stage, alpha, q, collisions = (
            validate_candidate(candidate)
        )

        if valid:

            safe_paths.append({
                "distance": distance,
                "candidate": candidate.copy(),
            })

            print(
                f"\nSAFE PATH FOUND: candidate {index}/{total}"
            )

            print(
                f"Temporary left-arm q1/q2/q3: "
                f"{q1:.6f}, {q2:.6f}, {q3:.6f}"
            )

            print(
                f"Deviation from home: {distance:.6f} rad"
            )

            # Stop at the first candidate.
            break

        failure_counts[stage] += 1

        if index % 50 == 0:

            print(
                f"Checked {index}/{total} | "
                f"safe paths: {len(safe_paths)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 68)
    print("SEARCH RESULTS")
    print("=" * 68)

    print("Candidates tested:", index)
    print("Safe paths found:", len(safe_paths))

    print("\nFailure stages:")
    for stage, count in failure_counts.items():
        print(f"  {stage}: {count}")

    if not safe_paths:

        print("\nNO SAFE PATH FOUND IN THE SEARCH GRID.")

        print(
            "Next step: expand the temporary joint configuration "
            "search range or use Cartesian waypoint planning."
        )

        return

    result = safe_paths[0]
    candidate = result["candidate"]

    stage2_end = candidate.copy()
    stage2_end[3] = HOME[3]
    stage2_end[10] = HOME[10]

    print("\n========== VALIDATED WAYPOINTS ==========")

    print("\nWaypoint 1: Temporary posture")
    for name, value in zip(JOINT_NAMES, candidate):
        print(f"  {name}: {value:.6f}")

    print("\nWaypoint 2: Joint 4 raised")
    for name, value in zip(JOINT_NAMES, stage2_end):
        print(f"  {name}: {value:.6f}")

    print("\nWaypoint 3: Final home")
    for name, value in zip(JOINT_NAMES, HOME):
        print(f"  {name}: {value:.6f}")

    print("\nRESULT: THREE-STAGE PATH PASSED ALL SAMPLED COLLISION CHECKS.")

    print(
        "\nImportant: this is a kinematic path validation. "
        "Actuator tracking, velocity, acceleration, and dynamic "
        "contact behavior still need to be tested separately."
    )


if __name__ == "__main__":
    main()