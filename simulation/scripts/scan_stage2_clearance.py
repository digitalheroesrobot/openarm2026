#!/usr/bin/env python3

"""
OpenArm V1 — Stage 2 Physical Clearance Scan

Purpose:
    Evaluate whether changing symmetric joint-2 angles improves
    the hand/table clearance during the joint-4 movement.

Important:
    Only geometries whose names end in "_collision" are included.
    Visual geometries are excluded.

Method:
    - Hold joints 1, 3, 5, 6, and 7 at the measured collision state.
    - Sweep joint 2 using symmetric offsets.
    - Sweep joint 4 from 0 to the final home value.
    - Measure minimum distance between robot collision geometries
      and the tabletop.

This is a geometric diagnostic, not an actuator-driven trajectory test.
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

FINAL_JOINT4 = 1.486495

# Refined scan around the promising region.
JOINT2_OFFSETS = np.arange(0.020, 0.101, 0.005)

JOINT4_SAMPLES = 301

DISTANCE_LIMIT = 1.0

TARGET_CLEARANCE_MM = 5.0


ACTUAL_COLLISION_STATE = {
    "openarm_left_joint1": -0.02797842,
    "openarm_left_joint2": -0.78014217,
    "openarm_left_joint3": 0.02074379,
    "openarm_left_joint4": 0.41183687,
    "openarm_left_joint5": -0.00056361,
    "openarm_left_joint6": -0.01442052,
    "openarm_left_joint7": 0.00501762,

    "openarm_right_joint1": 0.02797318,
    "openarm_right_joint2": 0.78007019,
    "openarm_right_joint3": -0.02075523,
    "openarm_right_joint4": 0.41182547,
    "openarm_right_joint5": 0.00042943,
    "openarm_right_joint6": 0.01436131,
    "openarm_right_joint7": -0.00509042,
}

ARM_JOINTS = list(ACTUAL_COLLISION_STATE.keys())

FINGER_NAMES = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]


# ============================================================
# HELPERS
# ============================================================

def get_id(model, object_type, name):

    result = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if result < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")

    return result


def get_geom_name(model, geom_id):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    return name or f"geom_{geom_id}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("OPENARM V1 — REFINED STAGE 2 CLEARANCE SCAN")
    print("=" * 72)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Resolve joint addresses
    # --------------------------------------------------------

    qpos_addresses = {}

    for name in ARM_JOINTS:

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_addresses[name] = int(
            model.jnt_qposadr[jid]
        )

    left_q2 = qpos_addresses["openarm_left_joint2"]
    right_q2 = qpos_addresses["openarm_right_joint2"]

    left_q4 = qpos_addresses["openarm_left_joint4"]
    right_q4 = qpos_addresses["openarm_right_joint4"]

    # --------------------------------------------------------
    # Identify table and robot collision geometries
    # --------------------------------------------------------

    table_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    robot_collision_ids = []
    robot_visual_ids = []

    for gid in range(model.ngeom):

        name = get_geom_name(model, gid)

        if not name.startswith("openarm_"):
            continue

        if name.endswith("_collision"):
            robot_collision_ids.append(gid)

        elif "visual" in name:
            robot_visual_ids.append(gid)

    print("\nGeometry classification:")
    print(f"  Robot collision geometries: {len(robot_collision_ids)}")
    print(f"  Robot visual geometries:    {len(robot_visual_ids)}")
    print(f"  Table geometry:              {get_geom_name(model, table_id)}")

    if not robot_collision_ids:
        raise RuntimeError(
            "No robot collision geometries found. "
            "Check the geometry naming convention."
        )

    # --------------------------------------------------------
    # Joint configuration
    # --------------------------------------------------------

    baseline = model.qpos0.copy()

    for name, value in ACTUAL_COLLISION_STATE.items():
        baseline[qpos_addresses[name]] = value

    finger_addresses = []

    for name in FINGER_NAMES:

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        finger_addresses.append(
            int(model.jnt_qposadr[jid])
        )

    for address in finger_addresses:
        baseline[address] = 0.044

    # --------------------------------------------------------
    # Distance measurement
    # --------------------------------------------------------

    fromto = np.zeros(6)

    def measure_clearance():

        minimum_distance = float("inf")
        closest_geom = None

        for gid in robot_collision_ids:

            distance = mujoco.mj_geomDistance(
                model,
                data,
                gid,
                table_id,
                DISTANCE_LIMIT,
                fromto,
            )

            if distance < minimum_distance:

                minimum_distance = distance
                closest_geom = get_geom_name(model, gid)

        return minimum_distance, closest_geom

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    results = []

    q4_values = np.linspace(
        0.0,
        FINAL_JOINT4,
        JOINT4_SAMPLES,
    )

    print("\nScan configuration:")
    print(f"  Joint-2 offsets: {len(JOINT2_OFFSETS)}")
    print(f"  Joint-4 samples: {len(q4_values)}")
    print(f"  Total configurations: {len(JOINT2_OFFSETS) * len(q4_values)}")

    print("\nScanning...\n")

    for index, offset in enumerate(JOINT2_OFFSETS, start=1):

        minimum_distance = float("inf")
        minimum_q4 = None
        closest_geom = None

        for q4 in q4_values:

            data.qpos[:] = baseline

            # Increase/decrease the magnitude symmetrically.
            data.qpos[left_q2] = (
                ACTUAL_COLLISION_STATE["openarm_left_joint2"]
                - offset
            )

            data.qpos[right_q2] = (
                ACTUAL_COLLISION_STATE["openarm_right_joint2"]
                + offset
            )

            data.qpos[left_q4] = q4
            data.qpos[right_q4] = q4

            mujoco.mj_forward(model, data)

            distance, geom_name = measure_clearance()

            if distance < minimum_distance:

                minimum_distance = distance
                minimum_q4 = q4
                closest_geom = geom_name

        clearance_mm = minimum_distance * 1000.0

        results.append({
            "offset": offset,
            "clearance_mm": clearance_mm,
            "q4": minimum_q4,
            "geom": closest_geom,
        })

        print(
            f"[{index:02d}/{len(JOINT2_OFFSETS)}] "
            f"q2 offset={offset:+.3f} rad "
            f"({np.degrees(offset):+.2f} deg) | "
            f"minimum clearance={clearance_mm:+.3f} mm | "
            f"q4={minimum_q4:.4f} rad | "
            f"{closest_geom}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results.sort(
        key=lambda item: item["clearance_mm"],
        reverse=True,
    )

    print("\n" + "=" * 72)
    print("BEST CANDIDATES")
    print("=" * 72)

    for result in results[:10]:

        print(
            f"Offset={result['offset']:+.3f} rad "
            f"({np.degrees(result['offset']):+.2f} deg) | "
            f"Clearance={result['clearance_mm']:+.3f} mm | "
            f"q4={result['q4']:.4f} rad | "
            f"Geometry={result['geom']}"
        )

    # --------------------------------------------------------
    # Clearance assessment
    # --------------------------------------------------------

    safe_results = [
        result
        for result in results
        if result["clearance_mm"] >= TARGET_CLEARANCE_MM
    ]

    print("\n" + "=" * 72)
    print("CLEARANCE ASSESSMENT")
    print("=" * 72)

    if safe_results:

        print(
            f"Candidates with at least "
            f"{TARGET_CLEARANCE_MM:.1f} mm clearance:"
        )

        for result in safe_results:

            print(
                f"  Offset={result['offset']:+.3f} rad | "
                f"Clearance={result['clearance_mm']:.3f} mm"
            )

    else:

        print(
            f"No candidate achieved "
            f"{TARGET_CLEARANCE_MM:.1f} mm clearance."
        )

    print("\nScan completed.")
    print(
        "Remember: these results hold the other joints fixed "
        "at the measured collision configuration."
    )
    print(
        "An actuator-driven trajectory test is still required."
    )


if __name__ == "__main__":
    main()