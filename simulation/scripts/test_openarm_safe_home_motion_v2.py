#!/usr/bin/env python3

"""
OpenArm V1 — Safe Home Motion V2

Purpose:
    Test the revised three-stage home trajectory using actual
    MuJoCo position actuators.

Revised waypoint:
    Left joint 2:  -0.874334 rad
    Right joint 2: +0.874334 rad

This adds a symmetric 0.060 rad offset relative to the
previous temporary waypoint.

Stages:
    1. Initial pose -> temporary posture
    2. Raise joint 4
    3. Temporary posture -> final home posture

Collision policy:
    Allowed:
        Cup <-> Table

    Forbidden:
        Robot <-> Table
        Robot <-> Cup
        Robot <-> Robot

Original actuator gains and force limits are unchanged.

This script runs headlessly to avoid viewer-related GLX errors.
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

DT = 0.001

SETTLE_TIME = 1.0

STAGE1_DURATION = 4.0
STAGE2_DURATION = 6.0
STAGE3_DURATION = 4.0

FINGER_OPEN = 0.044

PRINT_INTERVAL = 0.5

COLLISION_THRESHOLD_MM = 0.01

DISTANCE_LIMIT = 0.5


# Joint order:
# Left joints 1-7, then right joints 1-7.

START = np.zeros(14)


# Revised temporary waypoint.
WAYPOINT_1 = np.array([
    -0.035630,
    -0.874334,
     0.015058,
     0.0,
     0.0,
     0.0,
     0.0,

     0.035630,
     0.874334,
    -0.015058,
     0.0,
     0.0,
     0.0,
     0.0,
])


# Stage 2: raise joint 4.
WAYPOINT_2 = WAYPOINT_1.copy()

WAYPOINT_2[3] = 1.486495
WAYPOINT_2[10] = 1.486495


# Final home posture.
HOME = np.array([
     0.064370,
    -0.214334,
     0.015058,
     1.486495,
     0.0,
     0.0,
     0.0,

    -0.064370,
     0.214334,
    -0.015058,
     1.486495,
     0.0,
     0.0,
     0.0,
])


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

FINGER_ACTUATOR_NAMES = [
    "left_finger1_ctrl",
    "right_finger1_ctrl",
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


def smoothstep(value):

    value = np.clip(value, 0.0, 1.0)

    return value * value * (3.0 - 2.0 * value)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("OPENARM V1 — SAFE HOME MOTION V2")
    print("=" * 72)

    print("\nRevised joint-2 waypoint:")
    print(f"  Left:  {WAYPOINT_1[1]:+.6f} rad")
    print(f"  Right: {WAYPOINT_1[8]:+.6f} rad")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = DT

    data = mujoco.MjData(model)

    print("\nModel:")
    print(f"  nq: {model.nq}")
    print(f"  nv: {model.nv}")
    print(f"  nu: {model.nu}")
    print(f"  timestep: {model.opt.timestep}")

    # --------------------------------------------------------
    # Resolve joint addresses and actuator IDs
    # --------------------------------------------------------

    qpos_addresses = {}
    arm_actuator_ids = {}

    for name in JOINT_NAMES:

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_addresses[name] = int(
            model.jnt_qposadr[jid]
        )

        actuator_id = -1

        for aid in range(model.nu):

            if int(model.actuator_trnid[aid, 0]) == jid:
                actuator_id = aid
                break

        if actuator_id < 0:
            raise RuntimeError(
                f"No actuator found for joint: {name}"
            )

        arm_actuator_ids[name] = actuator_id

    # --------------------------------------------------------
    # Finger addresses and actuators
    # --------------------------------------------------------

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

    finger_actuator_ids = {}

    for name in FINGER_ACTUATOR_NAMES:

        finger_actuator_ids[name] = get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

    # --------------------------------------------------------
    # Robot collision geometry IDs
    # --------------------------------------------------------

    robot_collision_ids = set()

    for gid in range(model.ngeom):

        name = get_geom_name(model, gid)

        if (
            name.startswith("openarm_")
            and name.endswith("_collision")
        ):
            robot_collision_ids.add(gid)

    if not robot_collision_ids:
        raise RuntimeError(
            "No OpenArm collision geometries found."
        )

    table_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    print("\nCollision geometry:")
    print(f"  Robot collision geometries: {len(robot_collision_ids)}")
    print(f"  Table: {get_geom_name(model, table_id)}")

    # --------------------------------------------------------
    # Initialize simulation
    # --------------------------------------------------------

    data.qpos[:] = model.qpos0

    for name in JOINT_NAMES:
        data.qpos[qpos_addresses[name]] = 0.0

    for address in finger_addresses:
        data.qpos[address] = FINGER_OPEN

    mujoco.mj_forward(model, data)

    def command_positions(target):

        for i, name in enumerate(JOINT_NAMES):

            aid = arm_actuator_ids[name]

            data.ctrl[aid] = target[i]

    def actual_positions():

        return np.array([
            data.qpos[qpos_addresses[name]]
            for name in JOINT_NAMES
        ])

    # Initialize actuator controls from current positions.
    for i, name in enumerate(JOINT_NAMES):

        aid = arm_actuator_ids[name]

        data.ctrl[aid] = data.qpos[qpos_addresses[name]]

    for name, aid in finger_actuator_ids.items():

        data.ctrl[aid] = FINGER_OPEN

    # --------------------------------------------------------
    # Collision detection
    # --------------------------------------------------------

    def forbidden_contacts():

        result = []

        for i in range(data.ncon):

            contact = data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            robot1 = geom1 in robot_collision_ids
            robot2 = geom2 in robot_collision_ids

            if not robot1 and not robot2:
                continue

            distance_mm = contact.dist * 1000.0

            if distance_mm < -COLLISION_THRESHOLD_MM:

                result.append({
                    "geom1": get_geom_name(model, geom1),
                    "geom2": get_geom_name(model, geom2),
                    "distance_mm": distance_mm,
                })

        return result

    # --------------------------------------------------------
    # Table clearance measurement
    # --------------------------------------------------------

    fromto = np.zeros(6)

    def minimum_table_clearance():

        minimum = float("inf")
        closest = None

        for gid in robot_collision_ids:

            distance = mujoco.mj_geomDistance(
                model,
                data,
                gid,
                table_id,
                DISTANCE_LIMIT,
                fromto,
            )

            if distance < minimum:

                minimum = distance
                closest = get_geom_name(model, gid)

        return minimum * 1000.0, closest

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def report(stage, commanded):

        actual = actual_positions()

        error = commanded - actual

        clearance, closest = minimum_table_clearance()

        print(f"\n[{stage}]")
        print(f"  Simulation time: {data.time:.3f} s")

        print(
            "  Maximum tracking error:",
            f"{np.degrees(np.max(np.abs(error))):.4f} deg",
        )

        print(
            "  Left/right joint 4:",
            f"{actual[3]:.5f}, {actual[10]:.5f}",
        )

        print(
            "  Minimum robot/table clearance:",
            f"{clearance:.3f} mm",
        )

        print("  Closest geometry:", closest)

        contacts = forbidden_contacts()

        if contacts:

            print("  Forbidden contacts:")

            for contact in contacts:

                print(
                    f"    {contact['geom1']} <-> "
                    f"{contact['geom2']} | "
                    f"{contact['distance_mm']:.4f} mm"
                )

        else:

            print("  Forbidden contacts: None")

        return clearance

    # --------------------------------------------------------
    # Execute one trajectory stage
    # --------------------------------------------------------

    def execute_stage(name, start, target, duration):

        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)

        steps = int(round(duration / DT))

        next_report_time = data.time

        stage_minimum_clearance = float("inf")
        stage_closest_geometry = None

        for step in range(steps + 1):

            alpha = step / steps

            blend = smoothstep(alpha)

            commanded = (
                start + blend * (target - start)
            )

            command_positions(commanded)

            mujoco.mj_step(model, data)

            # Check collisions at every simulation step.
            contacts = forbidden_contacts()

            if contacts:

                print("\n!!! ROBOT COLLISION DETECTED !!!")

                report(
                    name + " — COLLISION STATE",
                    commanded,
                )

                print("\nActual joint configuration:")

                for joint, value in zip(
                    JOINT_NAMES,
                    actual_positions(),
                ):

                    print(f"  {joint}: {value:.8f}")

                return False

            # Measure geometric clearance periodically.
            if (
                data.time >= next_report_time
                or step == steps
            ):

                clearance = report(name, commanded)

                if clearance < stage_minimum_clearance:

                    stage_minimum_clearance = clearance

                    _, stage_closest_geometry = (
                        minimum_table_clearance()
                    )

                next_report_time += PRINT_INTERVAL

        print(f"\n{name} PASSED")

        print(
            "  Lowest reported robot/table clearance:",
            f"{stage_minimum_clearance:.3f} mm",
        )

        print(
            "  Closest geometry at reported minimum:",
            stage_closest_geometry,
        )

        return True

    # --------------------------------------------------------
    # Settling
    # --------------------------------------------------------

    print("\n========== SETTLING ==========")

    for _ in range(int(SETTLE_TIME / DT)):

        command_positions(START)

        mujoco.mj_step(model, data)

        contacts = forbidden_contacts()

        if contacts:

            print("Collision during settling.")

            for contact in contacts:
                print(contact)

            return

    print("Settling completed.")

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    success = execute_stage(
        "STAGE 1 — INITIAL TO TEMPORARY POSTURE",
        START,
        WAYPOINT_1,
        STAGE1_DURATION,
    )

    if not success:

        print("\nMOTION ABORTED DURING STAGE 1.")
        return

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    success = execute_stage(
        "STAGE 2 — RAISE JOINT 4",
        WAYPOINT_1,
        WAYPOINT_2,
        STAGE2_DURATION,
    )

    if not success:

        print("\nMOTION ABORTED DURING STAGE 2.")
        return

    # --------------------------------------------------------
    # Stage 3
    # --------------------------------------------------------

    success = execute_stage(
        "STAGE 3 — FINAL HOME ADJUSTMENT",
        WAYPOINT_2,
        HOME,
        STAGE3_DURATION,
    )

    if not success:

        print("\nMOTION ABORTED DURING STAGE 3.")
        return

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL HOME VERIFICATION")
    print("=" * 72)

    final_actual = actual_positions()

    final_error = HOME - final_actual

    print(
        "Maximum final tracking error:",
        f"{np.degrees(np.max(np.abs(final_error))):.4f} deg",
    )

    print("\nFinal joint configuration:")

    for name, value in zip(JOINT_NAMES, final_actual):

        print(f"  {name}: {value:.6f}")

    final_contacts = forbidden_contacts()

    if final_contacts:

        print("\nFINAL COLLISION CHECK: FAIL")

        for contact in final_contacts:
            print(contact)

        return

    print("\nFINAL COLLISION CHECK: PASS")

    print("\n" + "=" * 72)
    print("ALL THREE HOME-MOTION STAGES PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()