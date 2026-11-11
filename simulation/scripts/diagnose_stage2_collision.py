#!/usr/bin/env python3

"""
OpenArm V1 — Stage 2 Collision Diagnostics

Purpose:
    Reproduce the three-stage home motion using MuJoCo actuators.

Collision policy:
    Allowed:
        Cup <-> Table

    Forbidden:
        Robot <-> Table
        Robot <-> Cup
        Robot <-> Robot

Diagnostics:
    - Actual and commanded joint positions
    - Joint tracking errors
    - Actuator forces
    - First forbidden contact
    - Full joint configuration at collision

The original MuJoCo XML and actuator parameters are unchanged.
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
FINGER_OPEN = 0.044

SETTLE_TIME = 1.0
STAGE1_DURATION = 4.0
STAGE2_DURATION = 6.0

PRINT_INTERVAL = 0.5
COLLISION_THRESHOLD_MM = 0.01

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

START = np.zeros(14)

WAYPOINT_1 = np.array([
    -0.035630, -0.814334,  0.015058, 0.0,
     0.0,       0.0,       0.0,

     0.035630,  0.814334, -0.015058, 0.0,
     0.0,       0.0,       0.0,
])

WAYPOINT_2 = WAYPOINT_1.copy()
WAYPOINT_2[3] = 1.486495
WAYPOINT_2[10] = 1.486495


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
        raise RuntimeError(f"Object not found: {name}")

    return result


def geom_name(model, geom_id):

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPENARM V1 — STAGE 2 COLLISION DIAGNOSTICS")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = DT

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Joint addresses
    # --------------------------------------------------------

    joint_ids = {
        name: object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in JOINT_NAMES
    }

    qpos_addresses = {
        name: model.jnt_qposadr[jid]
        for name, jid in joint_ids.items()
    }

    actuator_ids = {}

    for aid in range(model.nu):

        jid = int(model.actuator_trnid[aid, 0])

        if jid >= 0:
            actuator_ids[jid] = aid

    arm_actuators = {
        name: actuator_ids[jid]
        for name, jid in joint_ids.items()
    }

    finger_addresses = []

    for name in FINGER_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        finger_addresses.append(
            model.jnt_qposadr[jid]
        )

    # --------------------------------------------------------
    # Identify robot geometry
    # --------------------------------------------------------

    robot_geom_ids = set()

    for gid in range(model.ngeom):

        name = geom_name(model, gid)

        if name and name.startswith("openarm_"):
            robot_geom_ids.add(gid)

    print("\nRobot collision geometries:", len(robot_geom_ids))

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    data.qpos[:] = model.qpos0

    for name in JOINT_NAMES:
        data.qpos[qpos_addresses[name]] = 0.0

    for address in finger_addresses:
        data.qpos[address] = FINGER_OPEN

    mujoco.mj_forward(model, data)

    for aid in range(model.nu):

        jid = int(model.actuator_trnid[aid, 0])

        if jid >= 0:

            qadr = model.jnt_qposadr[jid]

            data.ctrl[aid] = data.qpos[qadr]

    for name in (
        "openarm_left_finger_joint1",
        "openarm_right_finger_joint1",
    ):

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        data.ctrl[actuator_ids[jid]] = FINGER_OPEN

    # --------------------------------------------------------
    # Position helpers
    # --------------------------------------------------------

    def actual_positions():

        return np.array([
            data.qpos[qpos_addresses[name]]
            for name in JOINT_NAMES
        ])

    def command_positions(target):

        for i, name in enumerate(JOINT_NAMES):
            data.ctrl[arm_actuators[name]] = target[i]

    # --------------------------------------------------------
    # Correct collision classification
    # --------------------------------------------------------

    def get_forbidden_contacts():

        forbidden = []
        allowed = []

        for i in range(data.ncon):

            contact = data.contact[i]

            name1 = geom_name(model, contact.geom1)
            name2 = geom_name(model, contact.geom2)

            robot1 = contact.geom1 in robot_geom_ids
            robot2 = contact.geom2 in robot_geom_ids

            distance_mm = contact.dist * 1000.0

            info = {
                "geom1": name1,
                "geom2": name2,
                "distance_mm": distance_mm,
            }

            # Cup-table contact is allowed.
            if not robot1 and not robot2:
                allowed.append(info)
                continue

            # Robot contact with another robot or environment.
            if distance_mm < -COLLISION_THRESHOLD_MM:
                forbidden.append(info)

        return forbidden, allowed

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def print_diagnostics(stage, commanded):

        actual = actual_positions()
        error = commanded - actual

        forbidden, allowed = get_forbidden_contacts()

        print(f"\n[{stage}]")
        print(f"Simulation time: {data.time:.3f} s")

        print(
            "Maximum tracking error:",
            f"{np.degrees(np.max(np.abs(error))):.4f} deg",
        )

        print(
            "Actual joint 4 left/right:",
            f"{actual[3]:.6f}, {actual[10]:.6f}",
        )

        print(
            "Commanded joint 4 left/right:",
            f"{commanded[3]:.6f}, {commanded[10]:.6f}",
        )

        print("\nJoint errors:")

        for name, value in zip(JOINT_NAMES, error):

            if abs(value) > np.radians(0.1):

                print(
                    f"  {name}: "
                    f"{np.degrees(value):.4f} deg"
                )

        print("\nActuator forces:")

        for name, aid in arm_actuators.items():

            force = data.actuator_force[aid]

            if abs(force) > 0.1:
                print(f"  {name}: {force:.3f}")

        print("\nForbidden robot contacts:")

        if not forbidden:
            print("  None")

        for contact in forbidden:

            print(
                f"  {contact['geom1']} <-> "
                f"{contact['geom2']} | "
                f"{contact['distance_mm']:.4f} mm"
            )

        if allowed:

            print("\nAllowed non-robot contacts:")

            for contact in allowed:

                print(
                    f"  {contact['geom1']} <-> "
                    f"{contact['geom2']} | "
                    f"{contact['distance_mm']:.4f} mm"
                )

    # --------------------------------------------------------
    # Trajectory execution
    # --------------------------------------------------------

    def execute_stage(name, start, target, duration):

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        steps = int(duration / DT)
        last_print = -PRINT_INTERVAL

        for step in range(steps + 1):

            alpha = step / steps

            smooth = alpha * alpha * (3.0 - 2.0 * alpha)

            commanded = start + smooth * (target - start)

            command_positions(commanded)

            mujoco.mj_step(model, data)

            if data.time - last_print >= PRINT_INTERVAL:

                print_diagnostics(name, commanded)
                last_print = data.time

            forbidden, _ = get_forbidden_contacts()

            if forbidden:

                print("\n!!! ROBOT COLLISION DETECTED !!!")

                print_diagnostics(
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

        print_diagnostics(name + " COMPLETED", target)

        return True

    # --------------------------------------------------------
    # Settling
    # --------------------------------------------------------

    print("\n========== SETTLING ==========")

    for _ in range(int(SETTLE_TIME / DT)):

        command_positions(START)
        mujoco.mj_step(model, data)

        forbidden, _ = get_forbidden_contacts()

        if forbidden:

            print("Robot collision during settling.")

            print_diagnostics(
                "SETTLING",
                START,
            )

            return

    print("Settling completed.")
    print("Allowed cup-table contacts are ignored.")

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    success = execute_stage(
        "STAGE 1 — TEMPORARY POSTURE",
        START,
        WAYPOINT_1,
        STAGE1_DURATION,
    )

    if not success:

        print("\nStopped during Stage 1.")
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

    print("\n" + "=" * 70)

    if success:
        print("STAGE 2 COMPLETED WITHOUT ROBOT COLLISION")
    else:
        print("STAGE 2 ABORTED — ROBOT COLLISION CONFIRMED")

    print("=" * 70)


if __name__ == "__main__":
    main()