#!/usr/bin/env python3

"""
OpenArm V1 — Three-Stage Collision-Aware Home Motion

Roadmap:
    1. Initialize OpenArm V1 with open grippers.
    2. Move to a collision-free temporary posture.
    3. Raise both arms using joint 4.
    4. Adjust joints 1-3 to the final home posture.
    5. Monitor contacts, tracking error, and actuator forces.

What you learn:
    - Position-actuator control in MuJoCo.
    - Smooth joint-space trajectory execution.
    - Collision-aware motion planning.
    - Actuator tracking and force-limit diagnostics.

Important:
    - Original V1 XML is not modified.
    - Original actuator gains and force limits are preserved.
    - Motion stops if robot collision is detected.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

XML_PATH = (
    ROOT
    / "simulation/vendor/openarm_mujoco/v1/scene_with_cup.xml"
)

TIMESTEP = 0.001
PRINT_INTERVAL = 1.0
FINGER_OPEN = 0.044

COLLISION_TOLERANCE = -1e-5

SETTLE_TIME = 1.0
STAGE_DURATIONS = [4.0, 6.0, 4.0]
FINAL_HOLD_TIME = 2.0

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

# ------------------------------------------------------------
# Three validated configurations
# ------------------------------------------------------------

START = np.zeros(14)

WAYPOINT_1 = np.array([
    -0.035630,
    -0.814334,
     0.015058,
     0.0,
     0.0,
     0.0,
     0.0,

     0.035630,
     0.814334,
    -0.015058,
     0.0,
     0.0,
     0.0,
     0.0,
])

WAYPOINT_2 = np.array([
    -0.035630,
    -0.814334,
     0.015058,
     1.486495,
     0.0,
     0.0,
     0.0,

     0.035630,
     0.814334,
    -0.015058,
     1.486495,
     0.0,
     0.0,
     0.0,
])

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


# ============================================================
# MODEL SETUP
# ============================================================

def get_joint_id(model, name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")

    return jid


def get_geom_name(model, geom_id):

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )


def main():

    print("=" * 70)
    print("OPENARM V1 — SAFE HOME MOTION")
    print("=" * 70)

    print("\nLoading:", XML_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    model.opt.timestep = TIMESTEP

    # --------------------------------------------------------
    # Joint addresses
    # --------------------------------------------------------

    joint_ids = {
        name: get_joint_id(model, name)
        for name in JOINT_NAMES
    }

    joint_addresses = {
        name: model.jnt_qposadr[jid]
        for name, jid in joint_ids.items()
    }

    finger_addresses = []

    for name in FINGER_NAMES:

        jid = get_joint_id(model, name)

        finger_addresses.append(
            model.jnt_qposadr[jid]
        )

    # --------------------------------------------------------
    # Actuator-to-joint mapping
    # --------------------------------------------------------

    actuator_for_joint = {}

    for aid in range(model.nu):

        joint_id = int(model.actuator_trnid[aid, 0])

        if joint_id < 0:
            continue

        actuator_for_joint[joint_id] = aid

    arm_actuators = {}

    for name, jid in joint_ids.items():

        if jid not in actuator_for_joint:
            raise RuntimeError(
                f"No actuator found for joint: {name}"
            )

        arm_actuators[name] = actuator_for_joint[jid]

    finger_actuators = {}

    for name in (
        "openarm_left_finger_joint1",
        "openarm_right_finger_joint1",
    ):

        jid = get_joint_id(model, name)

        if jid not in actuator_for_joint:
            raise RuntimeError(
                f"No actuator found for finger joint: {name}"
            )

        finger_actuators[name] = actuator_for_joint[jid]

    print("\nModel information:")
    print("  nq:", model.nq)
    print("  nv:", model.nv)
    print("  nu:", model.nu)

    print("\nArm actuators:", len(arm_actuators))
    print("Finger actuators:", len(finger_actuators))

    if len(arm_actuators) != 14:
        raise RuntimeError("Expected 14 arm actuators")

    if len(finger_actuators) != 2:
        raise RuntimeError("Expected 2 finger actuators")

    # --------------------------------------------------------
    # Collision geometry
    # --------------------------------------------------------

    robot_geom_ids = set()

    for gid in range(model.ngeom):

        name = get_geom_name(model, gid)

        if name and name.startswith("openarm_"):
            robot_geom_ids.add(gid)

    # --------------------------------------------------------
    # Initial configuration
    # --------------------------------------------------------

    data.qpos[:] = model.qpos0

    for name in JOINT_NAMES:
        data.qpos[joint_addresses[name]] = 0.0

    for address in finger_addresses:
        data.qpos[address] = FINGER_OPEN

    mujoco.mj_forward(model, data)

    # Initialize all actuator targets to current joint positions.
    for aid in range(model.nu):

        jid = int(model.actuator_trnid[aid, 0])

        if jid >= 0:

            qpos_address = model.jnt_qposadr[jid]

            data.ctrl[aid] = data.qpos[qpos_address]

    # Explicitly command open fingers.
    for aid in finger_actuators.values():
        data.ctrl[aid] = FINGER_OPEN

    # --------------------------------------------------------
    # Collision monitoring
    # --------------------------------------------------------

    def check_collisions():

        table_contacts = []
        self_contacts = []

        for i in range(data.ncon):

            contact = data.contact[i]

            if contact.dist >= COLLISION_TOLERANCE:
                continue

            geom1 = contact.geom1
            geom2 = contact.geom2

            name1 = get_geom_name(model, geom1)
            name2 = get_geom_name(model, geom2)

            robot1 = geom1 in robot_geom_ids
            robot2 = geom2 in robot_geom_ids

            if robot1 and robot2:

                self_contacts.append(
                    (name1, name2, contact.dist * 1000.0)
                )

            elif robot1 or robot2:

                table_contacts.append(
                    (name1, name2, contact.dist * 1000.0)
                )

        return table_contacts, self_contacts

    # --------------------------------------------------------
    # Joint position helpers
    # --------------------------------------------------------

    def read_arm_positions():

        return np.array([
            data.qpos[joint_addresses[name]]
            for name in JOINT_NAMES
        ])

    def command_arm_positions(target):

        for i, name in enumerate(JOINT_NAMES):

            aid = arm_actuators[name]

            data.ctrl[aid] = target[i]

    def maximum_tracking_error(target):

        actual = read_arm_positions()

        return np.max(np.abs(target - actual))

    def maximum_actuator_saturation():

        saturated = 0

        for aid in arm_actuators.values():

            if not model.actuator_forcelimited[aid]:
                continue

            lower, upper = model.actuator_forcerange[aid]

            force = data.actuator_force[aid]

            limit = max(abs(lower), abs(upper))

            if limit > 0 and abs(force) >= 0.98 * limit:
                saturated += 1

        return saturated

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def print_status(stage_name, target):

        table, self_collision = check_collisions()

        print(f"\n[{stage_name}]")
        print(f"Simulation time: {data.time:.3f} s")

        print(
            "Maximum arm tracking error:",
            f"{np.degrees(maximum_tracking_error(target)):.3f} deg",
        )

        print(
            "Saturated arm actuators:",
            maximum_actuator_saturation(),
        )

        print("Robot-environment contacts:", len(table))
        print("Robot self-contacts:", len(self_collision))

        sid_left = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_ee_control_point",
        )

        sid_right = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_ee_control_point",
        )

        print(
            "Left EE:",
            np.round(data.site_xpos[sid_left], 4),
        )

        print(
            "Right EE:",
            np.round(data.site_xpos[sid_right], 4),
        )

        return table, self_collision

    # --------------------------------------------------------
    # Simulation step
    # --------------------------------------------------------

    def step_simulation():

        mujoco.mj_step(model, data)

        table, self_collision = check_collisions()

        if table or self_collision:
            return False, table, self_collision

        return True, [], []

    # --------------------------------------------------------
    # Smooth trajectory execution
    # --------------------------------------------------------

    def execute_segment(
        viewer,
        name,
        start,
        target,
        duration,
    ):

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print("Duration:", duration, "seconds")

        elapsed = 0.0
        last_print = -PRINT_INTERVAL

        steps = int(duration / TIMESTEP)

        wall_start = time.perf_counter()

        for step in range(steps + 1):

            if not viewer.is_running():
                print("Viewer closed by user.")
                return False

            alpha = min(step / steps, 1.0)

            # Smoothstep: zero velocity at both endpoints.
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)

            target_q = start + smooth * (target - start)

            command_arm_positions(target_q)

            mujoco.mj_step(model, data)

            elapsed += TIMESTEP

            table, self_collision = check_collisions()

            if table or self_collision:

                print("\n!!! COLLISION DETECTED — STOPPING !!!")

                for name1, name2, distance in table:
                    print(
                        f"Environment: {name1} <-> {name2} "
                        f"({distance:.4f} mm)"
                    )

                for name1, name2, distance in self_collision:
                    print(
                        f"Self: {name1} <-> {name2} "
                        f"({distance:.4f} mm)"
                    )

                # Hold current arm positions.
                command_arm_positions(read_arm_positions())

                viewer.sync()

                return False

            if elapsed - last_print >= PRINT_INTERVAL:

                print_status(name, target_q)

                last_print = elapsed

            if step % 10 == 0:
                viewer.sync()

            # Pace simulation close to real time.
            expected = (step + 1) * TIMESTEP
            actual = time.perf_counter() - wall_start

            delay = expected - actual

            if delay > 0:
                time.sleep(delay)

        command_arm_positions(target)

        viewer.sync()

        print_status(name + " COMPLETED", target)

        return True

    # ========================================================
    # EXECUTION
    # ========================================================

    print("\n========== INITIALIZATION ==========")

    mujoco.mj_forward(model, data)

    print("Initial arm positions:", read_arm_positions())

    print("Grippers commanded to:", FINGER_OPEN, "m")

    print("\nLaunching MuJoCo viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.4, 0.0, 0.35]

        # ----------------------------------------------------
        # Settle
        # ----------------------------------------------------

        print("\n========== SETTLING ==========")

        settle_steps = int(SETTLE_TIME / TIMESTEP)

        for step in range(settle_steps):

            if not viewer.is_running():
                print("Viewer closed.")
                return

            command_arm_positions(START)

            mujoco.mj_step(model, data)

            table, self_collision = check_collisions()

            if table or self_collision:

                print("Collision during initial settling.")

                print(table)
                print(self_collision)

                return

            if step % 10 == 0:
                viewer.sync()

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        success = execute_segment(
            viewer,
            "STAGE 1 — MOVE TO TEMPORARY POSTURE",
            START,
            WAYPOINT_1,
            STAGE_DURATIONS[0],
        )

        if not success:
            print("\nMOTION ABORTED AT STAGE 1.")
            return

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        success = execute_segment(
            viewer,
            "STAGE 2 — RAISE JOINT 4",
            WAYPOINT_1,
            WAYPOINT_2,
            STAGE_DURATIONS[1],
        )

        if not success:
            print("\nMOTION ABORTED AT STAGE 2.")
            return

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        success = execute_segment(
            viewer,
            "STAGE 3 — MOVE TO FINAL HOME",
            WAYPOINT_2,
            HOME,
            STAGE_DURATIONS[2],
        )

        if not success:
            print("\nMOTION ABORTED AT STAGE 3.")
            return

        # ----------------------------------------------------
        # Final hold
        # ----------------------------------------------------

        print("\n========== FINAL HOME HOLD ==========")

        command_arm_positions(HOME)

        hold_steps = int(FINAL_HOLD_TIME / TIMESTEP)

        for step in range(hold_steps):

            if not viewer.is_running():
                break

            mujoco.mj_step(model, data)

            table, self_collision = check_collisions()

            if table or self_collision:

                print("Collision during final hold.")
                print(table)
                print(self_collision)

                break

            if step % 10 == 0:
                viewer.sync()

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("FINAL MOTION REPORT")
        print("=" * 70)

        print_status("FINAL HOME", HOME)

        actual = read_arm_positions()

        final_error = np.degrees(
            np.max(np.abs(HOME - actual))
        )

        print("\nFinal maximum joint error:", f"{final_error:.3f} deg")

        if final_error < 2.0:

            print("JOINT TRACKING: PASS")

        else:

            print("JOINT TRACKING: CHECK REQUIRED")

        table, self_collision = check_collisions()

        if not table and not self_collision:
            print("COLLISION CHECK: PASS")
        else:
            print("COLLISION CHECK: FAIL")

        print("\nFinal home motion test completed.")

        while viewer.is_running():

            command_arm_positions(HOME)

            mujoco.mj_step(model, data)

            viewer.sync()

            time.sleep(TIMESTEP)


if __name__ == "__main__":
    main()