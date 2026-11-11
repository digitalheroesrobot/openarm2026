
#!/usr/bin/env python3

"""
OPENARM V1 — VISUAL HOME POSE VERIFICATION V3

Purpose:
    Execute and visually verify the complete OpenArm V1
    initial-to-home trajectory.

Motion sequence:
    1. Initialize both arms with grippers CLOSED.
    2. Move to temporary waypoint.
    3. Raise both arms by moving joint 4.
    4. Move both arms to final home configuration.
    5. Stabilize the arms at home while keeping grippers CLOSED.
    6. Open both grippers to 44 mm.
    7. Hold the final home position with grippers OPEN.

Important:
    Grippers must remain CLOSED until both arms have
    reached and stabilized at the final home position.

Safety:
    - Original actuator gains and force limits unchanged.
    - Collision checking at every simulation step.
    - Robot self-collision monitoring.
    - Robot-environment collision monitoring.
    - Viewer-based visualization.

This script does not modify the original MuJoCo XML.
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

DT = 0.001

SETTLE_TIME = 1.0

STAGE1_DURATION = 4.0
STAGE2_DURATION = 6.0
STAGE3_DURATION = 4.0

HOME_STABILIZATION_TIME = 2.0

GRIPPER_OPEN_DURATION = 1.5

FINGER_CLOSED = 0.0
FINGER_OPEN = 0.044

COLLISION_THRESHOLD = -1e-5

VIEWER_SYNC_INTERVAL = 10


# ============================================================
# JOINT CONFIGURATIONS
# ============================================================

JOINT_NAMES = [
    f"openarm_left_joint{i}" for i in range(1, 8)
] + [
    f"openarm_right_joint{i}" for i in range(1, 8)
]

# Initial arm configuration.
START = np.zeros(14)


# Stage 1: temporary collision-free waypoint.
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


# Stage 3: final home configuration.
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
# GRIPPER CONFIGURATION
# ============================================================

FINGER_NAMES = [
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2",
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
]

FINGER_ACTUATORS = [
    "left_finger1_ctrl",
    "right_finger1_ctrl",
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


def geom_name(model, geom_id):

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
    print("OPENARM V1 — STABLE HOME POSITION VERIFICATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------

    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"Scene not found: {XML_PATH}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    model.opt.timestep = DT

    data = mujoco.MjData(model)

    print("\nLoaded scene:")
    print(XML_PATH)

    print(f"\nnq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")

    # --------------------------------------------------------
    # Resolve arm joints and actuators
    # --------------------------------------------------------

    qpos_addresses = {}
    actuator_ids = {}

    for name in JOINT_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_addresses[name] = int(
            model.jnt_qposadr[jid]
        )

        aid = -1

        for i in range(model.nu):

            if int(model.actuator_trnid[i, 0]) == jid:
                aid = i
                break

        if aid < 0:
            raise RuntimeError(
                f"Missing actuator for joint: {name}"
            )

        actuator_ids[name] = aid

    # --------------------------------------------------------
    # Resolve finger joints and actuators
    # --------------------------------------------------------

    finger_addresses = []

    for name in FINGER_NAMES:

        jid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        finger_addresses.append(
            int(model.jnt_qposadr[jid])
        )

    finger_actuator_ids = []

    for name in FINGER_ACTUATORS:

        aid = object_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        finger_actuator_ids.append(aid)

    # --------------------------------------------------------
    # Identify robot collision geometries
    # --------------------------------------------------------

    robot_collision_ids = set()

    for gid in range(model.ngeom):

        name = geom_name(model, gid)

        if (
            name.startswith("openarm_")
            and name.endswith("_collision")
        ):
            robot_collision_ids.add(gid)

    if not robot_collision_ids:
        raise RuntimeError(
            "No OpenArm collision geometries found."
        )

    print(
        "\nRobot collision geometries:",
        len(robot_collision_ids),
    )

    # --------------------------------------------------------
    # Initialize robot
    # --------------------------------------------------------

    data.qpos[:] = model.qpos0

    # Initial arm configuration.
    for name in JOINT_NAMES:

        data.qpos[
            qpos_addresses[name]
        ] = 0.0

    # IMPORTANT:
    # Both grippers start fully CLOSED.
    for address in finger_addresses:

        data.qpos[address] = FINGER_CLOSED

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Position and gripper command functions
    # --------------------------------------------------------

    def actual_positions():

        return np.array([
            data.qpos[qpos_addresses[name]]
            for name in JOINT_NAMES
        ])

    def command_positions(target):

        for i, name in enumerate(JOINT_NAMES):

            data.ctrl[
                actuator_ids[name]
            ] = target[i]

    def command_grippers(position):

        for aid in finger_actuator_ids:

            data.ctrl[aid] = position

    # Initialize actuator targets.
    command_positions(START)
    command_grippers(FINGER_CLOSED)

    # --------------------------------------------------------
    # Collision monitoring
    # --------------------------------------------------------

    def check_collisions():

        collisions = []

        for i in range(data.ncon):

            contact = data.contact[i]

            g1 = contact.geom1
            g2 = contact.geom2

            robot_contact = (
                g1 in robot_collision_ids
                or g2 in robot_collision_ids
            )

            if not robot_contact:
                continue

            if contact.dist < COLLISION_THRESHOLD:

                collisions.append({
                    "geom1": geom_name(model, g1),
                    "geom2": geom_name(model, g2),
                    "distance_mm": contact.dist * 1000.0,
                })

        return collisions

    def report_collisions(collisions):

        for item in collisions:

            print(
                f"  {item['geom1']} <-> "
                f"{item['geom2']} | "
                f"{item['distance_mm']:.4f} mm"
            )

    # --------------------------------------------------------
    # Arm trajectory stage
    # --------------------------------------------------------

    def execute_arm_stage(
        viewer,
        name,
        start,
        target,
        duration,
    ):

        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)

        steps = int(round(duration / DT))

        for step in range(steps + 1):

            if not viewer.is_running():
                return False

            alpha = step / steps

            blend = smoothstep(alpha)

            command = (
                start
                + blend * (target - start)
            )

            command_positions(command)

            # Grippers MUST remain closed.
            command_grippers(FINGER_CLOSED)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DETECTED!")

                report_collisions(collisions)

                print("\nActual joint configuration:")

                for name, value in zip(
                    JOINT_NAMES,
                    actual_positions(),
                ):

                    print(
                        f"  {name}: {value:.8f}"
                    )

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:

                viewer.sync()

            if step % int(1.0 / DT) == 0:

                error = np.degrees(
                    np.max(
                        np.abs(
                            command - actual_positions()
                        )
                    )
                )

                print(
                    f"  t={data.time:.2f}s | "
                    f"progress={alpha * 100:.1f}% | "
                    f"tracking error={error:.3f} deg | "
                    f"grippers=CLOSED"
                )

            time.sleep(DT)

        viewer.sync()

        print(f"\n{name}: PASS")

        return True

    # --------------------------------------------------------
    # Stabilization at home
    # --------------------------------------------------------

    def stabilize_home(viewer):

        print("\n" + "=" * 72)
        print("HOME STABILIZATION — GRIPPERS CLOSED")
        print("=" * 72)

        steps = int(
            round(HOME_STABILIZATION_TIME / DT)
        )

        for step in range(steps):

            if not viewer.is_running():
                return False

            # Maintain final home position.
            command_positions(HOME)

            # Keep grippers closed during stabilization.
            command_grippers(FINGER_CLOSED)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DURING STABILIZATION!")

                report_collisions(collisions)

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:

                viewer.sync()

            if step % int(0.5 / DT) == 0:

                error = np.degrees(
                    np.max(
                        np.abs(
                            HOME - actual_positions()
                        )
                    )
                )

                print(
                    f"  Stabilizing: "
                    f"{step * DT:.1f}/"
                    f"{HOME_STABILIZATION_TIME:.1f}s | "
                    f"tracking error={error:.3f} deg | "
                    f"grippers=CLOSED"
                )

            time.sleep(DT)

        viewer.sync()

        print("\nHOME STABILIZATION: PASS")
        print("Arms have been held at home.")
        print("Both grippers remain CLOSED.")

        return True

    # --------------------------------------------------------
    # Final gripper-opening stage
    # --------------------------------------------------------

    def execute_gripper_open_stage(viewer):

        print("\n" + "=" * 72)
        print("FINAL STAGE — OPEN BOTH GRIPPERS")
        print("=" * 72)

        steps = int(
            round(GRIPPER_OPEN_DURATION / DT)
        )

        for step in range(steps + 1):

            if not viewer.is_running():
                return False

            alpha = step / steps

            blend = smoothstep(alpha)

            finger_target = (
                FINGER_CLOSED
                + blend * (
                    FINGER_OPEN - FINGER_CLOSED
                )
            )

            # Arms remain stationary at final home.
            command_positions(HOME)

            # Open both grippers simultaneously.
            command_grippers(finger_target)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION WHILE OPENING GRIPPERS!")

                report_collisions(collisions)

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:

                viewer.sync()

            if step % int(0.25 / DT) == 0:

                print(
                    f"  Gripper target: "
                    f"{finger_target * 1000:.1f} mm"
                )

            time.sleep(DT)

        viewer.sync()

        print("\nGRIPPER OPENING: PASS")
        print("Both grippers commanded to 44 mm.")

        return True

    # ========================================================
    # LAUNCH VIEWER
    # ========================================================

    print("\nLaunching MuJoCo viewer...")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        viewer.cam.lookat[:] = [
            0.35,
            0.0,
            0.35,
        ]

        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        viewer.sync()

        print("\nViewer started.")
        print("Initial gripper state: CLOSED")

        # ----------------------------------------------------
        # Initial settling
        # ----------------------------------------------------

        print("\nSettling initial configuration...")

        for _ in range(
            int(round(SETTLE_TIME / DT))
        ):

            if not viewer.is_running():
                return

            command_positions(START)

            command_grippers(FINGER_CLOSED)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DURING INITIAL SETTLING!")

                report_collisions(collisions)

                return

            viewer.sync()

            time.sleep(DT)

        print("Initial settling completed.")

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        if not execute_arm_stage(
            viewer,
            "STAGE 1 — INITIAL TO TEMPORARY POSTURE",
            START,
            WAYPOINT_1,
            STAGE1_DURATION,
        ):

            print("\nTRAJECTORY ABORTED.")
            return

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        if not execute_arm_stage(
            viewer,
            "STAGE 2 — RAISE JOINT 4",
            WAYPOINT_1,
            WAYPOINT_2,
            STAGE2_DURATION,
        ):

            print("\nTRAJECTORY ABORTED.")
            return

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        if not execute_arm_stage(
            viewer,
            "STAGE 3 — FINAL HOME ADJUSTMENT",
            WAYPOINT_2,
            HOME,
            STAGE3_DURATION,
        ):

            print("\nTRAJECTORY ABORTED.")
            return

        # ----------------------------------------------------
        # Stabilization
        # ----------------------------------------------------

        if not stabilize_home(viewer):

            print("\nHOME STABILIZATION FAILED.")
            return

        # ----------------------------------------------------
        # Final gripper opening
        # ----------------------------------------------------

        if not execute_gripper_open_stage(viewer):

            print("\nGRIPPER OPENING FAILED.")
            return

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------

        print("\n" + "=" * 72)
        print("FINAL HOME VERIFICATION")
        print("=" * 72)

        actual = actual_positions()

        error = np.degrees(
            np.max(
                np.abs(HOME - actual)
            )
        )

        print(
            f"\nMaximum arm tracking error: "
            f"{error:.4f} deg"
        )

        print("\nFinal arm configuration:")

        for name, value in zip(
            JOINT_NAMES,
            actual,
        ):

            print(
                f"  {name}: {value:.6f}"
            )

        print("\nFinal finger positions:")

        for name, address in zip(
            FINGER_NAMES,
            finger_addresses,
        ):

            print(
                f"  {name}: "
                f"{data.qpos[address] * 1000:.2f} mm"
            )

        collisions = check_collisions()

        if collisions:

            print("\nFINAL COLLISION CHECK: FAIL")

            report_collisions(collisions)

            return

        print("\nFINAL COLLISION CHECK: PASS")

        print("\n" + "=" * 72)
        print("ALL MOTION STAGES COMPLETED SUCCESSFULLY")
        print("=" * 72)

        print("""
Final state:
    Left arm:       HOME
    Right arm:      HOME
    Left gripper:   OPEN
    Right gripper:  OPEN

The robot will maintain this configuration.
Close the MuJoCo viewer when finished.
""")

        # ----------------------------------------------------
        # Hold final home configuration
        # ----------------------------------------------------

        while viewer.is_running():

            command_positions(HOME)

            command_grippers(FINGER_OPEN)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION WHILE HOLDING HOME!")

                report_collisions(collisions)

                break

            viewer.sync()

            time.sleep(DT)


if __name__ == "__main__":
    main()
