
#!/usr/bin/env python3

"""
OPENARM V1 — STABLE HOME POSITION VERIFICATION V4

Motion sequence:
    1. Automatically calibrate a collision-free closed-gripper position.
    2. Initialize both grippers at that position.
    3. Move from the initial position to a temporary waypoint.
    4. Raise both arms.
    5. Move to the final home configuration.
    6. Stabilize for 2 seconds with grippers closed.
    7. Open both grippers to 44 mm.
    8. Hold the final home configuration.

Safety:
    - Does not modify the original MuJoCo XML.
    - Does not change actuator gains or force limits.
    - Checks robot collisions at every simulation step.
    - Automatically avoids the known V1 finger-mesh overlap
      at the fully closed 0 mm position.
    - Keeps the viewer open if a collision aborts the trajectory.

Note:
    The operationally closed position is the smallest opening
    that removes the known finger-to-finger mesh overlap.
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

FINGER_OPEN = 0.044

# Gripper calibration resolution: 0.25 mm.
GRIPPER_SEARCH_INCREMENT = 0.00025

COLLISION_THRESHOLD = -1e-5

VIEWER_SYNC_INTERVAL = 10


# ============================================================
# ARM CONFIGURATIONS
# ============================================================

JOINT_NAMES = (
    [f"openarm_left_joint{i}" for i in range(1, 8)]
    + [f"openarm_right_joint{i}" for i in range(1, 8)]
)

START = np.zeros(14)

WAYPOINT_1 = np.array([
    -0.035630,
    -0.874334,
     0.015058,
     0.0, 0.0, 0.0, 0.0,

     0.035630,
     0.874334,
    -0.015058,
     0.0, 0.0, 0.0, 0.0,
])

WAYPOINT_2 = WAYPOINT_1.copy()
WAYPOINT_2[3] = 1.486495
WAYPOINT_2[10] = 1.486495

HOME = np.array([
     0.064370,
    -0.214334,
     0.015058,
     1.486495,
     0.0, 0.0, 0.0,

    -0.064370,
     0.214334,
    -0.015058,
     1.486495,
     0.0, 0.0, 0.0,
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

FINGER_ACTUATOR_NAMES = [
    "left_finger1_ctrl",
    "right_finger1_ctrl",
]

FINGER_COLLISION_PAIRS = [
    (
        "openarm_left_right_finger_collision",
        "openarm_left_left_finger_collision",
    ),
    (
        "openarm_right_right_finger_collision",
        "openarm_right_left_finger_collision",
    ),
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
        raise RuntimeError(
            f"MuJoCo object not found: {name}"
        )

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
    print("OPENARM V1 — STABLE HOME POSITION VERIFICATION V4")
    print("=" * 72)

    if not XML_PATH.exists():
        raise FileNotFoundError(XML_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    model.opt.timestep = DT

    data = mujoco.MjData(model)

    print(f"\nLoaded scene: {XML_PATH}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")

    # --------------------------------------------------------
    # Resolve arm joint addresses and actuator IDs
    # --------------------------------------------------------

    arm_qpos = {}
    arm_actuators = {}

    for name in JOINT_NAMES:

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        arm_qpos[name] = int(
            model.jnt_qposadr[jid]
        )

        aid = -1

        for i in range(model.nu):

            if int(model.actuator_trnid[i, 0]) == jid:
                aid = i
                break

        if aid < 0:
            raise RuntimeError(
                f"No actuator found for {name}"
            )

        arm_actuators[name] = aid

    # --------------------------------------------------------
    # Resolve gripper joints and actuators
    # --------------------------------------------------------

    finger_qpos = []

    for name in FINGER_NAMES:

        jid = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        finger_qpos.append(
            int(model.jnt_qposadr[jid])
        )

    finger_actuators = [
        get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in FINGER_ACTUATOR_NAMES
    ]

    # --------------------------------------------------------
    # Identify robot collision geometries
    # --------------------------------------------------------

    robot_collision_ids = set()

    for gid in range(model.ngeom):

        name = get_geom_name(model, gid)

        if (
            name.startswith("openarm_")
            and name.endswith("_collision")
        ):
            robot_collision_ids.add(gid)

    print(
        "\nRobot collision geometries:",
        len(robot_collision_ids),
    )

    # --------------------------------------------------------
    # Resolve known finger collision pairs
    # --------------------------------------------------------

    finger_pair_ids = []

    for name1, name2 in FINGER_COLLISION_PAIRS:

        gid1 = get_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name1,
        )

        gid2 = get_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name2,
        )

        finger_pair_ids.append(
            frozenset((gid1, gid2))
        )

    # --------------------------------------------------------
    # Collision-free closed-gripper calibration
    # --------------------------------------------------------

    def finger_pair_penetrations():

        penetrations = []

        for i in range(data.ncon):

            contact = data.contact[i]

            pair = frozenset((
                contact.geom1,
                contact.geom2,
            ))

            if pair in finger_pair_ids:

                if contact.dist < COLLISION_THRESHOLD:

                    penetrations.append(
                        contact.dist
                    )

        return penetrations

    def find_safe_closed_position():

        print("\n" + "=" * 72)
        print("AUTOMATIC GRIPPER CALIBRATION")
        print("=" * 72)

        print(
            "Searching for the smallest collision-free "
            "gripper opening..."
        )

        candidate = 0.0

        while candidate <= FINGER_OPEN + 1e-12:

            for address in finger_qpos:
                data.qpos[address] = candidate

            for aid in finger_actuators:
                data.ctrl[aid] = candidate

            mujoco.mj_forward(model, data)

            penetrations = finger_pair_penetrations()

            if not penetrations:

                print(
                    "\nCollision-free closed position found:"
                )

                print(
                    f"  {candidate * 1000:.2f} mm"
                )

                return candidate

            candidate += GRIPPER_SEARCH_INCREMENT

        raise RuntimeError(
            "Could not find a collision-free gripper position."
        )

    # Initialize all arm joints at zero.
    data.qpos[:] = model.qpos0

    for name in JOINT_NAMES:
        data.qpos[arm_qpos[name]] = 0.0

    mujoco.mj_forward(model, data)

    CLOSED_POSITION = find_safe_closed_position()

    # Reset initial state using calibrated closed position.
    for name in JOINT_NAMES:
        data.qpos[arm_qpos[name]] = 0.0

    for address in finger_qpos:
        data.qpos[address] = CLOSED_POSITION

    mujoco.mj_forward(model, data)

    # --------------------------------------------------------
    # Command functions
    # --------------------------------------------------------

    def actual_arm_positions():

        return np.array([
            data.qpos[arm_qpos[name]]
            for name in JOINT_NAMES
        ])

    def command_arms(target):

        for i, name in enumerate(JOINT_NAMES):

            data.ctrl[arm_actuators[name]] = target[i]

    def command_grippers(position):

        for aid in finger_actuators:
            data.ctrl[aid] = position

    # --------------------------------------------------------
    # Collision monitoring
    # --------------------------------------------------------

    def check_collisions():

        collisions = []

        for i in range(data.ncon):

            contact = data.contact[i]

            g1 = contact.geom1
            g2 = contact.geom2

            if (
                g1 not in robot_collision_ids
                and g2 not in robot_collision_ids
            ):
                continue

            if contact.dist < COLLISION_THRESHOLD:

                collisions.append({
                    "geom1": get_geom_name(model, g1),
                    "geom2": get_geom_name(model, g2),
                    "distance_mm": contact.dist * 1000.0,
                })

        return collisions

    def print_collisions(collisions):

        for item in collisions:

            print(
                f"  {item['geom1']} <-> "
                f"{item['geom2']} | "
                f"{item['distance_mm']:.4f} mm"
            )

    # --------------------------------------------------------
    # Execute an arm trajectory stage
    # --------------------------------------------------------

    def execute_arm_stage(
        viewer,
        title,
        start,
        target,
        duration,
    ):

        print("\n" + "=" * 72)
        print(title)
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

            command_arms(command)

            # Keep both grippers operationally closed.
            command_grippers(CLOSED_POSITION)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DETECTED!")
                print_collisions(collisions)

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()

            if step % int(1.0 / DT) == 0:

                error = np.degrees(
                    np.max(
                        np.abs(
                            command - actual_arm_positions()
                        )
                    )
                )

                print(
                    f"  t={step * DT:.1f}s | "
                    f"progress={alpha * 100:.1f}% | "
                    f"tracking error={error:.3f} deg | "
                    f"grippers=CLOSED"
                )

            time.sleep(DT)

        viewer.sync()

        print(f"\n{title}: PASS")

        return True

    # --------------------------------------------------------
    # Stabilize at final home position
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

            command_arms(HOME)
            command_grippers(CLOSED_POSITION)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DURING STABILIZATION!")
                print_collisions(collisions)

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()

            if step % int(0.5 / DT) == 0:

                error = np.degrees(
                    np.max(
                        np.abs(
                            HOME - actual_arm_positions()
                        )
                    )
                )

                print(
                    f"  Stabilizing: {step * DT:.1f}/"
                    f"{HOME_STABILIZATION_TIME:.1f}s | "
                    f"error={error:.3f} deg | "
                    f"grippers=CLOSED"
                )

            time.sleep(DT)

        print("\nHOME STABILIZATION: PASS")

        return True

    # --------------------------------------------------------
    # Open both grippers after stabilization
    # --------------------------------------------------------

    def open_grippers(viewer):

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
                CLOSED_POSITION
                + blend * (
                    FINGER_OPEN - CLOSED_POSITION
                )
            )

            # Arms remain at home.
            command_arms(HOME)

            command_grippers(finger_target)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION WHILE OPENING!")
                print_collisions(collisions)

                return False

            if step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()

            if step % int(0.25 / DT) == 0:

                print(
                    f"  Gripper target: "
                    f"{finger_target * 1000:.1f} mm"
                )

            time.sleep(DT)

        print("\nGRIPPER OPENING: PASS")

        return True

    # ========================================================
    # VIEWER
    # ========================================================

    print("\nLaunching MuJoCo viewer...")

    success = False

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

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

        print(
            f"Initial gripper state: "
            f"CLOSED ({CLOSED_POSITION * 1000:.2f} mm)"
        )

        # ----------------------------------------------------
        # Initial settling
        # ----------------------------------------------------

        print("\nSettling initial configuration...")

        initial_steps = int(
            round(SETTLE_TIME / DT)
        )

        for step in range(initial_steps):

            if not viewer.is_running():
                break

            command_arms(START)
            command_grippers(CLOSED_POSITION)

            mujoco.mj_step(model, data)

            collisions = check_collisions()

            if collisions:

                print("\nCOLLISION DURING INITIAL SETTLING!")
                print_collisions(collisions)

                break

            if step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()

            time.sleep(DT)

        else:

            print("Initial settling: PASS")

            # ------------------------------------------------
            # Stage 1
            # ------------------------------------------------

            success = execute_arm_stage(
                viewer,
                "STAGE 1 — INITIAL TO TEMPORARY POSTURE",
                START,
                WAYPOINT_1,
                STAGE1_DURATION,
            )

            # ------------------------------------------------
            # Stage 2
            # ------------------------------------------------

            if success:

                success = execute_arm_stage(
                    viewer,
                    "STAGE 2 — RAISE JOINT 4",
                    WAYPOINT_1,
                    WAYPOINT_2,
                    STAGE2_DURATION,
                )

            # ------------------------------------------------
            # Stage 3
            # ------------------------------------------------

            if success:

                success = execute_arm_stage(
                    viewer,
                    "STAGE 3 — FINAL HOME ADJUSTMENT",
                    WAYPOINT_2,
                    HOME,
                    STAGE3_DURATION,
                )

            # ------------------------------------------------
            # Stabilization
            # ------------------------------------------------

            if success:
                success = stabilize_home(viewer)

            # ------------------------------------------------
            # Gripper opening
            # ------------------------------------------------

            if success:
                success = open_grippers(viewer)

        # ----------------------------------------------------
        # Final verification
        # ----------------------------------------------------

        if success:

            print("\n" + "=" * 72)
            print("FINAL HOME VERIFICATION")
            print("=" * 72)

            actual = actual_arm_positions()

            error = np.degrees(
                np.max(
                    np.abs(HOME - actual)
                )
            )

            print(
                f"\nMaximum arm tracking error: "
                f"{error:.4f} deg"
            )

            print("\nFinal finger positions:")

            for name, address in zip(
                FINGER_NAMES,
                finger_qpos,
            ):

                print(
                    f"  {name}: "
                    f"{data.qpos[address] * 1000:.2f} mm"
                )

            collisions = check_collisions()

            if collisions:

                print("\nFINAL COLLISION CHECK: FAIL")
                print_collisions(collisions)

                success = False

            else:

                print("\nFINAL COLLISION CHECK: PASS")

        if success:

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
Close the viewer when finished.
""")

            # Hold final configuration.
            while viewer.is_running():

                command_arms(HOME)
                command_grippers(FINGER_OPEN)

                mujoco.mj_step(model, data)

                collisions = check_collisions()

                if collisions:

                    print("\nCOLLISION WHILE HOLDING HOME!")
                    print_collisions(collisions)

                    break

                viewer.sync()
                time.sleep(DT)

        else:

            print("\nTRAJECTORY ABORTED.")
            print(
                "The viewer will remain open so you can "
                "inspect the robot state."
            )

            # Avoid immediately destroying the viewer after
            # an early collision abort.
            while viewer.is_running():

                viewer.sync()
                time.sleep(0.03)


if __name__ == "__main__":
    main()
