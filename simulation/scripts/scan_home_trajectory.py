#!/usr/bin/env python3

"""
OpenArm V1 — Home Trajectory Collision Scanner

Purpose:
    Determine whether linear joint-space interpolation from the
    initial configuration to the custom home configuration is safe.

Tests:
    1. Initial configuration.
    2. Final custom home configuration.
    3. 201 intermediate configurations.
    4. Table collisions and robot self-collisions.

Important:
    - Uses the original MuJoCo model.
    - Does not modify actuator gains or force limits.
    - Opens both grippers to 44 mm before scanning.
    - Does not modify any XML or README files.
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

NUM_STEPS = 201

FINGER_OPEN = 0.044

HOME = {
    "openarm_left_joint1": 0.064370,
    "openarm_left_joint2": -0.214334,
    "openarm_left_joint3": 0.015058,
    "openarm_left_joint4": 1.486495,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,

    "openarm_right_joint1": -0.064370,
    "openarm_right_joint2": 0.214334,
    "openarm_right_joint3": -0.015058,
    "openarm_right_joint4": 1.486495,
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
}


# ============================================================
# MODEL
# ============================================================

def get_joint_address(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")

    return model.jnt_qposadr[jid]


def get_geom_name(model, geom_id):
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )


def main():

    print("=" * 65)
    print("OPENARM V1 HOME TRAJECTORY COLLISION SCANNER")
    print("=" * 65)

    print("\nLoading:", XML_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Joint addresses
    # --------------------------------------------------------

    joint_names = list(HOME.keys())

    joint_addresses = {
        name: get_joint_address(model, name)
        for name in joint_names
    }

    left_finger_addr = get_joint_address(
        model,
        "openarm_left_finger_joint1",
    )

    left_finger2_addr = get_joint_address(
        model,
        "openarm_left_finger_joint2",
    )

    right_finger_addr = get_joint_address(
        model,
        "openarm_right_finger_joint1",
    )

    right_finger2_addr = get_joint_address(
        model,
        "openarm_right_finger_joint2",
    )

    finger_addresses = [
        left_finger_addr,
        left_finger2_addr,
        right_finger_addr,
        right_finger2_addr,
    ]

    # --------------------------------------------------------
    # Initial and target configurations
    # --------------------------------------------------------

    start = np.zeros(len(joint_names))

    target = np.array([
        HOME[name] for name in joint_names
    ])

    # Validate joint limits.
    for name, value in HOME.items():

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        lower, upper = model.jnt_range[jid]

        if not lower <= value <= upper:
            raise ValueError(
                f"{name}: target {value:.6f} "
                f"outside [{lower:.6f}, {upper:.6f}]"
            )

    print("\nInitial arm configuration:")
    print(start)

    print("\nTarget arm configuration:")
    print(target)

    # --------------------------------------------------------
    # Geometry identification
    # --------------------------------------------------------

    table_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    if table_geom_id < 0:
        raise RuntimeError("table_top geometry not found")

    print("\nTable geometry ID:", table_geom_id)

    # --------------------------------------------------------
    # Collision inspection
    # --------------------------------------------------------

    def inspect_configuration(q):

        # Set arm joints.
        for i, name in enumerate(joint_names):
            data.qpos[joint_addresses[name]] = q[i]

        # Open all fingers.
        for address in finger_addresses:
            data.qpos[address] = FINGER_OPEN

        mujoco.mj_forward(model, data)

        table_contacts = []
        self_contacts = []
        other_contacts = []

        for i in range(data.ncon):

            contact = data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            name1 = get_geom_name(model, geom1)
            name2 = get_geom_name(model, geom2)

            # Only report actual penetration beyond 0.001 mm.
            if contact.dist >= -1e-6:
                continue

            info = (
                name1,
                name2,
                contact.dist * 1000.0,
            )

            if (
                geom1 == table_geom_id
                or geom2 == table_geom_id
            ):
                table_contacts.append(info)

            elif (
                name1 is not None
                and name2 is not None
                and name1.startswith("openarm_")
                and name2.startswith("openarm_")
            ):
                self_contacts.append(info)

            else:
                other_contacts.append(info)

        return table_contacts, self_contacts, other_contacts

    # --------------------------------------------------------
    # Initial configuration
    # --------------------------------------------------------

    print("\n========== INITIAL CONFIGURATION ==========")

    initial = inspect_configuration(start)

    print("Table penetrations:", len(initial[0]))
    print("Self penetrations:", len(initial[1]))
    print("Other penetrations:", len(initial[2]))

    # --------------------------------------------------------
    # Target configuration
    # --------------------------------------------------------

    print("\n========== TARGET CONFIGURATION ==========")

    final = inspect_configuration(target)

    print("Table penetrations:", len(final[0]))
    print("Self penetrations:", len(final[1]))
    print("Other penetrations:", len(final[2]))

    # --------------------------------------------------------
    # Linear interpolation scan
    # --------------------------------------------------------

    print("\n========== TRAJECTORY SCAN ==========")

    first_collision = None

    maximum_table_penetration = 0.0
    maximum_self_penetration = 0.0

    for step in range(NUM_STEPS):

        alpha = step / (NUM_STEPS - 1)

        q = start + alpha * (target - start)

        table, self_collision, other = (
            inspect_configuration(q)
        )

        table_depth = max(
            [abs(item[2]) for item in table],
            default=0.0,
        )

        self_depth = max(
            [abs(item[2]) for item in self_collision],
            default=0.0,
        )

        maximum_table_penetration = max(
            maximum_table_penetration,
            table_depth,
        )

        maximum_self_penetration = max(
            maximum_self_penetration,
            self_depth,
        )

        collision = bool(table or self_collision or other)

        if collision and first_collision is None:

            first_collision = {
                "step": step,
                "alpha": alpha,
                "q": q.copy(),
                "table": table,
                "self": self_collision,
                "other": other,
            }

        if step % 20 == 0:

            print(
                f"Step {step:3d}/{NUM_STEPS - 1} "
                f"| progress: {alpha * 100:6.2f}% "
                f"| table: {len(table)} "
                f"| self: {len(self_collision)}"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n========== SCAN RESULTS ==========")

    print(
        "Maximum table penetration:",
        f"{maximum_table_penetration:.4f} mm",
    )

    print(
        "Maximum self penetration:",
        f"{maximum_self_penetration:.4f} mm",
    )

    if first_collision is None:

        print("\nRESULT: LINEAR PATH IS COLLISION-FREE")

    else:

        print("\nRESULT: LINEAR PATH IS NOT COLLISION-FREE")

        print(
            "First collision at step:",
            first_collision["step"],
        )

        print(
            "Trajectory progress:",
            f"{first_collision['alpha'] * 100:.2f}%",
        )

        print("\nJoint configuration:")

        for name, value in zip(
            joint_names,
            first_collision["q"],
        ):
            print(f"{name}: {value:.6f} rad")

        for label in ("table", "self", "other"):

            contacts = first_collision[label]

            if contacts:

                print(f"\n{label.upper()} COLLISIONS:")

                for name1, name2, distance in contacts:

                    print(
                        f"{name1} <-> {name2}: "
                        f"{distance:.4f} mm"
                    )

    print("\n========== FINAL END-EFFECTOR POSITIONS ==========")

    for name in (
        "left_ee_control_point",
        "right_ee_control_point",
    ):

        sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        print(name, data.site_xpos[sid])

    print("\nCollision scan completed.")


if __name__ == "__main__":
    main()