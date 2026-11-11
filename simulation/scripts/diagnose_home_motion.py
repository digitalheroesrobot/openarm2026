#!/usr/bin/env python3

"""
OpenArm V1 — Home Motion Collision Diagnosis

Purpose:
    Identify which joint movements cause link/table collisions.

Tests:
    A. Joint 1-3 movement, with joint 4 fixed at zero.
    B. Joint 4 movement, with joints 1-3 fixed at zero.
    C. Joint 1-3 movement first, followed by joint 4 movement.

All four grippers' finger joints are opened to 44 mm.

This is a kinematic collision test, not an actuator simulation.
"""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

XML_PATH = (
    ROOT
    / "simulation/vendor/openarm_mujoco/v1/scene_with_cup.xml"
)

STEPS = 201
FINGER_OPEN = 0.044

ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

HOME = np.array([
     0.064370, -0.214334,  0.015058, 1.486495,
     0.0,      0.0,       0.0,
    -0.064370,  0.214334, -0.015058, 1.486495,
     0.0,       0.0,       0.0,
])


def main():

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    addresses = {}

    for name in ARM_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(f"Missing joint: {name}")

        addresses[name] = model.jnt_qposadr[jid]

    finger_names = [
        "openarm_left_finger_joint1",
        "openarm_left_finger_joint2",
        "openarm_right_finger_joint1",
        "openarm_right_finger_joint2",
    ]

    finger_addresses = []

    for name in finger_names:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(f"Missing joint: {name}")

        finger_addresses.append(model.jnt_qposadr[jid])

    table_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    if table_id < 0:
        raise RuntimeError("table_top not found")

    def set_configuration(q):

        for i, name in enumerate(ARM_JOINTS):
            data.qpos[addresses[name]] = q[i]

        for address in finger_addresses:
            data.qpos[address] = FINGER_OPEN

        mujoco.mj_forward(model, data)

    def inspect(q):

        set_configuration(q)

        contacts = []

        for i in range(data.ncon):

            contact = data.contact[i]

            if (
                contact.geom1 == table_id
                or contact.geom2 == table_id
            ):

                name1 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom1,
                )

                name2 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom2,
                )

                if contact.dist < -1e-6:
                    contacts.append((
                        name1,
                        name2,
                        contact.dist * 1000.0,
                    ))

        return contacts

    def scan(name, configurations):

        print("\n" + "=" * 65)
        print(name)
        print("=" * 65)

        first = None
        max_penetration = 0.0
        collision_samples = 0

        for i, q in enumerate(configurations):

            contacts = inspect(q)

            if contacts:
                collision_samples += 1

                depth = max(abs(c[2]) for c in contacts)
                max_penetration = max(max_penetration, depth)

                if first is None:
                    first = (i, q.copy(), contacts)

        print("Samples:", len(configurations))
        print("Samples with collision:", collision_samples)
        print(f"Maximum penetration: {max_penetration:.4f} mm")

        if first is None:
            print("RESULT: COLLISION-FREE")
        else:
            index, q, contacts = first

            print("RESULT: COLLISION DETECTED")
            print("First collision sample:", index)
            print(
                "Progress:",
                f"{index / (len(configurations) - 1) * 100:.2f}%",
            )

            print("\nJoint configuration:")

            for joint, value in zip(ARM_JOINTS, q):
                print(f"{joint}: {value:.6f} rad")

            print("\nFirst collision contacts:")

            for name1, name2, distance in contacts:
                print(
                    f"{name1} <-> {name2}: "
                    f"{distance:.4f} mm"
                )

        return first is None

    zero = np.zeros(14)

    # A. Move joints 1-3 while joint 4 remains zero.
    stage_a = []

    for alpha in np.linspace(0.0, 1.0, STEPS):

        q = zero.copy()

        q[[0, 1, 2, 7, 8, 9]] = (
            HOME[[0, 1, 2, 7, 8, 9]] * alpha
        )

        stage_a.append(q)

    # B. Move joint 4 independently.
    stage_b = []

    for alpha in np.linspace(0.0, 1.0, STEPS):

        q = zero.copy()

        q[[3, 10]] = HOME[[3, 10]] * alpha

        stage_b.append(q)

    # C. First joints 1-3, then joints 4.
    stage_c1 = []

    for alpha in np.linspace(0.0, 1.0, STEPS):

        q = zero.copy()

        q[[0, 1, 2, 7, 8, 9]] = (
            HOME[[0, 1, 2, 7, 8, 9]] * alpha
        )

        stage_c1.append(q)

    stage_c2 = []

    for alpha in np.linspace(0.0, 1.0, STEPS):

        q = HOME.copy()

        q[[3, 10]] = HOME[[3, 10]] * alpha

        stage_c2.append(q)

    # Execute diagnostic scans.
    result_a = scan(
        "TEST A: JOINTS 1-3 FIRST",
        stage_a,
    )

    result_b = scan(
        "TEST B: JOINT 4 ONLY",
        stage_b,
    )

    result_c1 = scan(
        "TEST C1: JOINTS 1-3 BEFORE JOINT 4",
        stage_c1,
    )

    result_c2 = scan(
        "TEST C2: JOINT 4 AFTER JOINTS 1-3",
        stage_c2,
    )

    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)

    print("A. Joints 1-3:", "PASS" if result_a else "FAIL")
    print("B. Joint 4 only:", "PASS" if result_b else "FAIL")
    print("C1. Joints 1-3 first:", "PASS" if result_c1 else "FAIL")
    print("C2. Joint 4 second:", "PASS" if result_c2 else "FAIL")

    if result_a and result_c2:
        print("\nA two-stage collision-free path was found.")

    print("\nDiagnosis completed.")


if __name__ == "__main__":
    main()