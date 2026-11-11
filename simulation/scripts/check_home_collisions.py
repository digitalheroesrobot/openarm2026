"""
OpenArm V1 home-pose collision diagnostic.

Checks:
1. Restore the calculated left/right arm home configuration.
2. Run MuJoCo forward dynamics and collision detection.
3. Report contacts involving the table and cup.
4. Report end-effector positions and orientations.

A negative contact distance indicates penetration.
A positive contact distance indicates separation.
"""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "simulation"
    / "vendor"
    / "openarm_mujoco"
    / "v1"
    / "scene_with_cup.xml"
)

HOME = {
    "openarm_left_joint1": -0.17904,
    "openarm_left_joint2": -0.12148,
    "openarm_left_joint3": -0.00198,
    "openarm_left_joint4": 0.66494,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,

    "openarm_right_joint1": 0.17904,
    "openarm_right_joint2": 0.12148,
    "openarm_right_joint3": 0.00198,
    "openarm_right_joint4": 0.66494,
    "openarm_right_joint5": 0.0,
    "openarm_right_joint6": 0.0,
    "openarm_right_joint7": 0.0,
}


def object_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name else f"unnamed_{object_id}"


def main():

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Restore the calculated home posture.
    for name, value in HOME.items():

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")

        qpos_id = model.jnt_qposadr[joint_id]
        data.qpos[qpos_id] = value

    # Calculate positions and collision contacts.
    mujoco.mj_forward(model, data)

    print("\n========== END-EFFECTOR POSE ==========")

    for side in ("left", "right"):

        site_name = f"{side}_ee_control_point"

        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        position = data.site_xpos[site_id]
        rotation = data.site_xmat[site_id].reshape(3, 3)

        print(f"\n{side.upper()}")
        print("Position (m):", np.round(position, 4))
        print("Orientation matrix:")
        print(np.round(rotation, 3))

    print("\n========== COLLISION CONTACTS ==========")
    print("Total contacts:", data.ncon)

    table_or_cup_contacts = 0
    penetrations = 0

    for i in range(data.ncon):

        contact = data.contact[i]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        name1 = object_name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom1
        )
        name2 = object_name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom2
        )

        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])

        body_name1 = object_name(
            model, mujoco.mjtObj.mjOBJ_BODY, body1
        )
        body_name2 = object_name(
            model, mujoco.mjtObj.mjOBJ_BODY, body2
        )

        involves_table_or_cup = (
            "table" in name1.lower()
            or "table" in name2.lower()
            or "cup" in name1.lower()
            or "cup" in name2.lower()
            or "table" in body_name1.lower()
            or "table" in body_name2.lower()
            or "cup" in body_name1.lower()
            or "cup" in body_name2.lower()
        )

        if involves_table_or_cup:

            table_or_cup_contacts += 1

            if contact.dist < 0:
                penetrations += 1

            print(
                f"{name1} ({body_name1}) <-> "
                f"{name2} ({body_name2}) | "
                f"distance = {contact.dist * 1000:.3f} mm"
            )

    print("\n========== SUMMARY ==========")
    print("Table/cup-related contacts:", table_or_cup_contacts)
    print("Penetrating contacts:", penetrations)

    if penetrations:
        print("RESULT: CHECK HOME POSTURE — PENETRATION DETECTED")
    else:
        print("RESULT: No detected table/cup penetration contacts")

    print("=============================")


if __name__ == "__main__":
    main()