
#!/usr/bin/env python3

"""
OPENARM V1 — CUP PICKUP GEOMETRY INSPECTION

Milestone 1:
    Inspect the MuJoCo scene before developing cup pickup.

Objectives:
    1. Inspect cup body, free joint, and collision geometry.
    2. Inspect table dimensions and position.
    3. Identify robot end-effector sites.
    4. Identify gripper collision geometries.
    5. Inspect arm joint and actuator configurations.
    6. Display the scene in the MuJoCo viewer.

Expected cup dimensions:
    Top diameter:    75 mm
    Bottom diameter: 50 mm
    Height:          85 mm

This script does not modify the XML or move the robot.
"""

from pathlib import Path

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


# ============================================================
# HELPERS
# ============================================================

def get_name(model, object_type, object_id):

    name = mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )

    return name or f"unnamed_{object_id}"


def print_section(title):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def geom_type_name(geom_type):

    names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "PLANE",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "HEIGHTFIELD",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "SPHERE",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "CAPSULE",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ELLIPSOID",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "CYLINDER",
        int(mujoco.mjtGeom.mjGEOM_BOX): "BOX",
        int(mujoco.mjtGeom.mjGEOM_MESH): "MESH",
    }

    return names.get(int(geom_type), str(geom_type))


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("OPENARM V1 — CUP PICKUP GEOMETRY INSPECTION")

    if not XML_PATH.exists():
        raise FileNotFoundError(XML_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print(f"\nScene: {XML_PATH}")

    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"nbody: {model.nbody}")
    print(f"ngeom: {model.ngeom}")
    print(f"nsite: {model.nsite}")

    # --------------------------------------------------------
    # 1. BODY INSPECTION
    # --------------------------------------------------------

    print_section("1. BODY INSPECTION")

    for bid in range(model.nbody):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        )

        if any(
            key in name.lower()
            for key in ("cup", "table", "openarm")
        ):

            print(
                f"Body {bid:3d}: {name}"
            )

            print(
                "    World position:",
                np.round(data.xpos[bid], 6),
            )

    # --------------------------------------------------------
    # 2. CUP GEOMETRY
    # --------------------------------------------------------

    print_section("2. CUP GEOMETRY")

    cup_found = False

    for gid in range(model.ngeom):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        body_id = int(model.geom_bodyid[gid])

        body_name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if "cup" not in name.lower() and "cup" not in body_name.lower():
            continue

        cup_found = True

        print(f"\nGeom ID: {gid}")
        print(f"Geom name: {name}")
        print(f"Body: {body_name}")

        print(
            "Geom type:",
            geom_type_name(model.geom_type[gid]),
        )

        print(
            "Local position:",
            np.round(model.geom_pos[gid], 6),
        )

        print(
            "World position:",
            np.round(data.geom_xpos[gid], 6),
        )

        print(
            "Geom size parameters:",
            np.round(model.geom_size[gid], 6),
        )

        print(
            "Collision contype:",
            model.geom_contype[gid],
        )

        print(
            "Collision conaffinity:",
            model.geom_conaffinity[gid],
        )

        print(
            "Friction:",
            np.round(model.geom_friction[gid], 6),
        )

        print(
            "Mass of parent body:",
            model.body_mass[body_id],
        )

    if not cup_found:
        print("WARNING: No cup geometry found.")

    # --------------------------------------------------------
    # 3. CUP FREE JOINT
    # --------------------------------------------------------

    print_section("3. CUP JOINT INSPECTION")

    for jid in range(model.njnt):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        body_id = int(model.jnt_bodyid[jid])

        body_name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if "cup" not in name.lower() and "cup" not in body_name.lower():
            continue

        joint_type = int(model.jnt_type[jid])

        print(f"\nJoint: {name}")
        print(f"Parent body: {body_name}")
        print(f"Joint type: {joint_type}")

        print(
            "QPOS address:",
            model.jnt_qposadr[jid],
        )

        print(
            "DOF address:",
            model.jnt_dofadr[jid],
        )

        print(
            "Initial qpos:",
            data.qpos[
                model.jnt_qposadr[jid]:
                model.jnt_qposadr[jid] + 7
            ] if joint_type == int(
                mujoco.mjtJoint.mjJNT_FREE
            ) else data.qpos[
                model.jnt_qposadr[jid]
            ],
        )

    # --------------------------------------------------------
    # 4. TABLE GEOMETRY
    # --------------------------------------------------------

    print_section("4. TABLE GEOMETRY")

    for gid in range(model.ngeom):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        if "table" not in name.lower():
            continue

        print(f"\nGeom: {name}")

        print(
            "World position:",
            np.round(data.geom_xpos[gid], 6),
        )

        print(
            "Type:",
            geom_type_name(model.geom_type[gid]),
        )

        print(
            "Size:",
            np.round(model.geom_size[gid], 6),
        )

    # --------------------------------------------------------
    # 5. END-EFFECTOR SITES
    # --------------------------------------------------------

    print_section("5. END-EFFECTOR SITES")

    site_found = False

    for sid in range(model.nsite):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            sid,
        )

        if any(
            key in name.lower()
            for key in (
                "ee",
                "end",
                "tool",
                "tcp",
                "grip",
                "hand",
            )
        ):

            site_found = True

            body_id = int(model.site_bodyid[sid])

            body_name = get_name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )

            print(f"\nSite ID: {sid}")
            print(f"Site name: {name}")
            print(f"Parent body: {body_name}")

            print(
                "World position:",
                np.round(data.site_xpos[sid], 6),
            )

    if not site_found:
        print(
            "No sites matched the common end-effector names."
        )
        print("Listing all sites:")

        for sid in range(model.nsite):

            name = get_name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                sid,
            )

            print(
                f"  {sid}: {name} | "
                f"position={np.round(data.site_xpos[sid], 6)}"
            )

    # --------------------------------------------------------
    # 6. GRIPPER COLLISION GEOMETRY
    # --------------------------------------------------------

    print_section("6. GRIPPER COLLISION GEOMETRY")

    for gid in range(model.ngeom):

        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        if "finger" not in name.lower():
            continue

        print(f"\nGeom: {name}")

        print(
            "World position:",
            np.round(data.geom_xpos[gid], 6),
        )

        print(
            "Geom size:",
            np.round(model.geom_size[gid], 6),
        )

        print(
            "Collision enabled:",
            bool(
                model.geom_contype[gid]
                and model.geom_conaffinity[gid]
            ),
        )

    # --------------------------------------------------------
    # 7. ARM JOINT CONFIGURATION
    # --------------------------------------------------------

    print_section("7. ARM JOINT CONFIGURATION")

    for side in ("left", "right"):

        print(f"\n{side.upper()} ARM")

        for i in range(1, 8):

            name = f"openarm_{side}_joint{i}"

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if jid < 0:
                continue

            qadr = int(model.jnt_qposadr[jid])

            print(
                f"  {name}: "
                f"{data.qpos[qadr]:.6f} rad"
            )

    # --------------------------------------------------------
    # 8. CONTACTS AT INITIAL STATE
    # --------------------------------------------------------

    print_section("8. INITIAL CONTACT INSPECTION")

    print("Number of contacts:", data.ncon)

    for i in range(data.ncon):

        contact = data.contact[i]

        name1 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = get_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        print(
            f"  {name1} <-> {name2} | "
            f"{contact.dist * 1000:.3f} mm"
        )

    # --------------------------------------------------------
    # 9. CUP DIMENSION REMINDER
    # --------------------------------------------------------

    print_section("9. TARGET CUP SPECIFICATION")

    print("Top diameter:    75 mm")
    print("Bottom diameter: 50 mm")
    print("Height:          85 mm")

    print(
        "\nNote: MuJoCo geom_size values depend on geometry "
        "type. Mesh dimensions must be checked against the "
        "actual mesh vertices before calculating grasp width."
    )

    # --------------------------------------------------------
    # 10. VISUALIZATION
    # --------------------------------------------------------

    print_section("10. MUJOCO VISUALIZATION")

    print("Opening viewer.")
    print("Inspect the cup and gripper positions visually.")
    print("Close the viewer when finished.")

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

        while viewer.is_running():

            viewer.sync()

            import time
            time.sleep(0.03)


if __name__ == "__main__":
    main()
