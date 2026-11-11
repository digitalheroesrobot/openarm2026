"""
OpenArm V1 collision-aware home-pose diagnostic.

Objectives:
1. Calculate left and right arm positions using damped least-squares IK.
2. Target 150 mm above the tabletop.
3. Check physical robot/table/cup contacts.
4. Display the resulting configuration.

Important:
- This is a static-pose diagnostic, not trajectory planning.
- Position IK does not constrain end-effector orientation.
- A successful IK result does not imply a collision-free pose.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
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

TARGETS = {
    "left": np.array([0.20, 0.20, 0.45]),
    "right": np.array([0.20, -0.20, 0.45]),
}

MAX_ITERATIONS = 500
DAMPING = 0.03
TOLERANCE = 0.003


def get_joint_ids(model, side):
    ids = []

    for i in range(1, 8):
        name = f"openarm_{side}_joint{i}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")

        ids.append(joint_id)

    return ids


def solve_ik(model, data, side, target):

    site_name = f"{side}_ee_control_point"

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(f"Missing site: {site_name}")

    joint_ids = get_joint_ids(model, side)

    qpos_ids = [
        model.jnt_qposadr[j]
        for j in joint_ids
    ]

    dof_ids = [
        model.jnt_dofadr[j]
        for j in joint_ids
    ]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for iteration in range(MAX_ITERATIONS):

        mujoco.mj_forward(model, data)

        current = data.site_xpos[site_id].copy()
        error = target - current

        if np.linalg.norm(error) < TOLERANCE:
            break

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J = jacp[:, dof_ids]

        dq = J.T @ np.linalg.solve(
            J @ J.T + DAMPING**2 * np.eye(3),
            error,
        )

        dq = np.clip(dq, -0.05, 0.05)

        for k, joint_id in enumerate(joint_ids):

            lower, upper = model.jnt_range[joint_id]
            qpos_id = qpos_ids[k]

            data.qpos[qpos_id] = np.clip(
                data.qpos[qpos_id] + dq[k],
                lower,
                upper,
            )

    mujoco.mj_forward(model, data)

    actual = data.site_xpos[site_id].copy()
    error = np.linalg.norm(target - actual)

    return {
        "target": target,
        "actual": actual,
        "error": error,
        "success": error < TOLERANCE,
        "iterations": iteration + 1,
    }


def geom_info(model, geom_id):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    body_id = model.geom_bodyid[geom_id]

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    return name or f"geom_{geom_id}", body_name or f"body_{body_id}"


def main():

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    print("\n========== HIGHER HOME POSE IK ==========")

    results = {}

    for side in ("left", "right"):

        result = solve_ik(
            model,
            data,
            side,
            TARGETS[side],
        )

        results[side] = result

        print(f"\n{side.upper()} ARM")
        print("Target:", result["target"])
        print("Actual:", np.round(result["actual"], 5))
        print(f"Error: {result['error'] * 1000:.2f} mm")
        print("IK:", "PASS" if result["success"] else "FAIL")

    print("\n========== JOINT CONFIGURATION ==========")

    for side in ("left", "right"):

        for joint_id in get_joint_ids(model, side):

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            qpos_id = model.jnt_qposadr[joint_id]

            print(f"{name}: {data.qpos[qpos_id]:.6f}")

    # Recompute collision contacts at the final static pose.
    mujoco.mj_forward(model, data)

    print("\n========== TABLE/CUP CONTACTS ==========")

    relevant_contacts = 0
    penetrations = 0

    for i in range(data.ncon):

        contact = data.contact[i]

        name1, body1 = geom_info(model, contact.geom1)
        name2, body2 = geom_info(model, contact.geom2)

        involved = any(
            "table" in item.lower() or "cup" in item.lower()
            for item in (name1, name2, body1, body2)
        )

        if not involved:
            continue

        relevant_contacts += 1

        if contact.dist < 0:
            penetrations += 1

        print(
            f"{name1} ({body1}) <-> "
            f"{name2} ({body2}) | "
            f"{contact.dist * 1000:.2f} mm"
        )

    print("\n========== COLLISION SUMMARY ==========")
    print("Total MuJoCo contacts:", data.ncon)
    print("Table/cup contacts:", relevant_contacts)
    print("Penetrating contacts:", penetrations)

    if penetrations == 0:
        print("TABLE/CUP CHECK: PASS")
    else:
        print("TABLE/CUP CHECK: FAIL")

    print("=======================================")

    print("\nOpening viewer. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            start = time.perf_counter()

            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(
                max(0.0, 1.0 / 60.0 - (time.perf_counter() - start))
            )


if __name__ == "__main__":
    main()