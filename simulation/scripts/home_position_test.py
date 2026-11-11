"""
OpenArm V1 tabletop home-position test.

Objectives:
1. Load the existing V1 scene with table and cup.
2. Calculate a home posture for both arms using MuJoCo Jacobians.
3. Move both end-effector control points toward tabletop targets.
4. Display the resulting configuration in the MuJoCo viewer.

This is a position-only IK test.
Collision-free motion and end-effector orientation
will be validated in subsequent steps.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "vendor"
    / "openarm_mujoco"
    / "v1"
    / "scene_with_cup.xml"
)

TARGETS = {
    "left": np.array([0.20, 0.20, 0.34]),
    "right": np.array([0.20, -0.20, 0.34]),
}

SITE_NAMES = {
    "left": "left_ee_control_point",
    "right": "right_ee_control_point",
}

MAX_ITERATIONS = 500
DAMPING = 0.03
POSITION_TOLERANCE = 0.003


# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    return model, data


# ---------------------------------------------------------
# Joint identification
# ---------------------------------------------------------

def get_arm_joint_ids(model, side):

    joint_ids = []

    for i in range(1, 8):

        name = f"openarm_{side}_joint{i}"

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {name}")

        joint_ids.append(joint_id)

    return joint_ids


# ---------------------------------------------------------
# Position-only damped least-squares IK
# ---------------------------------------------------------

def solve_arm_ik(model, data, side, target):

    site_name = SITE_NAMES[side]

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(f"Missing site: {site_name}")

    joint_ids = get_arm_joint_ids(model, side)

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

        current_position = data.site_xpos[site_id].copy()

        error = target - current_position

        error_norm = np.linalg.norm(error)

        if error_norm < POSITION_TOLERANCE:
            break

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J = jacp[:, dof_ids]

        # Damped least-squares solution:
        # dq = J.T @ inv(J @ J.T + lambda^2 I) @ error

        dq = J.T @ np.linalg.solve(
            J @ J.T + (DAMPING ** 2) * np.eye(3),
            error,
        )

        # Limit each iteration to avoid unstable jumps.
        dq = np.clip(dq, -0.05, 0.05)

        for k, joint_id in enumerate(joint_ids):

            qpos_id = qpos_ids[k]

            lower, upper = model.jnt_range[joint_id]

            data.qpos[qpos_id] = np.clip(
                data.qpos[qpos_id] + dq[k],
                lower,
                upper,
            )

    mujoco.mj_forward(model, data)

    final_position = data.site_xpos[site_id].copy()
    final_error = np.linalg.norm(target - final_position)

    return {
        "target": target,
        "actual": final_position,
        "error": final_error,
        "iterations": iteration + 1,
        "success": final_error < POSITION_TOLERANCE,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    model, data = load_model()

    print("\n========== OPENARM V1 HOME IK ==========")

    results = {}

    # Solve each arm independently.
    for side in ("left", "right"):

        result = solve_arm_ik(
            model,
            data,
            side,
            TARGETS[side],
        )

        results[side] = result

        print(f"\n{side.upper()} ARM")
        print("Target:", result["target"])
        print("Actual:", result["actual"])
        print(f"Position error: {result['error'] * 1000:.2f} mm")
        print("Iterations:", result["iterations"])
        print("IK:", "PASS" if result["success"] else "CHECK")

    print("\n========== JOINT CONFIGURATION ==========")

    for side in ("left", "right"):

        for joint_id in get_arm_joint_ids(model, side):

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            qpos_id = model.jnt_qposadr[joint_id]

            print(f"{name}: {data.qpos[qpos_id]:.5f} rad")

    all_success = all(
        result["success"]
        for result in results.values()
    )

    print("\nOverall IK:", "PASS" if all_success else "NOT REACHED")
    print("========================================")

    print("\nOpening viewer. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            start = time.perf_counter()

            # Hold the calculated posture.
            mujoco.mj_forward(model, data)

            viewer.sync()

            elapsed = time.perf_counter() - start

            time.sleep(max(0.0, 1.0 / 60.0 - elapsed))


if __name__ == "__main__":
    main()
