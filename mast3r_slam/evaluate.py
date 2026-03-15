import pathlib
from typing import Optional
import cv2
import lietorch
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_poses(
    logdir,
    logfile,
    timestamps,
    keyframes: SharedKeyframes,
    chunks,
):
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    keyframe_poses = {}
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        keyframe_poses[keyframe.frame_id] = lietorch.Sim3(
            keyframe.T_WC.data.detach().cpu().clone()
        )

    poses_by_frame_id = {
        frame_id: as_SE3(T_WC) for frame_id, T_WC in keyframe_poses.items()
    }

    def normalize_quat(q):
        return q / torch.linalg.norm(q)

    def slerp_identity(q, alpha):
        q = normalize_quat(q)
        if q[-1] < 0:
            q = -q

        q0 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype)
        dot = torch.clamp(q[-1], -1.0, 1.0)

        if dot > 0.9995:
            q_interp = (1.0 - alpha) * q0 + alpha * q
            return normalize_quat(q_interp)

        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * alpha
        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = torch.sin(theta) / sin_theta_0
        return s0 * q0 + s1 * q

    def interpolate_delta(delta_data, alpha):
        t = delta_data[:3] * alpha
        q = slerp_identity(delta_data[3:7], alpha)
        s = delta_data[7:8].pow(alpha)
        return lietorch.Sim3(torch.cat([t, q, s], dim=0)[None])

    for chunk in chunks:
        start_frame_id = chunk["start_frame_id"]
        end_frame_id = chunk["end_frame_id"]
        poses = chunk["poses"]
        if (
            start_frame_id not in keyframe_poses
            or end_frame_id not in keyframe_poses
            or len(poses) == 0
        ):
            continue

        T_WC_start = keyframe_poses[start_frame_id]
        T_WC_end = keyframe_poses[end_frame_id]
        _, end_pose_data = poses[-1]
        T_CstartCend = lietorch.Sim3(end_pose_data)
        T_chunk_correction = T_WC_start.inv() * T_WC_end * T_CstartCend.inv()
        correction_data = T_chunk_correction.data.detach().cpu().reshape(-1)

        for i, (frame_id, pose_data) in enumerate(poses, start=1):
            alpha = i / len(poses)
            T_interp = interpolate_delta(correction_data, alpha)
            T_CstartCframe = lietorch.Sim3(pose_data)
            poses_by_frame_id[frame_id] = as_SE3(T_WC_start * T_interp * T_CstartCframe)

    with open(logfile, "w") as f:
        for frame_id in sorted(poses_by_frame_id):
            t = timestamps[frame_id]
            T_WC = poses_by_frame_id[frame_id]
            x, y, z, qx, qy, qz, qw = T_WC.data.cpu().numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
