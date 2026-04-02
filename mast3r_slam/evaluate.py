import pathlib
from decimal import Decimal, ROUND_DOWN
from typing import Optional
import cv2
import lietorch
import numpy as np
import open3d as o3d
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray


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


def get_timestamp_string(dataset, frame_id):
    if hasattr(dataset, "timestamp_strings") and len(dataset.timestamp_strings) > frame_id:
        return str(dataset.timestamp_strings[frame_id])
    return f"{float(dataset.timestamps[frame_id]):.10f}"


def format_sec_nsec_timestamp(timestamp):
    if isinstance(timestamp, str):
        if "." in timestamp:
            sec, frac = timestamp.split(".", 1)
            nsec = "".join(ch for ch in frac if ch.isdigit())
            nsec = (nsec + "0" * 9)[:9]
            return f"{sec}_{nsec}"
        return f"{timestamp}_000000000"

    ts = Decimal(str(timestamp))
    sec = int(ts.to_integral_value(rounding=ROUND_DOWN))
    nsec = int(
        ((ts - Decimal(sec)) * Decimal("1000000000")).to_integral_value(
            rounding=ROUND_DOWN
        )
    )
    return f"{sec}_{nsec:09d}"


def save_keyframe_traj(logdir, logfile, dataset, keyframes: SharedKeyframes):
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        for i in range(len(keyframes)):
            keyframe = keyframes[i]
            T_WC = as_SE3(keyframe.T_WC)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            t = get_timestamp_string(dataset, keyframe.frame_id)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_full_traj(logdir, logfile, dataset, keyframes: SharedKeyframes, chunks):
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

    unfinished_chunks = [chunk for chunk in chunks if chunk["end_frame_id"] is None]
    assert len(unfinished_chunks) <= 1, "At most one unfinished last chunk is expected."

    for chunk in chunks:
        start_frame_id = chunk["start_frame_id"]
        end_frame_id = chunk["end_frame_id"]
        poses = chunk["poses"]

        if start_frame_id not in keyframe_poses:
            continue

        T_WC_start = keyframe_poses[start_frame_id]

        if end_frame_id is None or end_frame_id not in keyframe_poses or len(poses) == 0:
            for frame_id, pose_data in poses:
                poses_by_frame_id[frame_id] = as_SE3(
                    T_WC_start * lietorch.Sim3(pose_data)
                )
            continue

        last_frame_id, _ = poses[-1]
        assert (
            last_frame_id == end_frame_id
        ), "For a closed chunk, the last stored pose must correspond to the ending keyframe."

        T_WC_end = keyframe_poses[end_frame_id]
        _, end_pose_data = poses[-1]
        T_CstartCend = lietorch.Sim3(end_pose_data)
        T_chunk_correction = T_WC_start.inv() * T_WC_end * T_CstartCend.inv()
        correction_data = T_chunk_correction.data.detach().cpu().reshape(-1)

        for i, (frame_id, pose_data) in enumerate(poses[:-1], start=1):
            alpha = i / len(poses)
            T_interp = interpolate_delta(correction_data, alpha)
            T_CstartCframe = lietorch.Sim3(pose_data)
            poses_by_frame_id[frame_id] = as_SE3(
                T_WC_start * T_interp * T_CstartCframe
            )

    with open(logfile, "w") as f:
        for frame_id in sorted(poses_by_frame_id):
            T_WC = poses_by_frame_id[frame_id]
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            t = get_timestamp_string(dataset, frame_id)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_full_traj_global_sim3(
    logdir, logfile, dataset, keyframes: SharedKeyframes, chunks
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

    unfinished_chunks = [chunk for chunk in chunks if chunk["end_frame_id"] is None]
    assert len(unfinished_chunks) <= 1, "At most one unfinished last chunk is expected."

    for chunk in chunks:
        start_frame_id = chunk["start_frame_id"]
        end_frame_id = chunk["end_frame_id"]
        poses = chunk["poses"]

        if start_frame_id not in keyframe_poses:
            continue

        T_WC_start = keyframe_poses[start_frame_id]

        if end_frame_id is None or end_frame_id not in keyframe_poses or len(poses) == 0:
            for frame_id, pose_data in poses:
                poses_by_frame_id[frame_id] = as_SE3(
                    T_WC_start * lietorch.Sim3(pose_data)
                )
            continue

        last_frame_id, _ = poses[-1]
        assert (
            last_frame_id == end_frame_id
        ), "For a closed chunk, the last stored pose must correspond to the ending keyframe."

        T_WC_end = keyframe_poses[end_frame_id]
        _, end_pose_data = poses[-1]
        T_CstartCend = lietorch.Sim3(end_pose_data)
        T_chunk_correction = T_WC_start.inv() * T_WC_end * T_CstartCend.inv()

        for frame_id, pose_data in poses[:-1]:
            T_CstartCframe = lietorch.Sim3(pose_data)
            poses_by_frame_id[frame_id] = as_SE3(
                T_WC_start * T_chunk_correction * T_CstartCframe
            )

    with open(logfile, "w") as f:
        for frame_id in sorted(poses_by_frame_id):
            T_WC = poses_by_frame_id[frame_id]
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            t = get_timestamp_string(dataset, frame_id)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_full_traj_anchor_keyframe(
    logdir, logfile, dataset, keyframes: SharedKeyframes, chunks
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

    for chunk in chunks:
        start_frame_id = chunk["start_frame_id"]
        poses = chunk["poses"]

        if start_frame_id not in keyframe_poses:
            continue

        T_WC_start = keyframe_poses[start_frame_id]
        for frame_id, pose_data in poses:
            poses_by_frame_id[frame_id] = as_SE3(
                T_WC_start * lietorch.Sim3(pose_data)
            )

    with open(logfile, "w") as f:
        for frame_id in sorted(poses_by_frame_id):
            T_WC = poses_by_frame_id[frame_id]
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            t = get_timestamp_string(dataset, frame_id)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        points_world, color = get_keyframe_pointcloud_world(
            keyframe, c_conf_threshold
        )
        if len(points_world) == 0:
            continue
        pointclouds.append(points_world)
        colors.append(color)
    if len(pointclouds) == 0:
        save_pcd(savedir / filename, np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))
        return
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_pcd(savedir / filename, pointclouds, colors)


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


def get_conf_mask(conf, c_conf_threshold=None):
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    if conf.size == 0:
        return np.zeros((0,), dtype=bool)
    if c_conf_threshold is None:
        threshold = np.percentile(conf, 25)
        threshold = max(threshold, 1.5)
    else:
        threshold = c_conf_threshold
    return conf >= threshold


def _get_keyframe_pointcloud_world_raw(keyframe):
    X_canon = keyframe.X_canon
    if config["use_calib"]:
        X_canon = constrain_points_to_ray(
            keyframe.img_shape.flatten()[:2], X_canon[None], keyframe.K
        ).squeeze(0)

    points_world = keyframe.T_WC.act(X_canon).cpu().numpy().reshape(-1, 3)
    color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    conf = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
    return points_world, color, conf


def get_keyframe_pointcloud_world(keyframe, c_conf_threshold=None):
    points_world, color, conf = _get_keyframe_pointcloud_world_raw(keyframe)
    valid = get_conf_mask(conf, c_conf_threshold)
    return points_world[valid], color[valid]


def save_frame_pointclouds(savedir, dataset, keyframes: SharedKeyframes, chunks):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    keyframes_by_frame_id = {}
    poses_by_frame_id = {}
    ref_keyframe_by_frame_id = {}
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        keyframes_by_frame_id[keyframe.frame_id] = keyframe
        poses_by_frame_id[keyframe.frame_id] = as_SE3(keyframe.T_WC)
        ref_keyframe_by_frame_id[keyframe.frame_id] = keyframe.frame_id

    for chunk in chunks:
        start_frame_id = chunk["start_frame_id"]
        poses = chunk["poses"]
        if start_frame_id not in keyframes_by_frame_id:
            continue

        T_WC_start = lietorch.Sim3(
            keyframes_by_frame_id[start_frame_id].T_WC.data.detach().cpu().clone()
        )
        for frame_id, pose_data in poses:
            poses_by_frame_id[frame_id] = as_SE3(T_WC_start * lietorch.Sim3(pose_data))
            ref_keyframe_by_frame_id[frame_id] = start_frame_id

    if len(poses_by_frame_id) == 0:
        return

    if getattr(dataset, "camera_intrinsics", None) is not None:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(dtype=torch.float32)
    elif config["use_calib"] and len(keyframes) > 0 and keyframes[0].K is not None:
        K = keyframes[0].K.detach().cpu().to(dtype=torch.float32)
    else:
        raise ValueError(
            "Frame pointcloud export requires intrinsics to compute each frame FOV."
        )

    sample_keyframe = keyframes[0]
    height, width = sample_keyframe.img_shape.flatten()[:2].detach().cpu().tolist()
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    for frame_id in sorted(poses_by_frame_id):
        ref_keyframe = keyframes_by_frame_id[ref_keyframe_by_frame_id[frame_id]]
        points_world, colors = get_keyframe_pointcloud_world(ref_keyframe)

        if len(points_world) == 0:
            points_frame = np.empty((0, 3), dtype=np.float32)
            colors_frame = np.empty((0, 3), dtype=np.uint8)
        else:
            points_world_t = torch.from_numpy(points_world).to(dtype=torch.float32)
            T_WC_frame = poses_by_frame_id[frame_id]
            points_frame_t = T_WC_frame.inv().act(points_world_t)
            points_frame = points_frame_t.detach().cpu().numpy().reshape(-1, 3)

            z = points_frame[:, 2]
            valid_z = z > 0
            u = fx * (points_frame[:, 0] / np.maximum(z, 1e-8)) + cx
            v = fy * (points_frame[:, 1] / np.maximum(z, 1e-8)) + cy
            valid_fov = (
                valid_z
                & (u >= 0.0)
                & (u < float(width))
                & (v >= 0.0)
                & (v < float(height))
            )

            points_frame = points_frame[valid_fov]
            colors_frame = colors[valid_fov]

        t = format_sec_nsec_timestamp(get_timestamp_string(dataset, frame_id))
        save_pcd(savedir / f"{t}.pcd", points_frame, colors_frame)


def save_ply(filename, points, colors):
    save_pointcloud(filename, points, colors)


def save_pcd(filename, points, colors):
    save_pointcloud(filename, points, colors)


def save_pointcloud(filename, points, colors):
    filename = pathlib.Path(filename)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = colors.astype(np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(filename), pcd)
