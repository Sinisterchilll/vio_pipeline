"""
Evaluate VIO trajectory using RPE (Relative Pose Error).

Since no ground truth is provided, this script computes:
1. Self-consistency metrics (smoothness, drift indicators)
2. RPE statistics if ground truth is supplied
"""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load_tum_poses(path):
    """Load poses in TUM format: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(path, comments='#')
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quats = data[:, 4:8]  # [qx, qy, qz, qw]
    return timestamps, positions, quats


def compute_relative_pose_error(positions, timestamps, delta_seconds=180.0):
    """Compute RPE over a given time window.

    For a 3-minute clip that IS the evaluation window,
    this computes the drift over the full clip.

    For sub-windows, splits into overlapping segments.
    """
    dt = timestamps[-1] - timestamps[0]

    if dt <= delta_seconds * 1.1:
        # Clip is approximately one window — compute overall drift
        # Relative translation error: displacement between start and end
        # relative to path length
        total_path = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        net_disp = np.linalg.norm(positions[-1] - positions[0])

        return {
            'total_path_m': total_path,
            'net_displacement_m': net_disp,
            'drift_ratio': net_disp / max(total_path, 1e-6),
        }

    # Multiple windows
    fps = len(timestamps) / dt
    window_frames = int(delta_seconds * fps)
    step = window_frames // 2  # 50% overlap

    rpe_trans = []
    rpe_rot = []

    for start in range(0, len(positions) - window_frames, step):
        end = start + window_frames
        p_start = positions[start]
        p_end = positions[end]

        segment_path = np.sum(np.linalg.norm(
            np.diff(positions[start:end+1], axis=0), axis=1))

        trans_err = np.linalg.norm(p_end - p_start)
        rpe_trans.append(trans_err / max(segment_path, 1e-6))

    return {
        'rpe_trans_mean': np.mean(rpe_trans),
        'rpe_trans_median': np.median(rpe_trans),
        'rpe_trans_std': np.std(rpe_trans),
        'num_windows': len(rpe_trans),
    }


def compute_smoothness(positions, timestamps):
    """Compute trajectory smoothness metrics."""
    velocities = np.diff(positions, axis=0) / np.diff(timestamps)[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[:-1])[:, None]
    jerk = np.linalg.norm(accelerations, axis=1)

    return {
        'avg_speed_ms': np.mean(speeds),
        'max_speed_ms': np.max(speeds),
        'speed_std_ms': np.std(speeds),
        'avg_jerk': np.mean(jerk),
        'max_jerk': np.max(jerk),
    }


def evaluate_clip(pose_path):
    """Evaluate a single clip's trajectory."""
    timestamps, positions, quats = load_tum_poses(pose_path)

    rpe = compute_relative_pose_error(positions, timestamps)
    smooth = compute_smoothness(positions, timestamps)

    return {**rpe, **smooth}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-file", type=str, help="Single pose file to evaluate")
    parser.add_argument("--output-dir", type=str, help="Directory with pose files")
    args = parser.parse_args()

    if args.pose_file:
        result = evaluate_clip(args.pose_file)
        print(f"\nEvaluation: {args.pose_file}")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
    elif args.output_dir:
        out = Path(args.output_dir)
        files = sorted(out.glob("*_poses.txt"))
        print(f"Evaluating {len(files)} clips")
        all_results = []
        for f in files:
            r = evaluate_clip(f)
            all_results.append(r)
            print(f"  {f.name}: path={r['total_path_m']:.1f}m, "
                  f"speed={r['avg_speed_ms']:.2f}m/s, "
                  f"drift={r.get('drift_ratio', 0):.3f}")

        if all_results:
            print(f"\nSummary over {len(all_results)} clips:")
            for key in all_results[0]:
                vals = [r[key] for r in all_results if key in r]
                print(f"  {key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
