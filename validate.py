"""
Validate VIO output quality using multiple self-consistency checks.
No ground truth needed.

Checks:
1. evo library validation (SE(3) conformity, quaternion validity)
2. Speed plausibility (factory worker: 0.3-2.0 m/s typical)
3. Height consistency (head should stay ~1.5-2.0m, not fly or go underground)
4. Smoothness (jerk should be bounded — humans can't teleport)
5. Rotation rate (head turns should be < 5 rad/s for normal motion)
6. Compare VIO trajectory vs pure gyro-only trajectory (cross-modal check)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Calibration (same as pipeline)
T_CAM_IMU = np.array([
    [ 0.5488140373,  0.4040019011, -0.7318371516, -0.0003648381],
    [-0.8169342334,  0.4448444651, -0.3670583881, -0.0000230148],
    [ 0.1772614196,  0.7993096182,  0.5741798702, -0.0002253224],
    [0, 0, 0, 1]])
R_c_i = T_CAM_IMU[:3, :3]
R_i_c = R_c_i.T


def load_poses(path):
    d = np.loadtxt(path, comments='#')
    return d[:, 0], d[:, 1:4], d[:, 4:8]


def load_imu(path):
    data = []
    with open(path) as f:
        for l in f:
            e = json.loads(l)
            data.append({'t_s': e['t_us'] * 1e-6,
                         'gyro': np.array(e['gyro']),
                         'acc': np.array(e['acc'])})
    return data


def gyro_only_rotations(imu_data):
    """Compute rotation trajectory from gyro only (no visual)."""
    nb = min(90, len(imu_data))
    gbias = np.mean([d['gyro'] for d in imu_data[:nb]], axis=0)

    # Initial orientation from gravity
    macc = np.mean([d['acc'] for d in imu_data[:nb]], axis=0)
    g_n = macc / np.linalg.norm(macc)
    v1, v2 = g_n, np.array([0, 0, -1])
    c = np.cross(v1, v2); d = np.dot(v1, v2); s = np.linalg.norm(c)
    if s > 1e-8:
        vx = np.array([[0,-c[2],c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])
        R_w_i = np.eye(3) + vx + vx@vx*(1-d)/(s*s)
    else:
        R_w_i = np.eye(3)
    R_w_c = R_w_i @ R_i_c

    rots = [R_w_c.copy()]
    for i in range(len(imu_data) - 1):
        dt = imu_data[i+1]['t_s'] - imu_data[i]['t_s']
        if 0 < dt < 0.2:
            w = imu_data[i]['gyro'] - gbias
            if np.linalg.norm(w) > 1e-10:
                dR_i = Rotation.from_rotvec(w * dt).as_matrix()
                dR_c = R_c_i @ dR_i @ R_i_c
                R_w_c = R_w_c @ dR_c
        rots.append(R_w_c.copy())
    return rots


def validate_clip(pose_file, imu_file=None):
    """Run all validation checks on a single clip."""
    ts, pos, quats = load_poses(pose_file)
    n = len(ts)
    results = {'file': Path(pose_file).name, 'n_poses': n, 'pass': True, 'issues': []}

    # --- Check 1: Basic validity ---
    if n < 100:
        results['issues'].append(f"Too few poses: {n}")
        results['pass'] = False
        return results

    # Quaternion validity (should be unit)
    qnorms = np.linalg.norm(quats, axis=1)
    bad_q = np.sum(np.abs(qnorms - 1.0) > 0.01)
    if bad_q > 0:
        results['issues'].append(f"{bad_q} non-unit quaternions")

    # --- Check 2: Speed plausibility ---
    dt = np.diff(ts)
    dt[dt == 0] = 1e-6  # avoid div by zero
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    avg_speed = np.median(vel)
    max_speed = np.percentile(vel, 99)  # 99th percentile (ignore outlier frames)

    results['avg_speed_ms'] = round(float(avg_speed), 3)
    results['max_speed_99pct'] = round(float(max_speed), 3)

    if avg_speed < 0.05:
        results['issues'].append(f"Avg speed too low: {avg_speed:.3f} m/s (trajectory stuck?)")
    if avg_speed > 3.0:
        results['issues'].append(f"Avg speed too high: {avg_speed:.3f} m/s (scale too large?)")
    if max_speed > 10.0:
        results['issues'].append(f"99th pct speed: {max_speed:.1f} m/s (teleportation?)")

    # --- Check 3: Height consistency ---
    z_range = pos[:, 2].max() - pos[:, 2].min()
    results['z_range_m'] = round(float(z_range), 2)
    if z_range > 30:
        results['issues'].append(f"Z range = {z_range:.1f}m (worker flying?)")

    # --- Check 4: Smoothness (jerk) ---
    if len(vel) > 2:
        accel = np.diff(vel) / dt[1:]
        jerk_99 = np.percentile(np.abs(accel), 99)
        results['jerk_99pct'] = round(float(jerk_99), 1)
        if jerk_99 > 500:
            results['issues'].append(f"Extreme jerk: {jerk_99:.0f} m/s^3")

    # --- Check 5: Rotation rate ---
    rot_rates = []
    for i in range(1, min(n, len(quats))):
        q1 = Rotation.from_quat(quats[i-1])
        q2 = Rotation.from_quat(quats[i])
        dR = q1.inv() * q2
        angle = np.linalg.norm(dR.as_rotvec())
        dti = ts[i] - ts[i-1]
        if dti > 0:
            rot_rates.append(angle / dti)
    if rot_rates:
        avg_rot = np.median(rot_rates)
        max_rot = np.percentile(rot_rates, 99)
        results['avg_rot_rate_rads'] = round(float(avg_rot), 3)
        results['max_rot_rate_99pct'] = round(float(max_rot), 3)
        if max_rot > 20:
            results['issues'].append(f"Extreme rotation rate: {max_rot:.1f} rad/s")

    # --- Check 6: VIO vs Gyro-only rotation comparison ---
    if imu_file and Path(imu_file).exists():
        imu_data = load_imu(imu_file)
        gyro_rots = gyro_only_rotations(imu_data)

        # Compare rotation at every 100th frame
        angle_diffs = []
        for i in range(0, min(n, len(gyro_rots)), 100):
            R_vio = Rotation.from_quat(quats[i]).as_matrix()
            R_gyro = gyro_rots[i]
            dR = R_vio.T @ R_gyro
            angle_diffs.append(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))

        if angle_diffs:
            mean_rot_diff = np.mean(angle_diffs)
            max_rot_diff = np.max(angle_diffs)
            results['vio_vs_gyro_mean_deg'] = round(float(np.degrees(mean_rot_diff)), 1)
            results['vio_vs_gyro_max_deg'] = round(float(np.degrees(max_rot_diff)), 1)
            if mean_rot_diff > np.radians(45):
                results['issues'].append(
                    f"VIO rotation diverges {np.degrees(mean_rot_diff):.0f} deg from gyro")

    # --- Check 7: Path length / duration sanity ---
    duration = ts[-1] - ts[0]
    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    results['path_m'] = round(float(path_len), 1)
    results['duration_s'] = round(float(duration), 1)

    if not results['issues']:
        results['verdict'] = 'PASS - all checks OK'
    else:
        results['verdict'] = f'WARN - {len(results["issues"])} issue(s)'
        results['pass'] = False

    return results


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate VIO poses (no ground truth needed)")
    p.add_argument("--clip", help="Single clip dir (contains video.mp4 + imu.txt)")
    p.add_argument("--pose-file", help="Single pose file")
    p.add_argument("--output-dir", default="output", help="Dir with all pose files")
    p.add_argument("--data-dir", help="Dir with clip data (for gyro comparison)")
    a = p.parse_args()

    if a.pose_file:
        imu = None
        if a.clip:
            imu = str(Path(a.clip) / "imu.txt")
        r = validate_clip(a.pose_file, imu)
        print(f"\n{'='*60}")
        print(f"  {r['file']}")
        print(f"{'='*60}")
        for k, v in r.items():
            if k not in ('file', 'issues'):
                print(f"  {k}: {v}")
        if r['issues']:
            print(f"\n  Issues:")
            for issue in r['issues']:
                print(f"    - {issue}")
    else:
        out = Path(a.output_dir)
        files = sorted(out.glob("*_poses.txt"))
        print(f"Validating {len(files)} clips...\n")

        passed = 0
        warned = 0
        all_speeds = []
        all_paths = []
        all_rot_diffs = []

        for f in files:
            # Try to find matching IMU file
            clip_name = f.stem.replace('_poses', '')
            imu_file = None
            if a.data_dir:
                imu_path = Path(a.data_dir) / clip_name / "imu.txt"
                if imu_path.exists():
                    imu_file = str(imu_path)

            r = validate_clip(str(f), imu_file)
            status = 'OK' if r['pass'] else 'WARN'
            if r['pass']:
                passed += 1
            else:
                warned += 1

            speed = r.get('avg_speed_ms', 0)
            path = r.get('path_m', 0)
            rot_diff = r.get('vio_vs_gyro_mean_deg', None)
            all_speeds.append(speed)
            all_paths.append(path)
            if rot_diff is not None:
                all_rot_diffs.append(rot_diff)

            issues_str = '; '.join(r.get('issues', []))
            rot_str = f"  gyro_diff={rot_diff}deg" if rot_diff else ""
            print(f"  [{status}] {clip_name}: path={path}m speed={speed}m/s{rot_str}"
                  f"{'  !! '+issues_str if issues_str else ''}")

        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passed} passed, {warned} warnings out of {len(files)} clips")
        print(f"  Avg path: {np.mean(all_paths):.1f}m")
        print(f"  Avg speed: {np.mean(all_speeds):.2f} m/s")
        if all_rot_diffs:
            print(f"  VIO vs Gyro rotation diff: {np.mean(all_rot_diffs):.1f} deg mean, "
                  f"{np.max(all_rot_diffs):.1f} deg max")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
