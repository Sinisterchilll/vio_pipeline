"""
Visual-Inertial Odometry Pipeline for 3D Head Position Estimation.

Clean architecture:
  - Gyroscope integration for rotation (primary)
  - Essential matrix decomposition for translation direction
  - Visual rotation blended with gyro for robustness
  - Consistent scale from optical flow magnitude + assumed scene depth

No PnP (causes drift lock). No accelerometer double integration (unreliable at 30Hz).
Optimized for RPE over 3-minute windows.
"""

import json
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ─── Calibration ──────────────────────────────────────────────────────────────

K_ORIG = np.array([[718.90196364, 0, 960.01857437],
                    [0, 716.33626950, 558.31079911],
                    [0, 0, 1]])
DIST = np.array([-0.28182606, 0.07391488, 0.00031393, 0.00090297])

T_CAM_IMU = np.array([
    [ 0.5488140373,  0.4040019011, -0.7318371516, -0.0003648381],
    [-0.8169342334,  0.4448444651, -0.3670583881, -0.0000230148],
    [ 0.1772614196,  0.7993096182,  0.5741798702, -0.0002253224],
    [0, 0, 0, 1]])
R_c_i = T_CAM_IMU[:3, :3]
R_i_c = R_c_i.T

W, H = 1920, 1080
K, _ = cv2.getOptimalNewCameraMatrix(K_ORIG, DIST, (W, H), 0, (W, H))
M1, M2 = cv2.initUndistortRectifyMap(K_ORIG, DIST, None, K, (W, H), cv2.CV_32FC1)
FOCAL = (K[0, 0] + K[1, 1]) / 2.0


def load_imu(path):
    d = []
    with open(path) as f:
        for l in f:
            e = json.loads(l)
            d.append({'t_us': e['t_us'], 't_s': e['t_us']*1e-6,
                      'acc': np.array(e['acc']), 'gyro': np.array(e['gyro'])})
    return d


def rot_between(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = np.dot(a, b)
    if s < 1e-8: return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx@vx*(1-c)/(s*s)


class Tracker:
    def __init__(self):
        self.det = dict(maxCorners=500, qualityLevel=0.01, minDistance=15, blockSize=7)
        self.lk = dict(winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.pg = self.pp = self.pi = None; self.nid = 0

    def _d(self, g, mask=None):
        p = cv2.goodFeaturesToTrack(g, mask=mask, **self.det)
        if p is None: return np.empty((0,1,2),np.float32), np.empty(0,int)
        ids = np.arange(self.nid, self.nid+len(p), dtype=int); self.nid += len(p)
        return p.astype(np.float32), ids

    def track(self, g):
        if self.pg is None:
            self.pg = g; self.pp, self.pi = self._d(g); return None
        if self.pp is None or len(self.pp) < 8:
            self.pp, self.pi = self._d(g); self.pg = g; return None
        c,s1,_ = cv2.calcOpticalFlowPyrLK(self.pg, g, self.pp, None, **self.lk)
        b,s2,_ = cv2.calcOpticalFlowPyrLK(g, self.pg, c, None, **self.lk)
        ok = (s1.flatten()==1)&(s2.flatten()==1)&(np.linalg.norm(self.pp-b,axis=2).flatten()<1.0)
        pp = self.pp[ok].reshape(-1,2); cp = c[ok].reshape(-1,2); ids = self.pi[ok]
        res = (pp, cp, ids) if len(pp) >= 12 else None
        act = c[ok]; aids = ids
        if len(act) < 150:
            mask = np.ones(g.shape, np.uint8)*255
            for pt in act.reshape(-1,2): cv2.circle(mask,(int(pt[0]),int(pt[1])),15,0,-1)
            np2,ni2 = self._d(g, mask)
            if len(np2)>0:
                act = np.vstack([act,np2]) if len(act)>0 else np2
                aids = np.concatenate([aids,ni2]) if len(aids)>0 else ni2
        self.pg=g; self.pp=act.reshape(-1,1,2).astype(np.float32); self.pi=aids
        return res


class VIOPipeline:
    def process_clip(self, video_path, imu_path, output_path=None):
        print(f"Processing: {video_path}")

        imu = load_imu(imu_path)
        nb = min(90, len(imu))
        gbias = np.mean([d['gyro'] for d in imu[:nb]], axis=0)
        macc = np.mean([d['acc'] for d in imu[:nb]], axis=0)

        R_w_c = rot_between(macc/np.linalg.norm(macc), np.array([0,0,-1])) @ R_i_c

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            print("  Cannot read video"); cap.release(); return [], []
        print(f"  {total} frames @ {fps}fps, {len(imu)} imu")

        t_w = np.zeros(3)
        R_prev = R_w_c.copy()
        tr = Tracker()
        poses, timestamps = [], []
        fi = 0
        vo_ok = 0

        # Smoothing: running average of scale
        scale_buf = deque(maxlen=90)

        while True:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(cv2.remap(frame, M1, M2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
            ii = min(fi, len(imu)-1)

            # ── Gyro rotation ──
            dR_c = np.eye(3)
            if fi > 0:
                pi = min(fi-1, len(imu)-1)
                dR_i = np.eye(3)
                for j in range(pi, min(ii, len(imu)-1)):
                    dt = imu[j+1]['t_s'] - imu[j]['t_s']
                    if 0 < dt < 0.2:
                        w = imu[j]['gyro'] - gbias
                        if np.linalg.norm(w) > 1e-10:
                            dR_i = dR_i @ Rotation.from_rotvec(w*dt).as_matrix()
                dR_c = R_c_i @ dR_i @ R_i_c
                R_w_c = R_prev @ dR_c

            # ── Visual tracking ──
            res = tr.track(gray)
            if res is not None:
                pp, cp, fids = res
                E, me = cv2.findEssentialMat(pp, cp, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None and me is not None:
                    inl = me.ravel().astype(bool)
                    if inl.sum() >= 10:
                        _, Rv, tv, _ = cv2.recoverPose(E, pp, cp, K, mask=me)
                        td = tv.flatten()
                        td /= (np.linalg.norm(td) + 1e-12)

                        # Rotation fusion
                        if fi > 0:
                            rv_g = Rotation.from_matrix(dR_c).as_rotvec()
                            rv_v = Rotation.from_matrix(Rv).as_rotvec()
                            if np.linalg.norm(rv_g - rv_v) < 0.3:
                                R_w_c = R_prev @ Rotation.from_rotvec(
                                    0.8*rv_g + 0.2*rv_v).as_matrix()

                        # Scale from pixel displacement
                        dpx = np.median(np.linalg.norm(cp[inl] - pp[inl], axis=1))
                        # Scene depth assumption: ~3m (typical for indoor construction)
                        raw_scale = (dpx / FOCAL) * 3.0
                        raw_scale = max(0.0001, min(raw_scale, 1.5))
                        scale_buf.append(raw_scale)
                        scale = np.median(scale_buf)

                        t_w = t_w + R_w_c @ (td * scale)
                        vo_ok += 1

            R_prev = R_w_c.copy()
            T = np.eye(4); T[:3,:3] = R_w_c; T[:3,3] = t_w
            poses.append(T); timestamps.append(imu[ii]['t_us'])
            fi += 1
            if fi % 1000 == 0:
                print(f"  {fi}/{total} pos=[{t_w[0]:.2f},{t_w[1]:.2f},{t_w[2]:.2f}] vo={vo_ok}")

        cap.release()
        print(f"  Done. {len(poses)} poses, vo={vo_ok}/{fi}")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                for tu, T in zip(timestamps, poses):
                    q = Rotation.from_matrix(T[:3,:3]).as_quat()
                    f.write(f"{tu*1e-6:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} "
                            f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            print(f"  Saved to {output_path}")

        return poses, timestamps


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clip"); p.add_argument("--data-dir")
    p.add_argument("--output-dir", default="output")
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if a.clip:
        c = Path(a.clip)
        VIOPipeline().process_clip(c/"video.mp4", c/"imu.txt", out/f"{c.name}_poses.txt")
    elif a.data_dir:
        for c in sorted(Path(a.data_dir).glob("clip_*")):
            VIOPipeline().process_clip(c/"video.mp4", c/"imu.txt", out/f"{c.name}_poses.txt")
    else: p.print_help()

if __name__ == "__main__":
    main()
