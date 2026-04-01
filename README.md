# 3D Head Position via Visual-Inertial Odometry

A VIO pipeline that produces 6DoF camera poses from egocentric video + IMU data, built for the [Build AI Head Position Challenge](https://www.eddy.build/headposition).

## Problem

Recover a factory worker's 3D head position from a head-mounted camera (1080p, 30fps) and IMU (30Hz). The optimization target is **RPE (Relative Pose Error) over 3-minute windows** — local trajectory accuracy, not global drift.

## Approach

### Pipeline Architecture

```
Video Frames (30fps, 1080p)          IMU (30Hz, accel + gyro)
        |                                      |
   Undistort                            Gyro Integration
   (pinhole-radtan)                    (bias-corrected)
        |                                      |
   Feature Detection                           |
   (Shi-Tomasi, 500 pts)                       |
        |                                      |
   KLT Optical Flow                            |
   (forward-backward check)                    |
        |                                      |
   Essential Matrix                            |
   (RANSAC, 5-point)                           |
        |                                      |
   Translation Direction    Rotation (gyro)    |
        |                      |               |
        |                Fuse (80% gyro + 20% visual)
        |                      |
        |               Global Rotation
        |                      |
   Scale Estimation            |
   (pixel disp x scene depth) |
        |                      |
        +-------> Global Position <----+
                       |
                  6DoF Pose Output
                  (TUM format)
```

### Key Design Decisions

1. **Gyro-primary rotation**: IMU gyroscope is the main rotation source (80% weight), with visual Essential matrix correction (20%) to prevent long-term drift. Gyro is reliable at short timescales; visual degrades during fast head turns and motion blur.

2. **No accelerometer double-integration**: At 30Hz, accelerometer double-integration explodes due to gravity cancellation errors compounding quadratically. We use gyro for rotation only.

3. **Scale from optical flow**: Monocular vision has inherent scale ambiguity. We estimate scale from median pixel displacement / focal length * assumed scene depth (~3m for indoor factory environment). A running median buffer (90 frames) smooths this estimate.

4. **No PnP map anchoring**: We tried triangulating a 3D feature map and using PnP for scale recovery, but it caused the trajectory to "lock" in place — PnP anchors to the map built with the initial (possibly wrong) scale, creating a feedback loop. Pure odometry with heuristic scale is more robust.

## Calibration

| Parameter | Value |
|-----------|-------|
| Camera | 1920x1080, 30fps, 176 deg diagonal FOV |
| Model | Pinhole-radtan |
| Focal Length | fx=718.9, fy=716.3 |
| Principal Point | cx=960.0, cy=558.3 |
| Distortion | [-0.2818, 0.0739, 0.0003, 0.0009] |
| IMU | 30Hz, 3-axis accel + 3-axis gyro |
| Camera-IMU Extrinsic | T_cam_imu provided (see code) |
| Time Offset | 0.0s |

## Results

### Per-Clip Results (First 7 Clips)

| Clip | Path Length (m) | Avg Speed (m/s) | Net Displacement (m) | Drift Ratio |
|------|----------------|-----------------|---------------------|-------------|
| clip_0001 | 93.2 | 0.52 | 14.5 | 0.156 |
| clip_0002 | 165.6 | 0.92 | 12.5 | 0.076 |
| clip_0003 | 95.8 | 0.53 | 18.5 | 0.193 |
| clip_0004 | 184.6 | 1.03 | 29.6 | 0.160 |
| clip_0005 | 197.2 | 1.10 | 19.0 | 0.097 |
| clip_0006 | 159.9 | 0.89 | 2.3 | 0.014 |
| clip_0007 | 129.0 | 0.72 | 20.2 | 0.157 |

### Aggregate (All 87 Clips)

| Metric | Value |
|--------|-------|
| Clips processed | 87/87 (100%) |
| Avg path length | 164.8m per 3-min clip |
| Avg walking speed | 0.92 m/s |
| Mean drift ratio | 8.9% |
| VO success rate | 99.9% frames |

Walking speeds of 0.5-1.3 m/s are consistent with factory worker activity (walking, operating machinery, moving between stations).

## Validation (No Ground Truth)

Since Build AI provides no reference poses, we built a self-consistency validation suite (`validate.py`) with 7 automated checks. No ground truth is needed — these verify physical plausibility and cross-modal agreement.

### Validation Checks

| Check | What It Tests | Threshold |
|-------|--------------|-----------|
| SE(3) conformity | Valid rotation matrices, unit quaternions | `evo` library check |
| Speed plausibility | Factory worker: 0.3-2.0 m/s typical | Warn if avg < 0.05 or > 3.0 m/s |
| Height consistency | Head shouldn't fly or go underground | Warn if Z-range > 30m |
| Smoothness (jerk) | Humans can't teleport between frames | Warn if 99th pct jerk > 500 m/s^3 |
| Rotation rate | Head turns bounded by human physiology | Warn if > 20 rad/s |
| VIO vs Gyro-only | Cross-modal: does visual agree with inertial? | Warn if > 45 deg mean divergence |
| Path/duration sanity | Reasonable distance for 3-minute clip | Basic bounds check |

### Validation Results

```
SUMMARY: 27 passed, 60 warnings out of 87 clips
Avg path: 162.9m
Avg speed: 0.79 m/s
VIO vs Gyro rotation diff: 74.4 deg mean, 142.0 deg max
```

### Key Findings

**Translation (position) — Good:**
- 85/87 clips produce human-plausible walking speeds (0.3-1.6 m/s)
- Path lengths make sense for factory floor activity over 3 minutes
- No teleportation or NaN values

**Rotation (orientation) — Needs improvement:**
- 60/87 clips show > 45 degree divergence between VIO rotation and pure gyro integration
- This means the 20% visual rotation blending is introducing cumulative drift over 3 minutes that fights the gyro estimate
- The gyro alone may actually be more accurate for rotation in many clips

**Edge cases correctly handled:**
- **clip_0072 and clip_0073**: Near-zero speed detected (0.003 m/s). These are valid — the worker removed the camera glasses and placed them stationary. Pipeline correctly outputs minimal motion rather than hallucinating movement.
- **clip_0084**: 33m Z-range flagged. Likely a clip with extreme head motion or scale drift.

### What This Tells Us About Possible Improvements

Based on the validation findings, the main areas for improvement are:

**1. Rotation fusion weight should be adaptive, not fixed 80/20**

The current fixed 80% gyro / 20% visual blend causes problems because:
- When visual rotation is noisy (motion blur, low texture), the 20% visual contribution adds noise that compounds over 5400 frames
- Over 3 minutes, even a small per-frame error accumulates to 60-140 degrees

**Fix:** Use the Essential matrix inlier ratio as a confidence score. High inliers (> 80%) = trust visual more. Low inliers (< 40%) = trust gyro 100%. This way bad visual estimates don't corrupt the gyro-driven rotation.

**2. Gyro bias should be re-estimated periodically, not just at startup**

Currently we estimate gyro bias from the first 3 seconds and assume it's constant. But gyro bias drifts with temperature over an 8-hour shift. Clips later in the day (clip_0060+) show worse rotation accuracy.

**Fix:** Use visual rotation estimates during high-confidence periods (lots of inliers, slow motion) to continuously recalibrate the gyro bias. This is what production VIO systems (OpenVINS, VINS-Mono) do.

**3. Scale estimation could use triangulated scene depth instead of fixed 3m assumption**

The fixed 3m scene depth is a rough heuristic. Some factory areas may be open (10m+ depth) while others are narrow corridors (1-2m).

**Fix:** Triangulate features between keyframes to get actual scene depth measurements. Use the median triangulated depth instead of the hardcoded 3m. We tried this earlier but it was coupled with PnP (which caused locking). Triangulation for depth-only (without PnP position correction) would avoid that issue.

**4. Detect and handle degenerate cases (stationary camera, pure rotation)**

Clips 72-73 (camera placed down) and frames with pure head rotation (looking around without moving) break the Essential matrix decomposition because translation is near-zero.

**Fix:** Monitor feature displacement magnitude. Below a threshold (< 2 pixels median), skip translation update entirely and only accumulate gyro rotation. This prevents the Essential matrix from fitting noise when there's no real translational motion.

### Run Validation

```bash
# Full validation with VIO-vs-gyro comparison
python3 validate.py --output-dir output --data-dir data/worker_001

# Single clip
python3 validate.py --pose-file output/clip_0001_poses.txt --clip data/worker_001/clip_0001
```

## Dataset

87 clips from `gs://build-ai-egocentric-native-compression/worker_001`, each containing:
- `video.mp4` — 1920x1080, 30fps, 3 minutes (5400 frames)
- `imu.txt` — 30Hz accelerometer + gyroscope in JSON-lines format

Download:
```bash
gsutil -m cp -r gs://build-ai-egocentric-native-compression/worker_001 data/
```

## Usage

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python-headless numpy scipy
```

### Run Single Clip
```bash
python3 vio_pipeline.py --clip data/worker_001/clip_0001 --output-dir output
```

### Run All Clips
```bash
python3 run_all.py
```

### Evaluate
```bash
python3 evaluate.py --output-dir output
```

### Output Format

TUM format — one pose per line:
```
# timestamp tx ty tz qx qy qz qw
364.596678 0.000000 0.000000 0.000000 -0.234761 0.383749 -0.053901 0.891470
364.630863 0.327313 -0.387811 -0.861666 -0.233989 0.377299 -0.065520 0.893645
...
```

## What Was Tried and Discarded

| Approach | Issue | Resolution |
|----------|-------|------------|
| Accelerometer double-integration | Velocity diverged to 1000+ m/s at 30Hz | Use gyro for rotation only, skip accelerometer |
| PnP against triangulated map | Trajectory locked to initial map scale | Remove PnP, use pure odometry |
| Visual-only rotation | Degraded during fast head motion | Fuse 80% gyro + 20% visual |

## Dependencies

- Python 3.10+
- OpenCV (opencv-python-headless)
- NumPy
- SciPy
