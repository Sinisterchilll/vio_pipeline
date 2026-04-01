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
