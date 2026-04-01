"""Batch process all clips, skipping incomplete downloads."""

import os
import sys
import time
from pathlib import Path

# Add CWD to path
sys.path.insert(0, str(Path(__file__).parent))
from vio_pipeline import VIOPipeline


def main():
    data_dir = Path("data/worker_001")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Also process data/clip_0001 if it exists
    clips = []
    if (Path("data/clip_0001") / "video.mp4").exists():
        clips.append(Path("data/clip_0001"))

    for clip in sorted(data_dir.glob("clip_*")):
        video = clip / "video.mp4"
        imu = clip / "imu.txt"
        # Skip if video not downloaded or still downloading
        if not video.exists():
            print(f"SKIP {clip.name}: no video.mp4")
            continue
        if not imu.exists():
            print(f"SKIP {clip.name}: no imu.txt")
            continue
        # Check file size (should be >1MB for a real video)
        if video.stat().st_size < 1_000_000:
            print(f"SKIP {clip.name}: video too small ({video.stat().st_size} bytes)")
            continue
        clips.append(clip)

    print(f"Processing {len(clips)} clips")

    for i, clip in enumerate(clips):
        out_file = out_dir / f"{clip.name}_poses.txt"
        # Skip if already processed with real data
        if out_file.exists() and out_file.stat().st_size > 1000:
            print(f"[{i+1}/{len(clips)}] SKIP {clip.name}: already processed")
            continue

        print(f"\n[{i+1}/{len(clips)}] {clip.name}")
        t0 = time.time()
        try:
            pipe = VIOPipeline()
            pipe.process_clip(clip / "video.mp4", clip / "imu.txt", out_file)
        except Exception as e:
            print(f"  ERROR: {e}")
        dt = time.time() - t0
        print(f"  Time: {dt:.1f}s")

    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
