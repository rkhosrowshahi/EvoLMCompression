#!/usr/bin/env python3
"""Encode a run's Pareto frames into a video.

Runs automatically at the end of every search; use this to re-encode at a
different frame rate, or to produce a video for a run made before the encoder
existed.

    python scripts/make_video.py latest
    python scripts/make_video.py latest --fps 8 --formats gif
    python scripts/make_video.py logs/20260804-142530__gpt2__nsga2-p24g20__k-type__p-global__uniform
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.rundir import find_run  # noqa: E402
from evolmc.video import find_frames, has_ffmpeg, make_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory, run name, or 'latest'")
    ap.add_argument("--root", default="logs")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--formats", default="mp4,gif")
    ap.add_argument("--hold-last", type=int, default=6,
                    help="hold the final frame for this many frame durations")
    ap.add_argument("--name", default="pareto_evolution")
    args = ap.parse_args()

    run_path = find_run(args.run, args.root)
    frames_dir = os.path.join(run_path, "figures", "pareto")
    frames = find_frames(frames_dir)
    if not frames:
        raise SystemExit(f"no gen_*.png frames in {frames_dir}")

    print(f"run    {run_path}")
    print(f"frames {len(frames)}  ({os.path.basename(frames[0])} .. "
          f"{os.path.basename(frames[-1])})")
    if not has_ffmpeg() and "mp4" in args.formats:
        print("note   ffmpeg not found; mp4 will be skipped, gif still works")
        print("       install with: brew install ffmpeg   (or apt install ffmpeg)")

    written = make_video(
        frames_dir=frames_dir,
        out_stem=os.path.join(run_path, "figures", args.name),
        fps=args.fps,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        hold_last=args.hold_last,
    )
    if not written:
        raise SystemExit("no video written")


if __name__ == "__main__":
    main()
