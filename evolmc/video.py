"""Turn the per-generation Pareto frames into a video.

Two encoders, tried in order:

  mp4 -- via ffmpeg if it is on PATH. Best quality, smallest file, and what you
         want for a slide or a supplementary-material upload.
  gif -- via Pillow, which ships with matplotlib. No external dependency, so
         this always works; larger files and a coarser palette.

Both are driven off `figures/pareto/gen_*.png`, so a video can be produced for
any run after the fact -- including runs made before this module existed.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

FRAME_GLOB = "gen_*.png"


def find_frames(frames_dir: str) -> list[str]:
    """Frame paths in generation order.

    Zero-padded names sort lexicographically into numeric order, which is also
    what `ffmpeg -i gen_%04d.png` assumes.
    """
    return sorted(glob.glob(os.path.join(frames_dir, FRAME_GLOB)))


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _encode_mp4(frames, out_path, fps, hold_last):
    """H.264 via ffmpeg's image2 demuxer, reading an explicit concat list.

    A concat list is used rather than `-i gen_%04d.png` so that gaps in the
    numbering (plot.every > 1) do not truncate the video at the first missing
    index.
    """
    listing = out_path + ".txt"
    per_frame = 1.0 / max(fps, 1)
    with open(listing, "w") as f:
        for i, p in enumerate(frames):
            dur = per_frame * (hold_last if i == len(frames) - 1 else 1)
            f.write(f"file '{os.path.abspath(p)}'\nduration {dur:.4f}\n")
        # The concat demuxer ignores the final entry's duration unless the last
        # file is repeated; without this the closing frame flashes past.
        f.write(f"file '{os.path.abspath(frames[-1])}'\n")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", listing,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dimensions
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(listing):
            os.remove(listing)
    return out_path


def _global_palette(images, max_side=240):
    """One 256-color palette representative of the *whole* sequence.

    Derived from a downsampled stack of every frame, not from the first one.
    A palette taken from frame 1 alone only covers the colors that happened to
    be on screen at generation 1: anything introduced later (a front line that
    did not exist yet, a marker in a fresh region) gets crushed onto the
    nearest existing entry, and near-identical frames can collapse entirely.
    Sharing one palette across frames is still what stops the animation
    shimmering, so the fix is a better palette rather than per-frame ones.
    """
    from PIL import Image

    thumbs = []
    for im in images:
        t = im.copy()
        t.thumbnail((max_side, max_side))
        thumbs.append(t)
    width = max(t.width for t in thumbs)
    stack = Image.new("RGB", (width, sum(t.height for t in thumbs)))
    y = 0
    for t in thumbs:
        stack.paste(t, (0, y))
        y += t.height
        t.close()
    return stack.quantize(colors=256, method=Image.MEDIANCUT)


def _encode_gif(frames, out_path, fps, hold_last):
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frames]
    pal = _global_palette(images)
    quantized = [im.quantize(palette=pal, dither=Image.NONE) for im in images]

    ms = int(1000 / max(fps, 1))
    durations = [ms] * len(quantized)
    durations[-1] = ms * hold_last
    # optimize=False: Pillow's optimizer drops frames it considers duplicates,
    # which silently desynchronises the animation from the generation count.
    # One frame in, one frame out.
    quantized[0].save(
        out_path, save_all=True, append_images=quantized[1:],
        duration=durations, loop=0, optimize=False, disposal=2,
    )
    for im in images:
        im.close()
    return out_path


def make_video(frames_dir, out_stem, fps=4, formats=("mp4", "gif"),
               hold_last=6, log=print):
    """Encode `frames_dir/gen_*.png` into `out_stem.<fmt>` for each format.

    Returns the paths written. Missing encoders are reported and skipped rather
    than raised -- a video is a convenience, and losing it must never fail a
    search that already produced its real artifacts.
    """
    frames = find_frames(frames_dir)
    if len(frames) < 2:
        log(f"  video skipped: found {len(frames)} frame(s) in {frames_dir}")
        return []

    written = []
    for fmt in formats:
        out_path = f"{out_stem}.{fmt}"
        try:
            if fmt == "mp4":
                if not has_ffmpeg():
                    log("  mp4 skipped: ffmpeg not on PATH (gif still written)")
                    continue
                _encode_mp4(frames, out_path, fps, hold_last)
            elif fmt == "gif":
                _encode_gif(frames, out_path, fps, hold_last)
            else:
                log(f"  unknown video format: {fmt}")
                continue
        except Exception as e:  # pragma: no cover - encoders vary by platform
            log(f"  {fmt} failed: {e}")
            continue
        size = os.path.getsize(out_path) / 2**20
        log(f"  {os.path.relpath(out_path)}  ({len(frames)} frames, {size:.1f} MB)")
        written.append(out_path)
    return written


def make_run_video(run, cfg, log=None):
    """Encode the Pareto frames of a finished run."""
    return make_video(
        frames_dir=run.file("figures", "pareto"),
        out_stem=run.file("figures", "pareto_evolution"),
        fps=cfg.plot.video_fps,
        formats=cfg.plot.video,
        log=log or run.log,
    )
