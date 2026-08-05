"""Video encoding tests.

The mp4 path is exercised with a stubbed `subprocess.run`, so the concat list
and the ffmpeg invocation are checked even on machines without ffmpeg
installed -- which is exactly where a silent breakage would go unnoticed.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc import video as V  # noqa: E402


@pytest.fixture
def frames(tmp_path):
    from PIL import Image

    d = tmp_path / "pareto"
    d.mkdir()
    for i in range(1, 6):
        Image.new("RGB", (40, 30), (10 * i, 60, 200)).save(d / f"gen_{i:04d}.png")
    # A stray file that is not a frame must be ignored.
    Image.new("RGB", (40, 30)).save(d / "pareto_final.png")
    return str(d)


def test_find_frames_orders_numerically_and_ignores_others(frames):
    got = [os.path.basename(p) for p in V.find_frames(frames)]
    assert got == [f"gen_{i:04d}.png" for i in range(1, 6)]


def test_gif_has_every_frame_and_holds_the_last(frames, tmp_path):
    from PIL import Image

    out = V.make_video(frames, str(tmp_path / "out"), fps=4,
                       formats=("gif",), hold_last=6, log=lambda *_: None)
    assert len(out) == 1
    gif = Image.open(out[0])
    assert gif.n_frames == 5
    assert gif.info.get("loop") == 0

    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info["duration"])
    assert durations[:-1] == [250] * 4  # 1000/fps
    assert durations[-1] == 250 * 6  # the closing frame lingers


def test_mp4_builds_a_concat_list_and_repeats_the_last_frame(frames, tmp_path,
                                                             monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        listing = cmd[cmd.index("-i") + 1]
        with open(listing) as f:
            captured["list"] = f.read()
        captured["cmd"] = cmd
        open(cmd[-1], "wb").write(b"\0" * 16)  # stand in for the encoded file
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(V.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(V.subprocess, "run", fake_run)

    out = V.make_video(frames, str(tmp_path / "out"), fps=4, formats=("mp4",),
                       hold_last=6, log=lambda *_: None)
    assert len(out) == 1

    listing = captured["list"]
    # Five frames, and the last repeated -- the concat demuxer drops the final
    # entry's duration otherwise, so the closing frame would flash past.
    assert listing.count("file '") == 6
    assert listing.count("gen_0005.png") == 2
    assert "duration 0.2500" in listing
    assert "duration 1.5000" in listing  # 0.25 * hold_last

    cmd = captured["cmd"]
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    # yuv420p rejects odd dimensions; the pad filter guarantees even ones.
    assert any("pad=ceil(iw/2)*2:ceil(ih/2)*2" in str(a) for a in cmd)
    assert not os.path.exists(str(tmp_path / "out.mp4") + ".txt")  # cleaned up


def test_missing_ffmpeg_skips_mp4_without_raising(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda _: None)
    msgs = []
    out = V.make_video(frames, str(tmp_path / "out"), formats=("mp4", "gif"),
                       log=msgs.append)
    # gif still gets written; losing a video must never fail a finished search.
    assert [os.path.splitext(p)[1] for p in out] == [".gif"]
    assert any("ffmpeg" in m for m in msgs)


def test_encoder_failure_is_reported_not_raised(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(V.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    msgs = []
    out = V.make_video(frames, str(tmp_path / "out"), formats=("mp4",),
                       log=msgs.append)
    assert out == []
    assert any("boom" in m for m in msgs)


def test_too_few_frames_is_a_no_op(tmp_path):
    from PIL import Image

    d = tmp_path / "pareto"
    d.mkdir()
    Image.new("RGB", (10, 10)).save(d / "gen_0001.png")
    msgs = []
    assert V.make_video(str(d), str(tmp_path / "out"), log=msgs.append) == []
    assert any("skipped" in m for m in msgs)
