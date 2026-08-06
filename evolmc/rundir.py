"""Run directory layout.

Every run gets its own timestamped, self-describing directory under `logs/`:

    logs/20260804-142530__gpt2__nsga2-p24g20__k-type__p-global__uniform/
      config.yaml                 fully resolved config for this run
      meta.json                   model/hardware/timing facts
      logs/
        run.log                   human-readable progress log
        evals.jsonl               one line per fitness evaluation
        generations.jsonl         one line per generation
      data/
        baselines.csv             fp16 and uniform-K reference points
        history.npz               every X and F, per generation
        front.csv / front.json    final Pareto front
        results.csv               full re-evaluation (from run_eval.py)
      checkpoints/
        gen_0005.pkl ... latest.pkl
      figures/
        pareto/gen_0001.png|pdf   one frame per generation, fixed axes
        pareto_final.png|pdf
        convergence.png|pdf

Scripts take a run by directory name or path. There is no `latest` alias: the
directory names already say which run they are, and an alias that points
somewhere invisible is worse than typing the name.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from datetime import datetime

import yaml


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def default_run_name(cfg) -> str:
    """A directory name you can read without opening anything."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        stamp,
        _slug(cfg.model.name.split("/")[-1]),
        f"{cfg.search.algorithm}-p{cfg.search.pop_size}g{cfg.search.n_gen}",
        f"k-{cfg.variables.k_grouping}",
        f"p-{cfg.variables.prune_grouping if cfg.prune.enabled else 'off'}",
        _slug(cfg.quant.binning),
    ]
    return "__".join(parts)


class RunDir:
    def __init__(self, cfg, name: str | None = None, root: str | None = None,
                 reuse: bool = False):
        """Create (or, with `reuse`, re-open) a run directory.

        Configs that pin `log.run_name` -- as the experiment configs do, so the
        directory is predictable -- would otherwise append a second run's
        records onto the first's `generations.jsonl` and `evals.jsonl`, mixing
        two searches in one file with nothing to separate them. A fresh run
        therefore takes the next free `-2`, `-3`, ... suffix instead. `reuse` is
        for resuming, which genuinely does want to continue the same files.
        """
        self.cfg = cfg
        self.root = root or cfg.log.root
        self.name = name or cfg.log.run_name or default_run_name(cfg)
        self.path = os.path.join(self.root, self.name)
        if not reuse:
            self.name, self.path = self._free_name(self.name)

        for sub in ("logs", "data", "checkpoints", "figures",
                    os.path.join("figures", "pareto")):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)

        self._log_fh = open(self.file("logs", "run.log"), "a", buffering=1)
        self._t0 = time.perf_counter()
        self.save_config()

    def _free_name(self, name: str) -> tuple[str, str]:
        base = os.path.join(self.root, name)
        if not self._has_results(base):
            return name, base
        for n in range(2, 1000):
            cand = f"{name}-{n}"
            path = os.path.join(self.root, cand)
            if not self._has_results(path):
                print(f"note: {base} already holds a run; writing to {path}")
                return cand, path
        raise RuntimeError(f"too many runs named {name}")

    @staticmethod
    def _has_results(path: str) -> bool:
        return any(os.path.exists(os.path.join(path, *p)) for p in
                   (("logs", "generations.jsonl"), ("logs", "evals.jsonl"),
                    ("data", "front.json")))

    # -- paths -------------------------------------------------------------

    def file(self, *parts: str) -> str:
        return os.path.join(self.path, *parts)

    def frame(self, gen: int) -> str:
        """Stem for one generation's figure -- zero-padded so shell globs and
        `ffmpeg -i gen_%04d.png` both order them correctly."""
        return self.file("figures", "pareto", f"gen_{gen:04d}")

    def checkpoint(self, gen: int) -> str:
        return self.file("checkpoints", f"gen_{gen:04d}.pkl")

    # -- logging -----------------------------------------------------------

    def log(self, msg: str = "", echo: bool = True) -> None:
        line = f"[{time.perf_counter() - self._t0:8.1f}s] {msg}" if msg else ""
        self._log_fh.write(line + "\n")
        if echo:
            print(msg, flush=True)

    def jsonl(self, name: str, record: dict) -> None:
        with open(self.file("logs", f"{name}.jsonl"), "a") as f:
            f.write(json.dumps(record, default=float) + "\n")

    # -- artefacts ---------------------------------------------------------

    def save_config(self) -> None:
        with open(self.file("config.yaml"), "w") as f:
            yaml.safe_dump(self.cfg.to_dict(), f, sort_keys=False)

    def save_meta(self, **extra) -> None:
        meta = {
            "run_name": self.name,
            "started": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - self._t0, 2),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": sys.argv,
            **extra,
        }
        try:
            import torch

            meta["torch"] = torch.__version__
            if torch.cuda.is_available():
                meta["gpu"] = torch.cuda.get_device_name(0)
                meta["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30, 1
                )
                meta["gpu_peak_alloc_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 2
                )
        except Exception:  # pragma: no cover - reporting must never break a run
            pass
        with open(self.file("meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def close(self) -> None:
        try:
            self._log_fh.close()
        except Exception:  # pragma: no cover
            pass

    def __repr__(self) -> str:
        return f"RunDir({self.path})"


def find_run(spec: str, root: str = "logs") -> str:
    """Resolve a run directory from a path or a run name under `root`."""
    for candidate in (spec, os.path.join(root, spec)):
        if os.path.isdir(candidate):
            return candidate
    known = sorted(n for n in os.listdir(root)
                   if os.path.exists(os.path.join(root, n, "config.yaml"))) \
        if os.path.isdir(root) else []
    hint = ("\n  runs found: " + ", ".join(known)) if known else ""
    raise FileNotFoundError(
        f"no run directory matching {spec!r} under {root}/{hint}")
