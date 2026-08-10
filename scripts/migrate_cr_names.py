#!/usr/bin/env python3
"""Rewrite stored runs to the cr_deploy / cr_archive names.

    cr_deployable -> cr_deploy     compression ratio, fixed-width indices, NO Huffman
    cr_archival   -> cr_archive    compression ratio, Huffman-coded

Reading old runs never needed this: `objectives.LEGACY_NAMES` resolves the old
spellings, so replot, compare_runs and backfill_results keep working untouched.
This is for when you want the stored files to say the new thing too, so a CSV
opened by hand or by a plotting script outside this repo reads consistently.

    python3 scripts/migrate_cr_names.py --dry-run        # report, change nothing
    python3 scripts/migrate_cr_names.py                  # migrate every run
    python3 scripts/migrate_cr_names.py --runs gpt2-k-block-3obj-np100-ng100
    python3 scripts/migrate_cr_names.py --revert         # back to the old names

WHAT IS TOUCHED, per run directory:

    config.yaml               search.objectives, search.report_metrics
    data/plot_box.json        the stored `objectives` list
    data/front.json           per-member `objectives` and `estimated` keys,
                              plus the embedded config
    data/front.csv            f<N>_cr_archival, est_cr_archival, est_cr_deployable
    data/results.csv          cr_archival, cr_deployable
    data/baselines.csv        cr_deployable and friends
    data/reprice.csv          cr_archival
    logs/evals.jsonl          per-evaluation summary keys (~5 MB per run)
    logs/generations.jsonl    per-generation records

NAMES ARE PREFIXED IN SOME FILES. front.csv writes objectives as `f3_cr_archival`
and estimates as `est_cr_archival`, so a bare key map misses them. Every rename
here goes through `_rename_token`, which strips a known prefix, maps the stem and
puts the prefix back -- and leaves anything it does not recognize alone.

SAFETY. Each file is rewritten only if its content actually changed, and only
after the replacement parses. `--dry-run` reports the exact per-file counts
first. Nothing outside the run directories is touched, and the figures are not
regenerated: run `scripts/replot.py` afterwards if you want axis labels redrawn.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolmc.objectives import LEGACY_NAMES  # noqa: E402

FORWARD = dict(LEGACY_NAMES)
BACKWARD = {v: k for k, v in LEGACY_NAMES.items()}

# Prefixes front.csv puts in front of an objective or estimate name. `f<N>_` is
# the objective index, `est_` marks the cost_only estimate. Order matters only in
# that the regex is anchored, so no prefix can swallow part of a stem.
_PREFIX = re.compile(r"^(f\d+_|est_|measured_)?(.*)$")


def _rename_token(token: str, mapping: dict[str, str]) -> str:
    """Map a possibly prefixed column or key name. Unknown stems pass through."""
    m = _PREFIX.match(token)
    prefix, stem = m.group(1) or "", m.group(2)
    return prefix + mapping[stem] if stem in mapping else token


def _rename_keys(obj, mapping):
    """Recursively rename dict KEYS, and any string VALUE that is a bare name.

    The string-value case is what catches `search.objectives: [.., cr_archival]`
    and plot_box's `objectives` list, where the name is data rather than a key.
    Only exact stems are replaced, so prose is never rewritten.
    """
    if isinstance(obj, dict):
        return {_rename_token(k, mapping): _rename_keys(v, mapping)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_keys(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def _sub_tokens(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    r"""Replace whole-word occurrences of every mapped name in raw text.

    Used for config.yaml and the .jsonl logs, where rewriting via a parser would
    reformat the whole file and bury the change in noise. `\b` keeps
    `cr_archival` from matching inside a longer identifier.
    """
    n = 0
    for old, new in mapping.items():
        text, k = re.subn(rf"\b{re.escape(old)}\b", new, text)
        n += k
    return text, n


# -- per-format handlers -------------------------------------------------------


def _csv_file(path, mapping):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    new_cols = [_rename_token(c, mapping) for c in cols]
    if new_cols == cols:
        return 0, None
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=new_cols, restval="")
    w.writeheader()
    for r in rows:
        w.writerow({_rename_token(k, mapping): v for k, v in r.items()
                    if k is not None})
    return sum(a != b for a, b in zip(cols, new_cols)), buf.getvalue()


def _json_file(path, mapping):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)
    out = json.dumps(_rename_keys(data, mapping), indent=2)
    # Count on the raw text: the structural diff is what changed, but the number
    # a reader wants is how many names moved.
    _, n = _sub_tokens(raw, mapping)
    return (n, out) if n else (0, None)


def _text_file(path, mapping):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    out, n = _sub_tokens(raw, mapping)
    return (n, out) if n else (0, None)


HANDLERS = [
    ("config.yaml", _text_file),
    ("data/plot_box.json", _json_file),
    ("data/front.json", _json_file),
    ("data/front.csv", _csv_file),
    ("data/results.csv", _csv_file),
    ("data/baselines.csv", _csv_file),
    ("data/reprice.csv", _csv_file),
    ("logs/evals.jsonl", _text_file),
    ("logs/generations.jsonl", _text_file),
]


def migrate_run(path, mapping, dry_run):
    """Returns [(relative file, names changed)] for whatever moved."""
    changed = []
    for rel, handler in HANDLERS:
        full = os.path.join(path, *rel.split("/"))
        if not os.path.exists(full):
            continue
        try:
            n, out = handler(full, mapping)
        except Exception as e:  # noqa: BLE001 - one bad file must not abort the sweep
            print(f"    !! {rel}: {type(e).__name__}: {e}")
            continue
        if not n:
            continue
        changed.append((rel, n))
        if not dry_run:
            with open(full, "w", newline="", encoding="utf-8") as f:
                f.write(out)
    return changed


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite stored runs to cr_deploy / cr_archive.")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run directory names; default is every run under --root")
    ap.add_argument("--root", default="logs")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and exit")
    ap.add_argument("--revert", action="store_true",
                    help="map the NEW names back to the old ones")
    args = ap.parse_args()

    mapping = BACKWARD if args.revert else FORWARD
    direction = "revert" if args.revert else "migrate"

    names = args.runs or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.root, "*"))
        if os.path.isdir(p))
    if not names:
        raise SystemExit(f"no run directories under {args.root}/")

    print(f"{direction}: " + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    print(f"{len(names)} run(s) under {args.root}/"
          + ("   [DRY RUN, nothing written]" if args.dry_run else ""))

    total_files = total_names = touched_runs = 0
    for name in names:
        path = os.path.join(args.root, name)
        changed = migrate_run(path, mapping, args.dry_run)
        if not changed:
            continue
        touched_runs += 1
        total_files += len(changed)
        total_names += sum(n for _, n in changed)
        print(f"\n  {name}")
        for rel, n in changed:
            print(f"    {rel:<26} {n:>7,} name(s)")

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {total_names:,} name(s) across {total_files} file(s) "
          f"in {touched_runs}/{len(names)} run(s)")
    if not args.dry_run and touched_runs:
        print("figures are NOT redrawn; run scripts/replot.py if you want the "
              "axis labels updated too")


if __name__ == "__main__":
    main()
