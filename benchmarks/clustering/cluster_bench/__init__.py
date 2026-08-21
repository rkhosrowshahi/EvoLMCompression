"""Companding quantization vs. k-means on clustering benchmarks.

Self-contained on purpose: nothing here imports `evolmc`, and there is no torch
and no transformers in the dependency list. `companding.py` is a numpy port of
`evolmc/quantize.py` rather than a wrapper around it, so the benchmark can be
read, run and reviewed without the parent project -- and so the two can only
drift apart deliberately, not by accident.

See README.md for what is being compared and how to read the output.
"""

__all__ = ["baselines", "companding", "config", "datasets", "genome",
           "kmeans", "metrics", "problem", "report", "runner", "search"]
