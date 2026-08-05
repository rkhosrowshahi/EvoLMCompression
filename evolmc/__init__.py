"""EvoLMCompression -- multi-objective PTQ search for language models.

Codebook quantization + threshold-band pruning + entropy coding, with an
evolutionary algorithm choosing the per-layer codebook size and pruning band.
"""

from .codec import LayerCost, ModelCost, huffman_bits, price_layer, shannon_bits
from .compressor import Candidate, Compressor
from .config import Config
from .evaluate import perplexity, proxy_fitness, rank_correlation
from .grouping import Genome
from .problem import CompressionProblem
from .quantize import compress_layer
from .search import run_search, save_front
from .video import make_run_video, make_video

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Compressor",
    "Candidate",
    "Genome",
    "CompressionProblem",
    "ModelCost",
    "LayerCost",
    "compress_layer",
    "price_layer",
    "huffman_bits",
    "shannon_bits",
    "perplexity",
    "proxy_fitness",
    "rank_correlation",
    "run_search",
    "save_front",
    "make_video",
    "make_run_video",
]
