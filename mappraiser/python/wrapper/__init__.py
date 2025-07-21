"""
This package contains the wrapper around the Mappraiser C library.
"""

from .map import MLmap, remove_baseline, sim_constrained_block, sim_noise_tod
from .types import (
    INDEX_TYPE,
    INVTT_TYPE,
    META_ID_TYPE,
    SIGNAL_TYPE,
    WEIGHT_TYPE,
    # TIMESTAMP_TYPE,
    # PSD_TYPE,
    GapStrategy,
    PrecondType,
    SolverType,
)

__all__ = [
    'INDEX_TYPE',
    'INVTT_TYPE',
    'META_ID_TYPE',
    'SIGNAL_TYPE',
    'WEIGHT_TYPE',
    'GapStrategy',
    'MLmap',
    'PrecondType',
    'SolverType',
    # "TIMESTAMP_TYPE",
    # "PSD_TYPE",
    'remove_baseline',
    'sim_constrained_block',
    'sim_noise_tod',
]
