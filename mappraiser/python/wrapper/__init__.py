"""
This package contains the wrapper around the Mappraiser C library.
"""

from .map import MLmap
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
    'GapStrategy',
    'PrecondType',
    'SolverType',
    'MLmap',
    'SIGNAL_TYPE',
    'INDEX_TYPE',
    'WEIGHT_TYPE',
    'INVTT_TYPE',
    'META_ID_TYPE',
    # "TIMESTAMP_TYPE",
    # "PSD_TYPE",
]
