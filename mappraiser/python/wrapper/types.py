from enum import IntEnum, auto

import numpy as np

__all__ = [
    'INDEX_TYPE',
    'INVTT_TYPE',
    'META_ID_TYPE',
    'SIGNAL_TYPE',
    'WEIGHT_TYPE',
    'GapStrategy',
    'PrecondType',
    'SolverType',
    # "TIMESTAMP_TYPE",
    # "PSD_TYPE",
]


SIGNAL_TYPE = np.float64
INDEX_TYPE = np.int32
WEIGHT_TYPE = np.float64
INVTT_TYPE = np.float64
META_ID_TYPE = np.uint64
TIMESTAMP_TYPE = np.float64
PSD_TYPE = np.float64


class GapStrategy(IntEnum):
    COND = 0
    MARG_LOCAL_SCAN = auto()
    NESTED_PCG = auto()
    NESTED_PCG_NO_GAPS = auto()
    MARG_PROC = auto()


class PrecondType(IntEnum):
    BJ = 0
    APRIORI = auto()
    APOSTERIORI = auto()


class SolverType(IntEnum):
    PCG = 0
    ECG = auto()
