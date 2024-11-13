import warnings
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

from ..wrapper import INDEX_TYPE, INVTT_TYPE, META_ID_TYPE, SIGNAL_TYPE, WEIGHT_TYPE

__all__ = ['MappraiserBuffers']


@dataclass
class MappraiserBuffers:
    local_blocksizes: npt.NDArray[INDEX_TYPE] | None = None
    data_size_proc: npt.NDArray[INDEX_TYPE] | None = None
    signal: npt.NDArray[SIGNAL_TYPE] | None = None
    noise: npt.NDArray[SIGNAL_TYPE] | None = None
    pixels: npt.NDArray[INDEX_TYPE] | None = None
    pixweights: npt.NDArray[WEIGHT_TYPE] | None = None
    invntt: npt.NDArray[INVTT_TYPE] | None = None
    ntt: npt.NDArray[INVTT_TYPE] | None = None
    telescopes: npt.NDArray[META_ID_TYPE] | None = None
    obsindxs: npt.NDArray[META_ID_TYPE] | None = None
    detindxs: npt.NDArray[META_ID_TYPE] | None = None

    def enforce_contiguous(self):
        """Enforce that all buffers be contiguous in memory"""
        for field in fields(self):
            array = getattr(self, field.name)
            if array.flags['C_CONTIGUOUS']:
                continue
            # Warn the user that the buffer is not contiguous
            msg = f'Buffer {field.name!r} is not contiguous in memory'
            warnings.warn(msg, stacklevel=2)
            # Fix the problem
            setattr(self, field.name, np.ascontiguousarray(array))
