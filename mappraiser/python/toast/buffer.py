from dataclasses import dataclass

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
    invtt: npt.NDArray[INVTT_TYPE] | None = None
    tt: npt.NDArray[INVTT_TYPE] | None = None
    telescopes: npt.NDArray[META_ID_TYPE] | None = None
    obsindxs: npt.NDArray[META_ID_TYPE] | None = None
    detindxs: npt.NDArray[META_ID_TYPE] | None = None

    def check(self):
        """Check that the buffer sizes are consistent"""
        # TODO: Implement this method
        raise NotImplementedError
