from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

from ..wrapper import INDEX_TYPE, INVTT_TYPE, META_ID_TYPE, SIGNAL_TYPE, WEIGHT_TYPE
from .interface import ToastContainer

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

    def stage(self, ctnr: ToastContainer, purge: bool = True) -> None:
        """Stage (copy) TOAST data into the buffers.

        Args:
            ctnr: A ToastContainer instance wrapping the toast.Data object.
            purge: Whether to purge the data from the toast.Data object after staging.
        """
        # n_blocks = ctnr.n_local_blocks
        # n_samples = ctnr.n_local_samples
        data_size = ctnr.local_data_size
        # Communicate between processes to know the sizes on each of them
        self.data_size_proc = np.array(ctnr.allgather(data_size), dtype=INDEX_TYPE)

        # Stage the signal
        self.signal = ctnr.signal
        if purge:
            ...
        self.noise = ctnr.noise
        if purge:
            ...
        assert self.data_size_proc == self.signal.size == self.noise.size

    def __del__(self):
        for buffer in fields(self):
            del buffer
