from dataclasses import InitVar, dataclass, fields
from typing import Any

from toast.data import Data
from toast.utils import dtype_to_aligned


@dataclass
class Buffer:
    raw: Any
    wrapped: Any

    @classmethod
    def create(cls, dtype, n_elements: int):
        storage, _ = dtype_to_aligned(dtype)
        raw = storage.zeros(n_elements)
        wrapped = raw.array()
        return cls(raw, wrapped)

    def __del__(self):
        if self.raw is not None:
            self.raw.clear()
        del self.raw
        del self.wrapped


@dataclass
class MappraiserBuffers:
    signal: Buffer | None = None
    blocksizes: Buffer | None = None
    detindxs: Buffer | None = None
    obsindxs: Buffer | None = None
    telescopes: Buffer | None = None
    noise: Buffer | None = None
    invtt: Buffer | None = None
    tt: Buffer | None = None
    pixels: Buffer | None = None
    pixweights: Buffer | None = None
    database: InitVar[Data | None] = None

    def __post_init__(self, database):
        """Initialize the buffers.

        This is where the data staging happens.

        Args:
            database (Data): The toast Data object containing the data.
        """
        if database is None:
            return

        # Copy the data into the buffers
        self.data_size_proc = None
        self.n_local_blocks = None
        raise NotImplementedError

    def __del__(self):
        # `fields` does not return init-only fields, so we only get the Buffer fields
        for buffer in fields(self):
            del buffer
