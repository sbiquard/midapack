import functools as ft
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy import units as u

import toast
from toast.observation import default_values as defaults
from toast.ops import Operator, PixelsHealpix, StokesWeights

from ..wrapper.types import INDEX_TYPE, INVTT_TYPE, SIGNAL_TYPE, WEIGHT_TYPE

MappraiserDtype = SIGNAL_TYPE | WEIGHT_TYPE | INVTT_TYPE | INDEX_TYPE


@dataclass
class ObservationData:
    observation: toast.Observation
    pair_diff: bool
    purge: bool
    det_selection: list[str] | None = None

    # fields that we want to copy
    det_data: str = defaults.det_data
    noise_data: str | None = 'noise'
    noise_model: str | None = defaults.noise_model

    # flagging
    det_mask: int = defaults.det_mask_nonscience
    det_flag_mask: int = defaults.det_mask_nonscience
    shared_flag_mask: int = defaults.shared_mask_nonscience
    det_flags: str = defaults.det_flags
    shared_flags: str = defaults.shared_flags

    @property
    def nnz(self) -> int:
        return 2 if self.pair_diff else 3

    @property
    def samples(self) -> int:
        return self.observation.n_local_samples

    @ft.cached_property
    def dets(self) -> list[str]:
        """Return a list of the detector names"""
        dets = self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )
        return dets

    @property
    def even_dets(self) -> list[str]:
        """Return a list of the even detector names"""
        return self.dets[::2]

    @property
    def focalplane(self) -> toast.Focalplane:
        return self.observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Return the sampling rate (in Hz) of the data"""
        return self.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]

    def do_pair_diff[T: MappraiserDtype](self, a: npt.NDArray[T]) -> npt.NDArray[T]:
        if not self.pair_diff:
            return a
        # check that there is an even number of detectors
        assert a.shape[0] % 2 == 0  # pyright: ignore[reportAny]
        return (0.5 * (a[::2] - a[1::2])).astype(a.dtype)

    def get_signal(self) -> npt.NDArray[SIGNAL_TYPE]:
        signal = np.array(self.observation.detdata[self.det_data][self.dets, :], dtype=SIGNAL_TYPE)
        if self.purge:
            del self.observation.detdata[self.det_data]
        return self.do_pair_diff(signal)

    def get_noise(self) -> npt.NDArray[SIGNAL_TYPE]:
        if self.noise_data is None:
            raise RuntimeError('Can not access noise without a field name')
        noise = np.array(self.observation.detdata[self.noise_data][self.dets, :], dtype=SIGNAL_TYPE)
        if self.purge:
            del self.observation.detdata[self.noise_data]
        return self.do_pair_diff(noise)

    def get_indices(self, op: PixelsHealpix) -> npt.NDArray[INDEX_TYPE]:
        # When doing pair differencing, we get the indices from the even detectors
        dets = self.even_dets if self.pair_diff else self.dets
        indices = np.array(self.observation[op.pixels][dets, :], dtype=INDEX_TYPE)
        if self.purge:
            del self.observation[op.pixels]
        # Arrange the pixel indices for Mappraiser
        indices = np.repeat(indices, nnz := self.nnz) * nnz
        for i in range(nnz):
            indices[i::nnz] += i
        return indices

    def get_weights(self, op: StokesWeights) -> npt.NDArray[WEIGHT_TYPE]:
        weights = np.array(self.observation[op.weights][self.dets, :], dtype=WEIGHT_TYPE)
        if self.purge:
            del self.observation[op.weights]
        return self.do_pair_diff(weights)

    # def get_psd_model(self):
    #     """Returns frequencies and PSD values of the noise model."""
    #     if self.noise_model is None:
    #         raise ValueError('Noise model not provided.')
    #     model = self.observation[self.noise_model]
    #     freq = np.array([model.freq(det) for det in self.dets])
    #     psd = np.array([model.psd(det) for det in self.dets])
    #     return freq, psd


@dataclass
class ToastContainer:
    """A wrapper around the TOAST Data object with additional functionality."""

    data: toast.Data
    pair_diff: bool
    purge: bool
    det_selection: list[str] | None = None

    # fields that we want to copy
    det_data: str = defaults.det_data
    noise_data: str | None = 'noise'
    noise_model: str | None = defaults.noise_model

    # flagging
    det_mask: int = defaults.det_mask_nonscience
    det_flag_mask: int = defaults.det_mask_nonscience
    shared_flag_mask: int = defaults.shared_mask_nonscience
    det_flags: str = defaults.det_flags
    shared_flags: str = defaults.shared_flags

    def get_signal(self) -> npt.NDArray[SIGNAL_TYPE]:
        return np.concatenate([ob.get_signal() for ob in self._obs], axis=None)

    def get_noise(self) -> npt.NDArray[SIGNAL_TYPE]:
        return np.concatenate([ob.get_noise() for ob in self._obs], axis=None)

    def get_pointing_indices(self, op: PixelsHealpix) -> npt.NDArray[INDEX_TYPE]:
        return np.concatenate([ob.get_indices(op) for ob in self._synthesized_obs(op)], axis=None)

    def get_pointing_weights(self, op: StokesWeights) -> npt.NDArray[WEIGHT_TYPE]:
        return np.concatenate([ob.get_weights(op) for ob in self._synthesized_obs(op)], axis=None)

    def allgather(self, value: Any) -> list[Any]:  # pyright: ignore[reportAny]
        assert (comm := self.data.comm.comm_world) is not None
        return comm.allgather(value)

    @property
    def n_local_blocks(self) -> int:
        """Compute the number of local blocks, summed over all observations"""
        s = sum(len(ob.dets) for ob in self._obs)
        if self.pair_diff:
            return s // 2
        return s

    @property
    def n_local_samples(self) -> int:
        """Compute the number of local samples, summed over all observations"""
        return sum(ob.samples for ob in self._obs)

    @property
    def local_data_size(self) -> int:
        """Compute the size of the local signal buffer"""
        s = sum(ob.samples * len(ob.dets) for ob in self._obs)
        if self.pair_diff:
            return s // 2
        return s

    @ft.cached_property
    def _obs(self) -> list[ObservationData]:
        return [
            ObservationData(
                observation=ob,
                pair_diff=self.pair_diff,
                purge=self.purge,
                det_selection=self.det_selection,
                det_data=self.det_data,
                noise_data=self.noise_data,
                noise_model=self.noise_model,
                det_mask=self.det_mask,
                det_flag_mask=self.det_flag_mask,
                shared_flag_mask=self.shared_flag_mask,
                det_flags=self.det_flags,
                shared_flags=self.shared_flags,
            )
            for ob in self.data.obs
        ]

    def _synthesized_obs(self, operator: Operator | None = None) -> list[ObservationData]:
        def synthesize(ob: ObservationData):
            if operator is not None:
                _ = operator.apply(self.data.select(obs_uid=ob.observation.uid), detectors=ob.dets)
            return ob

        return [synthesize(ob) for ob in self._obs]
