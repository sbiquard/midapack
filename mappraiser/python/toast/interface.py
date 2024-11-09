import functools as ft
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy import units as u

import toast
from mappraiser.python.wrapper.types import INDEX_TYPE, INVTT_TYPE, SIGNAL_TYPE, WEIGHT_TYPE
from toast.observation import default_values as defaults

MappraiserDtype = SIGNAL_TYPE | WEIGHT_TYPE | INVTT_TYPE | INDEX_TYPE


@dataclass
class ObservationData:
    observation: toast.Observation
    pair_diff: bool
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
    def samples(self) -> int:
        return self.observation.n_local_samples

    @ft.cached_property
    def dets(self) -> list[str]:
        """Return a list of the detector names"""
        dets = self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )
        # TODO: should we return only half of the detectors?
        return dets

    @property
    def focalplane(self) -> toast.Focalplane:
        assert isinstance(fp := self.observation.telescope.focalplane, toast.Focalplane)
        return fp

    @property
    def sample_rate(self) -> float:
        """Return the sampling rate (in Hz) of the data"""
        return self.focalplane.sample_rate.to_value(u.Hz)

    def do_pair_diff[T: MappraiserDtype](self, a: npt.NDArray[T]) -> npt.NDArray[T]:
        if not self.pair_diff:
            return a
        # check that there is an even number of detectors
        assert a.shape[0] % 2 == 0  # pyright: ignore[reportAny]
        return (0.5 * (a[::2] - a[1::2])).astype(a.dtype)

    @property
    def signal(self) -> npt.NDArray[SIGNAL_TYPE]:
        """Return the timestream data"""
        s = np.array(self.observation.detdata[self.det_data][self.dets, :], dtype=SIGNAL_TYPE)
        return self.do_pair_diff(s)

    @property
    def noise(self) -> npt.NDArray[SIGNAL_TYPE]:
        """Return the noise data"""
        if self.noise_data is None:
            raise RuntimeError('Can not access noise without a field name')
        n = np.array(self.observation.detdata[self.noise_data][self.dets, :], dtype=SIGNAL_TYPE)
        return self.do_pair_diff(n)

    def get_psd_model(self):
        """Returns frequencies and PSD values of the noise model."""
        if self.noise_model is None:
            raise ValueError('Noise model not provided.')
        model = self.observation[self.noise_model]
        freq = np.array([model.freq(det) for det in self.dets])
        psd = np.array([model.psd(det) for det in self.dets])
        return freq, psd


@dataclass
class ToastContainer:
    """A wrapper around the TOAST Data object with additional functionality."""

    data: toast.Data
    pair_diff: bool
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
    def signal(self) -> npt.NDArray[SIGNAL_TYPE]:
        return np.concatenate([ob.signal for ob in self.obs], axis=None)

    @property
    def noise(self) -> npt.NDArray[SIGNAL_TYPE]:
        return np.concatenate([ob.noise for ob in self.obs], axis=None)

    def allgather(self, value: Any) -> list[Any]:  # pyright: ignore[reportAny]
        assert (comm := self.data.comm.comm_world) is not None
        return comm.allgather(value)

    @property
    def n_local_blocks(self) -> int:
        """Compute the number of local blocks, summed over all observations"""
        s = sum(len(ob.dets) for ob in self.obs)
        if self.pair_diff:
            return s // 2
        return s

    @property
    def n_local_samples(self) -> int:
        """Compute the number of local samples, summed over all observations"""
        return sum(ob.samples for ob in self.obs)

    @property
    def local_data_size(self) -> int:
        """Compute the size of the local signal buffer"""
        s = sum(ob.samples * len(ob.dets) for ob in self.obs)
        if self.pair_diff:
            return s // 2
        return s

    @ft.cached_property
    def obs(self) -> list[ObservationData]:
        return [
            ObservationData(
                observation=ob,
                pair_diff=self.pair_diff,
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
