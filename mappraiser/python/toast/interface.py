import functools as ft
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u

import toast
from toast.observation import default_values as defaults


@dataclass
class ObservationData:
    observation: toast.Observation
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
        """Returns a list of the detector names."""
        dets = self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )
        return dets

    @property
    def focalplane(self) -> toast.Focalplane:
        return self.observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        return self.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore

    def get_tods(self):
        """Returns the timestream data."""
        return np.array(self.observation.detdata[self.det_data][self.dets, :])

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

    def signal(self):
        raise NotImplementedError
        # TODO: Implement this method and the others below

    def noise(self):
        pass

    def allgather(self, value: Any) -> list[Any]:
        assert (comm := self.data.comm.comm_world) is not None
        return comm.allgather(value)

    def n_local_blocks(self) -> int:
        """Compute the number of local blocks, summed over all observations"""
        return sum(len(ob.dets) for ob in self.obs)

    def n_local_samples(self) -> int:
        """Compute the number of local samples, summed over all observations"""
        return sum(ob.samples for ob in self.obs)

    def local_data_size(self) -> int:
        """Compute the size of the local signal buffer"""
        return sum(ob.samples * len(ob.dets) for ob in self.obs)

    @ft.cached_property
    def obs(self) -> list[ObservationData]:
        return [
            ObservationData(
                observation=ob,
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
