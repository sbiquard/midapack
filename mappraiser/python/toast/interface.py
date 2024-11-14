import functools as ft
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import toast
from astropy import units as u
from toast.observation import default_values as defaults
from toast.ops import Operator, PixelsHealpix, StokesWeights
from toast.utils import name_UID

from .. import wrapper as lib
from .utils import interpolate_psd

MappraiserDtype = lib.SIGNAL_TYPE | lib.WEIGHT_TYPE | lib.INVTT_TYPE | lib.INDEX_TYPE
ValidPairDiffTransform = Literal['half-sub', 'add']


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

    @property
    def local_block_sizes(self) -> npt.NDArray[lib.INDEX_TYPE]:
        # FIXME: When we take flags into account, we need to update this
        return np.array([self.samples for _ in self.sdets], dtype=lib.INDEX_TYPE)

    @ft.cached_property
    def sdets(self) -> list[str]:
        """Return a list of the detector names"""
        return self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )

    @property
    def even_dets(self) -> list[str]:
        """Return a list of the even detector names"""
        return self.sdets[::2]

    @property
    def focalplane(self) -> toast.Focalplane:
        return self.observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Return the sampling rate (in Hz) of the data"""
        return self.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]

    @property
    def detector_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.array([name_UID(det, int64=True) for det in self.sdets], dtype=lib.META_ID_TYPE)

    def transform_pairs(
        self, a: npt.NDArray, operation: ValidPairDiffTransform = 'half-sub'
    ) -> npt.NDArray:
        if not self.pair_diff:
            return a
        # check that there is an even number of detectors
        assert a.shape[0] % 2 == 0
        if operation == 'half-sub':
            transformed = 0.5 * (a[::2] - a[1::2])
        elif operation == 'add':
            transformed = a[::2] + a[1::2]
        else:
            raise ValueError(f'Invalid operation {operation!r}')
        return transformed.astype(a.dtype)

    def get_signal(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        signal = np.array(
            self.observation.detdata[self.det_data][self.sdets, :], dtype=lib.SIGNAL_TYPE
        )
        if self.purge:
            del self.observation.detdata[self.det_data]
        return self.transform_pairs(signal)

    def get_noise(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        if self.noise_data is None:
            raise RuntimeError('Can not access noise without a field name')
        noise = np.array(
            self.observation.detdata[self.noise_data][self.sdets, :], dtype=lib.SIGNAL_TYPE
        )
        if self.purge:
            del self.observation.detdata[self.noise_data]
        return self.transform_pairs(noise)

    def get_indices(self, op: PixelsHealpix) -> npt.NDArray[lib.INDEX_TYPE]:
        # When doing pair differencing, we get the indices from the even detectors
        dets = self.even_dets if self.pair_diff else self.sdets
        indices = np.array(self.observation[op.pixels][dets, :], dtype=lib.INDEX_TYPE)
        if self.purge:
            del self.observation[op.pixels]
        # Arrange the pixel indices for Mappraiser
        indices = np.repeat(indices, nnz := self.nnz) * nnz
        for i in range(nnz):
            indices[i::nnz] += i
        return indices

    def get_weights(self, op: StokesWeights) -> npt.NDArray[lib.WEIGHT_TYPE]:
        weights = np.array(self.observation[op.weights][self.sdets, :], dtype=lib.WEIGHT_TYPE)
        if self.purge:
            del self.observation[op.weights]
        return self.transform_pairs(weights)

    def get_interp_psds(self, fft_size: int, rate: float = 1.0):
        """Return a 2-d array of interpolated PSDs for the selected detectors"""
        if self.noise_model is None:
            raise ValueError('Noise model not provided')
        model = self.observation[self.noise_model]
        psds = np.array(
            [
                interpolate_psd(
                    model.freq(det).to_value(u.Hz),  # pyright: ignore[reportAttributeAccessIssue]
                    model.psd(det).to_value(u.K**2 * u.second),  # pyright: ignore[reportAttributeAccessIssue]
                    fft_size=fft_size,
                    rate=rate,
                )
                for det in self.sdets
            ]
        )
        # Add the PSDs of the detectors in a pair when doing pair differencing
        # This means that we assume no correlations between them
        return self.transform_pairs(psds, operation='add')


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

    def get_signal(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        return np.concatenate([ob.get_signal() for ob in self._obs], axis=None)

    def get_noise(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        return np.concatenate([ob.get_noise() for ob in self._obs], axis=None)

    def get_pointing_indices(self, op: PixelsHealpix) -> npt.NDArray[lib.INDEX_TYPE]:
        return np.concatenate([ob.get_indices(op) for ob in self._synthesized_obs(op)], axis=None)

    def get_pointing_weights(self, op: StokesWeights) -> npt.NDArray[lib.WEIGHT_TYPE]:
        return np.concatenate([ob.get_weights(op) for ob in self._synthesized_obs(op)], axis=None)

    def get_interp_psds(self, fft_size: int, rate: float = 1.0) -> npt.NDArray:
        """Return a 2-d array of interpolated PSDs for the selected detectors"""
        if self.noise_model is None:
            raise ValueError('Noise model not provided')
        return np.vstack([ob.get_interp_psds(fft_size, rate) for ob in self._obs])

    def allgather(self, value: Any) -> list[Any]:
        assert (comm := self.data.comm.comm_world) is not None
        return comm.allgather(value)

    @property
    def n_local_samples(self) -> int:
        """Compute the number of local samples, summed over all observations"""
        return sum(ob.samples for ob in self._obs)

    @property
    def n_local_blocks(self) -> int:
        """Compute the number of local blocks, summed over all observations"""
        s = sum(len(ob.sdets) for ob in self._obs)
        if self.pair_diff:
            return s // 2
        return s

    @property
    def local_data_size(self) -> int:
        """Compute the size of the local signal buffer"""
        s = sum(ob.samples * len(ob.sdets) for ob in self._obs)
        if self.pair_diff:
            return s // 2
        return s

    @property
    def local_block_sizes(self) -> npt.NDArray[lib.INDEX_TYPE]:
        """Compute the local block sizes for each observation"""
        return np.concatenate([ob.local_block_sizes for ob in self._obs], axis=None)

    @property
    def telescope_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.array([ob.telescope.uid for ob in self.data.obs], dtype=lib.META_ID_TYPE)

    @property
    def session_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.array([ob.session.uid for ob in self.data.obs], dtype=lib.META_ID_TYPE)

    @property
    def detector_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.concatenate([ob.detector_uids for ob in self._obs], axis=None)

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
                _ = operator.apply(self.data.select(obs_uid=ob.observation.uid), detectors=ob.sdets)
            return ob

        return [synthesize(ob) for ob in self._obs]
