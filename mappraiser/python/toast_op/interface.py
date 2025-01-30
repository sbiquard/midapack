import functools as ft
from dataclasses import dataclass
from os.path import commonprefix
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import toast
from astropy import units as u
from toast.observation import default_values as defaults
from toast.ops import Operator, PixelsHealpix, StokesWeights
from toast.utils import name_UID

from .. import wrapper as lib
from .utils import interpolate_psd, pairwise

MappraiserDtype = lib.SIGNAL_TYPE | lib.WEIGHT_TYPE | lib.INVTT_TYPE | lib.INDEX_TYPE
ValidPairDiffTransform = Literal['half-sub', 'add']


@dataclass
class ObservationData:
    ob: toast.Observation
    nnz: int
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

    def __post_init__(self):
        # Run some checks
        if self.nnz not in (1, 2, 3):
            raise ValueError('nnz must be either 1, 2, or 3')

    @property
    def samples(self) -> int:
        return self.ob.n_local_samples

    @property
    def local_block_sizes(self) -> npt.NDArray[lib.INDEX_TYPE]:
        # FIXME: When we take flags into account, we need to update this
        return np.array([self.samples for _ in self.fdets], dtype=lib.INDEX_TYPE)

    @ft.cached_property
    def sdets(self) -> list[str]:
        """Return a list of the selected detector names"""
        return self.ob.select_local_detectors(selection=self.det_selection, flagmask=self.det_mask)

    @property
    def fdets(self) -> list[str]:
        """Return a list of the 'final' detector names (including pair diff logic)"""
        return self.sdets[::2] if self.pair_diff else self.sdets

    @property
    def fdets_prefix(self) -> list[str]:
        """Return a list of detector name prefixes (common part between even and odd ones).

        This method only makes sense if we are doing pair differencing.
        """
        if not self.pair_diff:
            raise ValueError('This method should only be called when doing pair differencing')
        return [commonprefix([a, b]) for a, b in pairwise(self.sdets)]

    @property
    def focalplane(self) -> toast.Focalplane:
        return self.ob.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Return the sampling rate (in Hz) of the data"""
        return self.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]

    @property
    def telescope_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        # NB: we duplicate the information on purpose
        return np.array([self.ob.telescope.uid for _ in self.fdets], dtype=lib.META_ID_TYPE)

    @property
    def session_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        # NB: we duplicate the information on purpose
        if (session := self.ob.session) is None:
            raise ValueError('Observation does not have a session attribute')
        return np.array([session.uid for _ in self.fdets], dtype=lib.META_ID_TYPE)

    @property
    def detector_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        dets = self.sdets if not self.pair_diff else self.fdets_prefix
        return np.array([name_UID(det, int64=True) for det in dets], dtype=lib.META_ID_TYPE)

    def transform_pairs(
        self, a: npt.NDArray, operation: ValidPairDiffTransform = 'half-sub'
    ) -> npt.NDArray:
        if not self.pair_diff:
            return a
        # check that there is an even number of detectors
        if a.shape[0] % 2 == 1:
            msg = 'Expected even number of detectors for pair differencing'
            raise RuntimeError(msg)
        if operation == 'half-sub':
            transformed = 0.5 * (a[::2] - a[1::2])
        elif operation == 'add':
            transformed = a[::2] + a[1::2]
        else:
            raise ValueError(f'Invalid operation {operation!r}')
        return transformed.astype(a.dtype)

    def get_signal(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        signal = np.array(self.ob.detdata[self.det_data][self.sdets, :], dtype=lib.SIGNAL_TYPE)
        if self.purge:
            del self.ob.detdata[self.det_data]
        return self.transform_pairs(signal)

    def get_noise(self) -> npt.NDArray[lib.SIGNAL_TYPE]:
        if self.noise_data is None:
            raise RuntimeError('Can not access noise without a field name')
        noise = np.array(self.ob.detdata[self.noise_data][self.sdets, :], dtype=lib.SIGNAL_TYPE)
        if self.purge:
            del self.ob.detdata[self.noise_data]
        return self.transform_pairs(noise)

    def get_indices(self, op: PixelsHealpix) -> npt.NDArray[lib.INDEX_TYPE]:
        # When doing pair differencing, we get the indices from the even detectors
        indices = np.array(self.ob.detdata[op.pixels][self.fdets, :], dtype=lib.INDEX_TYPE)
        if self.purge:
            del self.ob.detdata[op.pixels]
        # Arrange the pixel indices for Mappraiser
        indices = np.repeat(indices, nnz := self.nnz) * nnz
        for i in range(nnz):
            indices[i::nnz] += i
        return indices

    def get_weights(self, op: StokesWeights) -> npt.NDArray[lib.WEIGHT_TYPE]:
        weights = np.array(self.ob.detdata[op.weights][self.sdets, :], dtype=lib.WEIGHT_TYPE)
        # we always expect I/Q/U weights to be provided
        if weights.shape[-1] != 3:
            msg = 'Expected I/Q/U weights to be provided'
            raise RuntimeError(msg)
        if self.nnz == 1:
            # only I weights
            weights = weights[..., 0]
        elif self.nnz == 2:
            # only Q/U weights
            weights = weights[..., 1:]
        if self.purge:
            del self.ob.detdata[op.weights]
        return self.transform_pairs(weights)

    def get_interp_psds(self, fft_size: int, rate: float = 1.0):
        """Return a 2-d array of interpolated PSDs for the selected detectors"""
        if self.noise_model is None:
            raise ValueError('Noise model not provided')
        model = self.ob[self.noise_model]
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
    nnz: int
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
        assert (comm := self.data.comm.comm_world) is not None  # pyright assert
        return comm.allgather(value)

    @property
    def n_local_samples(self) -> int:
        """Compute the number of local samples, summed over all observations"""
        return sum(ob.samples for ob in self._obs)

    @property
    def n_local_blocks(self) -> int:
        """Compute the number of local blocks, summed over all observations"""
        s = sum(len(ob.fdets) for ob in self._obs)
        return s

    @property
    def local_data_size(self) -> int:
        """Compute the size of the local signal buffer"""
        s = sum(ob.samples * len(ob.fdets) for ob in self._obs)
        return s

    @property
    def local_block_sizes(self) -> npt.NDArray[lib.INDEX_TYPE]:
        """Compute the local block sizes for each observation"""
        return np.concatenate([ob.local_block_sizes for ob in self._obs], axis=None)

    @property
    def telescope_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.concatenate([ob.telescope_uids for ob in self._obs], axis=None)

    @property
    def session_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.concatenate([ob.session_uids for ob in self._obs], axis=None)

    @property
    def detector_uids(self) -> npt.NDArray[lib.META_ID_TYPE]:
        return np.concatenate([ob.detector_uids for ob in self._obs], axis=None)

    @property
    def observation_names(self) -> list[str]:
        # repetition is intentional
        return [ob.ob.name for ob in self._obs for _ in ob.fdets]  # pyright: ignore[reportReturnType]

    @property
    def detector_names(self) -> list[str]:
        # concatenation of fdets for each observation
        return [name for ob in self._obs for name in ob.fdets]

    @ft.cached_property
    def _obs(self) -> list[ObservationData]:
        return [
            ObservationData(
                ob=ob,
                nnz=self.nnz,
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
                _ = operator.apply(self.data.select(obs_uid=ob.ob.uid), detectors=ob.sdets)
            return ob

        return [synthesize(ob) for ob in self._obs]
