from pathlib import Path
from typing import final, override

import astropy.units as u
import numpy as np
import tomlkit
import traitlets

from toast.data import Data as ToastData
from toast.observation import default_values as defaults
from toast.ops.operator import Operator as ToastOperator
from toast.ops.pixels_healpix import PixelsHealpix
from toast.ops.stokes_weights import StokesWeights
from toast.timing import Timer, function_timer
from toast.traits import Bool, Dict, Float, Instance, Int, Unicode, UseEnum, trait_docs
from toast.utils import Logger

from .. import wrapper as lib
from .buffer import MappraiserBuffers
from .interface import ToastContainer
from .utils import log_time_memory

__all__ = ['MapMaker', 'available']

type ObservationKeysDict = dict[str, list[str]]


def available():
    """Return True if libmappraiser is found in the library search path"""
    return lib.available


@final
@trait_docs
class MapMaker(ToastOperator):
    """Operator that passes data to libmappraiser for map-making"""

    API = Int(0, help='Internal interface version for this operator')

    # Operators which we depend on
    pixel_pointing = Instance(klass=PixelsHealpix, help='Operator to generate healpix indices')
    stokes_weights = Instance(klass=StokesWeights, help='Operator to generate I/Q/U weights')

    # TOAST names
    det_data = Unicode(defaults.det_data, help='Observation detdata key for the timestream data')
    noise_data = Unicode("noise", allow_None=True, help="Observation detdata key for the noise data")  # noqa # fmt: skip
    noise_model = Unicode(defaults.noise_model, help='Observation key containing the noise model')

    # Flagging and masking
    det_mask = Int(defaults.det_mask_nonscience, help='Bit mask value for per-detector flagging')
    det_flag_mask = Int(defaults.det_mask_nonscience, help="Bit mask value for detector sample flagging")  # noqa # fmt: skip
    det_flags = Unicode(defaults.det_flags, help='Observation detdata key for flags to use')
    shared_flag_mask = Int(defaults.shared_mask_nonscience, help="Bit mask value for shared flagging")  # noqa # fmt: skip
    shared_flags = Unicode(defaults.shared_flags, help="Observation shared key for telescope flags to use")  # noqa # fmt: skip

    # General configuration
    binned = Bool(False, help='Make a binned map')
    downscale = Int(1, help='Downscale the noise by sqrt of this factor')
    estimate_psd = Bool(False, help='Estimate the noise PSD from the data')
    estimate_spin_zero = Bool(False, help='When doing pair-diff, still estimate a spin-zero field')
    lagmax = Int(1_000, help='Maximum lag of the correlation function')
    mem_report = Bool(False, help='Print memory reports')
    nside = Int(64, help='HEALPix nside parameter for the output maps')
    output_dir = Unicode('.', help='Write output data products to this directory')
    pair_diff = Bool(False, help='Process differenced timestreams')
    plot_tod = Bool(False, help='Plot the signal+noise TOD after staging')
    purge_det_data = Bool(True, help='Clear all observation detector data after staging')
    ref = Unicode('run0', help='Reference that is added to the name of the output maps')
    zero_noise = Bool(False, help='Fill the noise buffer with zero')
    zero_signal = Bool(False, help='Fill the signal buffer with zero')

    # Underlying mappraiser configuration
    bs_red = Int(0, help='Use dynamic search reduction')
    enlFac = Int(1, help='Enlargement factor for ECG')
    fill_gaps = Bool(True, help='Perform gap filling on the data')
    gap_stgy = UseEnum(lib.GapStrategy, help='Strategy for handling timestream gaps')
    maxiter = Int(3000, help='Maximum number of iterations allowed for the solver')
    ortho_alg = Int(1, help='Orthogonalization scheme for ECG (O->odir, 1->omin)')
    params = Dict(default_value={}, help='Parameters to pass to mappraiser')
    precond = UseEnum(lib.PrecondType, help='Preconditiner choice')
    ptcomm_flag = Int(6, help='Choose collective communication scheme')
    realization = Int(0, help='Noise realization index (for gap filling)')
    solver = UseEnum(lib.SolverType, help='Solver choice')
    tol = Float(1e-12, help='Convergence threshold for the iterative solver')
    z_2lvl = Int(0, help='Size of 2lvl deflation space')

    @traitlets.validate('stokes_weights')
    def _check_stokes_weights(self, proposal):
        # Check that Stokes weights operator provide I/Q/U weights
        check = proposal['value']
        if check.mode != 'IQU':
            msg = 'Mappraiser assumes that I, Q and U weights are provided'
            msg += f'but stokes_weights operator has {check.mode=!r}'
            raise RuntimeError(msg)
        return check

    @traitlets.validate('det_mask')
    def _check_det_mask(self, proposal):
        check = proposal['value']
        if check < 0:
            raise traitlets.TraitError('Det mask should be a positive integer')
        return check

    @traitlets.validate('det_flag_mask')
    def _check_det_flag_mask(self, proposal):
        check = proposal['value']
        if check < 0:
            raise traitlets.TraitError('Det flag mask should be a positive integer')
        return check

    @traitlets.validate('shared_flag_mask')
    def _check_shared_flag_mask(self, proposal):
        check = proposal['value']
        if check < 0:
            raise traitlets.TraitError('Shared flag mask should be a positive integer')
        return check

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._logprefix = 'Mappraiser:'
        self._logger = Logger.get()
        self._timer = Timer()

    def clear(self) -> None:
        del self._buffers

    def __del__(self) -> None:
        self.clear()

    def _log_info(self, msg: str) -> None:
        self._logger.info_rank(
            f'{self._logprefix} {msg} in',
            comm=self._comm,  # None is also a valid argument here
            timer=self._timer,
        )

    def _log_memory(self, data: ToastData, msg: str) -> None:
        log_time_memory(
            data,
            prefix=self._logprefix,
            mem_msg=msg,
            full_mem=self.mem_report,
        )

    @function_timer
    @override
    def _exec(self, data: ToastData, detectors: list[str] | None = None, **kwargs) -> None:
        """Run mappraiser on the supplied data object"""
        if not available():
            raise RuntimeError('Mappraiser is either not installed or MPI is disabled')

        self._timer.start()

        # Setting up and staging the data
        self._log_memory(data, 'Before staging the data')
        self._prepare(data)
        self._stage(data, detectors)
        self._log_info('Staged data')

        # Call mappraiser
        self._log_memory(data, 'Before calling mappraiser')
        self._make_maps()
        self._log_info('Processed data')

    @function_timer
    def _prepare(self, data: ToastData):
        """Examine the data and determine quantities needed to set up the mappraiser run"""
        # Check that we have at least one observation
        if len(data.obs) == 0:
            raise RuntimeError('Every supplied data object must contain at least one observation')

        # Get the global communicator
        self._comm = data.comm.comm_world

        # Check if the noise data is available and set dependent traits
        if self.noise_data is None:
            self.zero_noise = True

        if self.binned:
            self.lagmax = 1

        nnz_full = len(self.stokes_weights.mode)  # 3, but keep it for future expansion
        if self.pair_diff:
            if self.estimate_spin_zero:
                self._nnz = nnz_full
            else:
                self._nnz = nnz_full - 1
        else:
            self._nnz = nnz_full

        # If not doing pair differencing, I/Q/U are all estimated
        if not self.pair_diff:
            self.estimate_spin_zero = True

        self.fsample = data.obs[0].telescope.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore[reportAttributeAccessIssue]

        # Populate the parameter dictionary passed to the C code
        self.params.update(
            {
                'bs_red': self.bs_red,
                'enlFac': self.enlFac,
                'fill_gaps': self.fill_gaps,
                'fsample': self.fsample,
                'gap_stgy': self.gap_stgy,
                'lambda': self.lagmax,
                'maxiter': self.maxiter,
                'nside': self.pixel_pointing.nside,
                'ortho_alg': self.ortho_alg,
                'output_dir': self.output_dir,
                'precond': self.precond,
                'ptcomm_flag': self.ptcomm_flag,
                'realization': self.realization,
                'ref': self.ref,
                'solver': self.solver,
                'tol': self.tol,
                'Z_2lvl': self.z_2lvl,
            }
        )

        # Log the parameters that were used, creating the output directory if necessary
        self.output_dir = Path(self.output_dir)
        if data.comm.world_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / 'mappraiser_args_log.toml', 'w') as file:
                tomlkit.dump(self.params, file, sort_keys=True)

    @function_timer
    def _stage(self, data: ToastData, detectors: list[str] | None) -> None:
        """Copy the data to the Mappraiser buffers"""
        # Wrap the TOAST Data container into our custom class
        ctnr = ToastContainer(
            data,
            self.pair_diff,
            self.purge_det_data,
            det_selection=detectors,
            det_data=self.det_data,
            noise_data=self.noise_data,
            noise_model=self.noise_model,
            det_mask=self.det_mask,
            det_flag_mask=self.det_flag_mask,
            shared_flag_mask=self.shared_flag_mask,
            det_flags=self.det_flags,
            shared_flags=self.shared_flags,
        )

        # Get data distribution information
        n_blocks = ctnr.n_local_blocks
        block_sizes = ctnr.local_block_sizes
        data_size = ctnr.local_data_size
        data_size_proc = np.array(ctnr.allgather(data_size), dtype=lib.INDEX_TYPE)

        # Metadata
        telescopes = ctnr.telescope_uids
        obsindxs = ctnr.session_uids
        detindxs = ctnr.detector_uids

        # Signal and noise
        signal = ctnr.get_signal()
        noise = ctnr.get_noise()

        # Pointing and weights
        pixels = ctnr.get_pointing_indices(self.pixel_pointing)
        weights = ctnr.get_pointing_weights(self.stokes_weights)

        # Inverse noise covariance
        if self.binned:
            # we are making a binned map, so lagmax is 1
            invntt = np.ones(n_blocks, dtype=lib.INVTT_TYPE)
            tt = invntt
        elif self.lagmax > any(block_sizes):
            msg = 'Maximum lag should be less than the number of samples of any data block'
            raise RuntimeError(msg)
        else:
            # TODO: finish this part
            if self.noise_model is None:
                # estimate the noise covariance from the data
                psd = estimate_psd(noise, block_sizes, rate=self.fsample)
            else:
                # use an existing Noise model
                freq, psd = ctnr.get_psd_model()
                psd = interpolate_psd(freq, psd, rate=self.fsample)
            invntt = psd_to_invntt(psd, self.lagmax)
            tt = ...

        self._buffers = MappraiserBuffers(
            local_blocksizes=block_sizes,
            data_size_proc=data_size_proc,
            signal=signal,
            noise=noise,
            pixels=pixels,
            pixweights=weights,
            invtt=invntt,
            tt=tt,
            telescopes=telescopes,
            obsindxs=obsindxs,
            detindxs=detindxs,
        )
        self._buffers.check()

        # if self.estimate_psd:
        #     # Compute autocorrelation from the data
        #     ob_uids = [ob.uid for ob in data.obs]
        #     det_names = all_dets if not self.pair_diff else [name[:-2] for name in all_dets[::2]]
        #     assert nobs == len(ob_uids)
        #     assert ndet == len(det_names)
        #     compute_autocorrelations(
        #         ob_uids,
        #         det_names,
        #         self._mappraiser_noise,
        #         self._mappraiser_blocksizes,
        #         lambda_,
        #         fsample,
        #         self._mappraiser_invtt,
        #         self._mappraiser_tt,
        #         libmappraiser.INVTT_TYPE,
        #         apod_window_type=self.apod_window_type,
        #         print_info=(data.comm.world_rank == 0),
        #         save_psd=(self.save_psd and data.comm.world_rank == 0),
        #         save_dir=os.path.join(params['path_output'], 'psd_fits'),
        #     )
        #     return

        # # Use existing noise model instead of estimating from the data

        # def _slice(idet):
        #     return slice(
        #         (idet * nobs + iob) * lambda_,
        #         (idet * nobs + iob) * lambda_ + lambda_,
        #     )

        # def _get_freq_psd(model, det):
        #     freq = model.freq(det).to_value(u.Hz)
        #     psd = model.psd(det).to_value(u.K**2 * u.second)
        #     return freq, psd

        # def _populate_buffers(idet, psd):
        #     ipsd = 1 / psd
        #     ipsd[0] = 0
        #     slc = _slice(idet)
        #     self._mappraiser_invtt[slc] = psd_to_autocorr(ipsd, lambda_, apo=True)
        #     self._mappraiser_tt[slc] = psd_to_autocorr(psd, lambda_, apo=True)

        # for iob, ob in enumerate(data.obs):
        #     # Get the fitted noise object for this observation.
        #     model = ob[self.noise_model]
        #     nsamp = ob.n_local_samples
        #     if lambda_ > nsamp:
        #         msg = 'Maximum correlation length should be less than the number of samples'
        #         msg += f' (got lambda={lambda_}, nsamp={nsamp})'
        #         raise RuntimeError(msg)
        #     fft_size = next_fast_fft_size(nsamp)
        #     local_dets: list[str] = ob.select_local_detectors(flagmask=self.det_mask)
        #     if self.pair_diff:
        #         for idet, (det1, det2) in enumerate(pairwise(local_dets)):
        #             if lambda_ == 1:
        #                 w1 = model.detector_weight(det1)
        #                 w2 = model.detector_weight(det2)
        #                 w = w1 + w2  # assume no correlation between detectors
        #                 self._mappraiser_invtt[idet * nobs + iob] = w.value
        #                 self._mappraiser_tt[idet * nobs + iob] = 1 / w.value
        #             else:
        #                 freq1, psd1 = _get_freq_psd(model, det1)
        #                 freq2, psd2 = _get_freq_psd(model, det2)
        #                 psd1 = interpolate_psd(freq1, psd1, fft_size=fft_size, rate=fsample)
        #                 psd2 = interpolate_psd(freq2, psd2, fft_size=fft_size, rate=fsample)
        #                 psd = psd1 + psd2  # assume no correlation between detectors
        #                 _populate_buffers(idet, psd)
        #     else:
        #         for idet, det in enumerate(local_dets):
        #             if lambda_ == 1:
        #                 w = model.detector_weight(det)
        #                 self._mappraiser_invtt[idet * nobs + iob] = w.value
        #                 self._mappraiser_tt[idet * nobs + iob] = 1 / w.value
        #             else:
        #                 freq, psd = _get_freq_psd(model, det)
        #                 psd = interpolate_psd(freq, psd, fft_size=fft_size, rate=fsample)
        #                 _populate_buffers(idet, psd)

    @function_timer
    def _make_maps(self) -> None:
        """Make maps from buffered data"""
        lib.MLmap(
            self._comm,
            self.params,
            self._buffers.data_size_proc,
            self._buffers.local_blocksizes,
            self._buffers.detindxs,
            self._buffers.obsindxs,
            self._buffers.telescopes,
            self._nnz,
            self._buffers.pixels,
            self._buffers.pixweights,
            self._buffers.signal,
            self._buffers.noise,
            self._buffers.invtt,
            self._buffers.tt,
        )

    @override
    def _finalize(self, data: ToastData, **kwargs) -> None:
        self.clear()

    @override
    def _requires(self) -> ObservationKeysDict:
        req = {
            'meta': [self.noise_model],
            'detdata': [self.det_data],
            'intervals': [],
            'shared': [],
        }
        if self.noise_data is not None:
            req['detdata'].append(self.noise_data)
        if self.shared_flags is not None:
            req['shared'].append(self.shared_flags)
        if self.det_flags is not None:
            req['detdata'].append(self.det_flags)
        return req

    @override
    def _provides(self) -> ObservationKeysDict:
        # We do not provide any data back to the pipeline
        return {}
