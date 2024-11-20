from pathlib import Path
from typing import TypeAlias

import astropy.units as u
import numpy as np
import numpy.typing as npt
import tomlkit
import traitlets
from toast.data import Data as ToastData
from toast.observation import default_values as defaults
from toast.ops.operator import Operator as ToastOperator
from toast.ops.pixels_healpix import PixelsHealpix
from toast.ops.stokes_weights import StokesWeights
from toast.timing import Timer, function_timer
from toast.traits import Bool, Float, Instance, Int, Unicode, UseEnum, trait_docs
from toast.utils import Logger

from .. import wrapper as lib
from .buffer import MappraiserBuffers
from .interface import ToastContainer
from .utils import effective_ntt, estimate_psd, log_time_memory, next_fast_fft_size, psd_to_invntt

__all__ = [
    'MapMaker',
]

ObservationKeysDict: TypeAlias = dict[str, list[str]]


@trait_docs
class MapMaker(ToastOperator):
    """Operator that passes data to libmappraiser for map-making"""

    API = Int(0, help='Internal interface version for this operator')

    # Operators which we depend on
    pixel_pointing = Instance(klass=PixelsHealpix, allow_none=True, help='Operator to generate healpix indices')  # noqa # fmt: skip
    stokes_weights = Instance(klass=StokesWeights, allow_none=True, help='Operator to generate I/Q/U weights')  # noqa # fmt: skip

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
    zero_noise = Bool(False, help='Fill the noise buffer with zero')
    zero_signal = Bool(False, help='Fill the signal buffer with zero')

    # Underlying mappraiser configuration
    bs_red = Int(0, help='Use dynamic search reduction')
    enl_fac = Int(1, help='Enlargement factor for ECG')
    fill_gaps = Bool(True, help='Perform gap filling on the data')
    gap_stgy = UseEnum(lib.GapStrategy, help='Strategy for handling timestream gaps')
    maxiter = Int(3000, help='Maximum number of iterations allowed for the solver')
    ortho_alg = Int(1, help='Orthogonalization scheme for ECG (O->odir, 1->omin)')
    precond = UseEnum(lib.PrecondType, help='Preconditiner choice')
    ptcomm_flag = Int(6, help='Choose collective communication scheme')
    realization = Int(0, help='Noise realization index (for gap filling)')
    ref = Unicode('run0', help='Reference that is added to the name of the output maps')
    solver = UseEnum(lib.SolverType, help='Solver choice')
    tol = Float(1e-12, help='Convergence threshold for the iterative solver')
    z_2lvl = Int(0, help='Size of 2lvl deflation space')

    @traitlets.validate('stokes_weights')
    def _check_stokes_weights(self, proposal):
        # Check that Stokes weights operator provide I/Q/U weights
        weights = proposal['value']
        if weights is not None and weights.mode != 'IQU':
            msg = 'Mappraiser assumes that I, Q and U weights are provided'
            msg += f'but stokes_weights operator has {weights.mode=!r}'
            raise traitlets.TraitError(msg)
        return weights

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
    def _exec(self, data: ToastData, detectors: list[str] | None = None, **kwargs) -> None:
        """Run mappraiser on the supplied data object"""
        self._timer.start()

        # Get the global communicator
        self._comm = data.comm.comm_world
        if self._comm is None:
            raise RuntimeError('Mappraiser requires MPI, but the global communicator is None')

        # Setting up and staging the data
        self._log_memory(data, 'Before staging the data')
        self._prepare(data)
        self._stage(data, detectors)
        self._log_info('Staged data')

        # Call mappraiser
        self._log_memory(data, 'Before calling mappraiser')
        self._make_maps()
        self._log_info('Processed data')

    def _finalize(self, *args, **kwargs):
        pass

    @function_timer
    def _prepare(self, data: ToastData):
        """Examine the data and determine quantities needed to set up the mappraiser run"""
        # Check that we have at least one observation
        if len(data.obs) == 0:
            raise RuntimeError('Every supplied data object must contain at least one observation')

        # Check that the pixel_pointing and stokes_weights operators are set
        if self.pixel_pointing is None:
            raise RuntimeError('pixel_pointing operator must be set')
        if self.stokes_weights is None:
            raise RuntimeError('stokes_weights operator must be set')

        # Check if the noise data is available and set dependent traits
        if self.noise_data is None:
            self.zero_noise = True

        if self.binned:
            self.lagmax = 1

        # If not doing pair differencing, I/Q/U are all estimated
        if not self.pair_diff:
            self.estimate_spin_zero = True

        self._nnz = 3
        if not self.estimate_spin_zero:
            self._nnz = 2

        self.fsample = data.obs[0].telescope.focalplane.sample_rate.to_value(u.Hz)  # pyright: ignore[reportAttributeAccessIssue]

        # Populate the parameter dictionary passed to the C code
        self._params = {
            'bs_red': self.bs_red,
            'enl_fac': self.enl_fac,
            'fill_gaps': self.fill_gaps,
            'fsample': self.fsample,
            'gap_stgy': self.gap_stgy,
            'lambda': self.lagmax,
            'maxiter': self.maxiter,
            'nside': self.pixel_pointing.nside,  # pyright: ignore[reportAttributeAccessIssue]
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

        # Log the parameters that were used, creating the output directory if necessary
        outdir = Path(self.output_dir)
        if data.comm.world_rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / 'mappraiser_args_log.toml', 'w') as file:
                tomlkit.dump(self._params, file, sort_keys=True)

    @function_timer
    def _stage(self, data: ToastData, detectors: list[str] | None) -> None:
        """Copy the data to the Mappraiser buffers"""
        # Wrap the TOAST Data container into our custom class
        ctnr = ToastContainer(
            data,
            self._nnz,
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
        n_blocks = ctnr.n_local_blocks  # roughly n_obs * n_det
        block_sizes = ctnr.local_block_sizes  # sizes of the blocks
        data_size = ctnr.local_data_size  # total size (sum of block sizes)
        data_size_proc = np.array(ctnr.allgather(data_size), dtype=lib.INDEX_TYPE)

        # Metadata
        telescopes = ctnr.telescope_uids
        obsindxs = ctnr.session_uids
        detindxs = ctnr.detector_uids

        # Signal and noise
        signal = ctnr.get_signal()
        noise = ctnr.get_noise() / np.sqrt(self.downscale)

        # Pointing and weights
        pixels = ctnr.get_pointing_indices(self.pixel_pointing)  # pyright: ignore[reportArgumentType]
        weights = ctnr.get_pointing_weights(self.stokes_weights)  # pyright: ignore[reportArgumentType]

        # Inverse noise covariance
        invntt, ntt = self._get_invntt(ctnr, noise, block_sizes)

        # Check that sizes are consistent
        assert n_blocks == block_sizes.size
        assert n_blocks == telescopes.size
        assert n_blocks == obsindxs.size
        assert n_blocks == detindxs.size
        assert n_blocks == invntt.size // self.lagmax
        assert n_blocks == ntt.size // self.lagmax
        assert data_size == block_sizes.sum()
        assert data_size == signal.size
        assert data_size == noise.size
        assert data_size == pixels.size // self._nnz
        assert data_size == weights.size // self._nnz

        # The user may have requested to zero the signal and/or the noise
        if self.zero_signal:
            signal.fill(0)
        if self.zero_noise:
            noise.fill(0)

        self._buffers = MappraiserBuffers(
            local_blocksizes=block_sizes,
            data_size_proc=data_size_proc,
            signal=signal,
            noise=noise,
            pixels=pixels,
            pixweights=weights,
            invntt=invntt,
            ntt=ntt,
            telescopes=telescopes,
            obsindxs=obsindxs,
            detindxs=detindxs,
        )
        self._buffers.enforce_contiguous()

        if not self.plot_tod:
            return

        if data.comm.world_rank == 0:
            # Debug: make some plots of the data before running mappraiser
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].set_title('Signal')
            axs[1].set_title('Noise')
            acc = 0
            for i, size in enumerate(block_sizes):
                slc = slice(acc, acc + size)
                axs[0].plot(signal[slc], alpha=0.5, color='k')
                axs[1].plot(noise[slc], alpha=0.5, color='k')
                acc += size
            fig.tight_layout()
            plt.savefig(Path(self.output_dir) / 'signal_and_noise.png')

    def _get_invntt(
        self,
        ctnr: ToastContainer,
        noise: npt.NDArray[lib.SIGNAL_TYPE],
        block_sizes: npt.NDArray[lib.INDEX_TYPE],
    ) -> tuple[npt.NDArray[lib.INVTT_TYPE], npt.NDArray[lib.INVTT_TYPE]]:
        if any(self.lagmax > block_sizes):
            msg = 'Maximum lag should be less than the number of samples of any data block'
            raise RuntimeError(msg)
        invntt = np.empty((len(block_sizes), self.lagmax), lib.INVTT_TYPE)
        ntt = np.empty_like(invntt)
        if self.binned:
            # uniform weighting
            invntt.fill(1)
            ntt.fill(1)
        elif self.lagmax == 1:
            # simply use the variance of each block of the noise data
            # TODO: is the noise model is provided, use its 'noise weights' instead?
            acc = 0
            for i, block_size in enumerate(block_sizes):
                tod = noise[acc : acc + block_size]
                invntt[i], ntt[i] = 1 / (v := np.var(tod)), v
                acc += block_size
        else:
            fft_size = max(next_fast_fft_size(block_size) for block_size in block_sizes)
            if self.noise_model is None:
                # estimate the noise covariance from the data
                psds, success = estimate_psd(noise, block_sizes, fft_size, rate=self.fsample)
                # TODO: handle blocks where the estimation failed
            else:
                # interpolate the PSD from an existing Noise model
                psds = ctnr.get_interp_psds(fft_size, rate=self.fsample)
            invntt = psd_to_invntt(psds, self.lagmax)
            ntt = effective_ntt(invntt, fft_size)

        # Ultimately we want 1-d buffers
        return invntt.ravel(), ntt.ravel()

    @function_timer
    def _make_maps(self) -> None:
        """Make maps from buffered data"""
        lib.MLmap(
            self._comm,
            self._params,
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
            self._buffers.invntt,
            self._buffers.ntt,
        )

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

    def _provides(self) -> ObservationKeysDict:
        # We do not provide any data back to the pipeline
        return {}
