import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

from toast.ops.memory_counter import MemoryCounter
from toast.utils import Logger, memreport


class UnreliableFit(BaseException):
    """Raised when the covariance matrix of the fit is poorly conditioned"""


def log_time_memory(data, timer=None, timer_msg=None, mem_msg=None, full_mem=False, prefix=''):
    """(This function is taken from madam_utils.py)"""
    log = Logger.get()
    data.comm.comm_world.barrier()
    restart = False

    if timer is not None:
        if timer.is_running():
            timer.stop()
            restart = True

        if data.comm.world_rank == 0:
            msg = f'{prefix} {timer_msg}: {timer.seconds():0.1f} s'
            log.debug(msg)

    if mem_msg is not None:
        # Dump toast memory use
        mem_count = MemoryCounter(silent=True)
        mem_count.total_bytes = 0
        toast_bytes = mem_count.apply(data)

        if data.comm.group_rank == 0 and toast_bytes is not None:  # silence basedpyright warning
            msg = f'{prefix} {mem_msg}'
            msg += f' Group {data.comm.group} memory = {toast_bytes / 1024**2:0.2f} GB'
            log.debug(msg)
        if full_mem:
            _ = memreport(msg=f'{prefix} {mem_msg}', comm=data.comm.comm_world)
    if restart and timer is not None:
        timer.start()


# def apo_window(lambd: int, kind: str = 'chebwin') -> np.ndarray:
#     if kind == 'gaussian':
#         # Apodization factor: cut happens at q_apo * sigma in the Gaussian window
#         q_apo = 3
#         window = ('general_gaussian', 1, 1 / q_apo * lambd)
#     elif kind == 'chebwin':
#         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html#scipy.signal.windows.chebwin
#         # Attenuation level (dB)
#         at = 150
#         window = ('chebwin', at)
#     else:
#         raise RuntimeError(f"Apodisation window '{kind}' is not supported.")

#     window = scipy.signal.get_window(window, 2 * lambd)
#     window = np.fft.ifftshift(window)[:lambd]
#     return window


# def compute_autocorrelations(
#     ob_uids,
#     dets,
#     mappraiser_noise,
#     local_block_sizes,
#     lambda_,
#     fsamp,
#     buffer_inv_tt,
#     buffer_tt,
#     invtt_dtype,
#     print_info=False,
#     save_psd=False,
#     save_dir='',
#     apod_window_type='chebwin',
# ):
#     """Compute the first lines of the blocks of the banded noise covariance"""
#     offset = 0
#     nobs = len(ob_uids)
#     for iob, uid in enumerate(ob_uids):
#         for idet, det in enumerate(dets):
#             blocksize = local_block_sizes[idet * nobs + iob]
#             nsetod = mappraiser_noise[offset : offset + blocksize]
#             slc = slice(
#                 (idet * nobs + iob) * lambda_,
#                 (idet * nobs + iob) * lambda_ + lambda_,
#                 1,
#             )
#             if lambda_ == 1:
#                 # no need for FFTs, just take the variance
#                 v = np.var(nsetod)
#                 buffer_inv_tt[slc] = 1 / v
#                 buffer_tt[slc] = v
#             else:
#                 buffer_inv_tt[slc], _ = noise_autocorrelation(
#                     nsetod,
#                     blocksize,
#                     lambda_,
#                     fsamp,
#                     idet,
#                     invtt_dtype,
#                     apod_window_type,
#                     verbose=(print_info and (idet == 0) and (iob == 0)),
#                     save_psd=save_psd,
#                     fname=os.path.join(save_dir, f'noise_fit_{uid}_{det}'),
#                 )
#                 buffer_tt[slc] = psd_to_autocorr(
#                     1 / compute_psd_eff(buffer_inv_tt[slc], blocksize), lambda_
#                 )
#             offset += blocksize
#     return


# def psd_model(f, sigma, alpha, fknee, fmin):
#     return sigma * (1 + ((f + fmin) / fknee) ** alpha)


# def logpsd_model(f, a, alpha, fknee, fmin):
#     return a + np.log10(1 + ((f + fmin) / fknee) ** alpha)


# def inversepsd_model(f, sigma, alpha, fknee, fmin):
#     return sigma * 1.0 / (1 + ((f + fmin) / fknee) ** alpha)


# def inverselogpsd_model(f, a, alpha, fknee, fmin):
#     return a - np.log10(1 + ((f + fmin) / fknee) ** alpha)


# def noise_autocorrelation(
#     nsetod,
#     nn,
#     lambda_,
#     fsamp,
#     idet,
#     invtt_dtype,
#     apod_window_type,
#     verbose=False,
#     save_psd=False,
#     fname='',
# ):
#     """Computes a periodogram from a noise timestream, and fits a PSD model
#     to it, which is then used to build the first row of a Toeplitz block.
#     """
#     # remove unit of fsamp to avoid problems when computing periodogram
#     try:
#         f_unit = fsamp.unit
#         fsamp = float(fsamp / (1.0 * f_unit))
#     except AttributeError:
#         pass

#     # Estimate psd from noise timestream

#     # Average over 10 minute segments
#     nperseg = int(600 * fsamp)

#     # Compute a periodogram with Welch's method
#     f, psd = scipy.signal.welch(nsetod, fsamp, nperseg=nperseg)

#     # Fit the psd model to the periodogram (in log scale)
#     popt, pcov = curve_fit(
#         logpsd_model,
#         f[1:],
#         np.log10(psd[1:]),
#         p0=np.array([-7, -1.0, 0.1, 1e-6]),
#         bounds=([-20, -10, 0.0, 0.0], [0.0, 0.0, 20, 1]),
#         maxfev=1000,
#     )

#     if verbose:
#         msg = f'\nPSD fit for det {idet}: '
#         msg += 'log(sigma2) = {:1.2f}, alpha = {:1.2f}, fknee = {:1.2f}, fmin = {:1.2e}\n'.format(
#             *tuple(popt)
#         )
#         print(msg)
#         print(f'PSD fit covariance for det {idet}', pcov, sep='\n', flush=True)

#     # Initialize full size inverse PSD in frequency domain
#     f_full = np.fft.rfftfreq(nn, 1.0 / fsamp)

#     # Compute inverse noise psd from fit and extrapolate (if needed) to lowest frequencies
#     ipsd_fit = inversepsd_model(f_full, 10 ** (-popt[0]), *popt[1:])
#     psd_fit = psd_model(f_full, 10 ** (popt[0]), *popt[1:])
#     psd_fit[0] = 0.0

#     # Compute inverse noise auto-correlation functions
#     inv_tt = np.fft.irfft(ipsd_fit, n=nn)
#     tt = np.fft.irfft(psd_fit, n=nn)

#     # Define apodization window
#     # Only allow max lambda = nn//2
#     if lambda_ > nn // 2:
#         msg = f'Bandwidth cannot be larger than timestream (lambda={lambda_}, {nn=}).'
#         raise RuntimeError(msg)

#     window = apo_window(lambda_, kind=apod_window_type)

#     # Apply window
#     inv_tt_w = np.multiply(window, inv_tt[:lambda_], dtype=invtt_dtype)
#     tt_w = np.multiply(window, tt[:lambda_], dtype=invtt_dtype)

#     # Optionally save some PSDs for plots
#     if save_psd:
#         noise_fit = {
#             'nn': nn,
#             'popt': popt,
#             'pcov': pcov,
#             'freqs': f,
#             'periodogram': psd,
#         }
#         np.savez(fname, **noise_fit)

#     return inv_tt_w, tt_w


# def compute_psd_eff(tt, fftlen) -> np.ndarray:
#     """
#     Computes the power spectral density from a given autocorrelation function.

#     :param tt: Input autocorrelation
#     :param m: FFT size

#     :return: The PSD (size = fft_size // 2 + 1 because we are using np.fft.rfft)
#     """
#     # Form the cylic autocorrelation
#     lag = len(tt)
#     circ_t = np.pad(tt, (0, fftlen - lag), 'constant')
#     if lag > 1:
#         circ_t[-lag + 1 :] = np.flip(tt[1:], 0)

#     # FFT
#     psd = np.abs(np.fft.rfft(circ_t, n=fftlen))
#     return psd


# def psd_to_autocorr(psd, lambda_: int, apo=True) -> np.ndarray:
#     """
#     Computes the autocorrelation function from a given power spectral density.

#     :param psd: Input PSD
#     :param lambd: Assumed noise correlation length
#     :param apo: if True, apodize the autocorrelation function

#     :return: The autocorrelation function apodised and cut after `lambda_` terms
#     """

#     # Compute the inverse FFT
#     autocorr = np.fft.irfft(psd)[:lambda_]

#     # Apodisation
#     if apo:
#         window = apo_window(lambda_)
#         autocorr *= window

#     return autocorr


def next_fast_fft_size(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


def interpolate_psd(freq, psd, fft_size: int, rate: float = 1.0):
    """Perform a logarithmic interpolation of PSD values."""
    interp_freq = np.fft.rfftfreq(fft_size, 1 / rate)
    # shift by fixed amounts in frequency and amplitude to avoid zeros
    freq_shift = rate / fft_size
    psd_shift = 0.01 * np.min(np.where(psd > 0, psd, 0))
    log_x = np.log10(interp_freq + freq_shift)
    log_xp = np.log10(freq + freq_shift)
    log_fp = np.log10(psd + psd_shift)
    interp_psd = np.interp(log_x, log_xp, log_fp)
    interp_psd = np.power(10.0, interp_psd) - psd_shift
    return interp_psd


def estimate_psd(noise, block_sizes, rate=1.0):
    """Estimate the PSD for each block of a noise timestream using Welch's method"""

    def func(tod, fft_size):
        # average the periodogram estimate over 10 minute segments
        nperseg = int(600 * rate)
        f, Pxx = welch(tod, fs=rate, nperseg=nperseg)
        # fit and compute full size PSD from fitted parameters
        params = fit_psd_model(f, Pxx)
        freq = np.fft.rfftfreq(fft_size, 1 / rate)
        psd = _model(freq, *params)
        return psd

    psd = np.empty_like(noise)
    success = np.ones_like(block_sizes, dtype=bool)
    acc = 0
    for i, block_size in enumerate(block_sizes):
        slc = slice(acc, acc + block_size)
        tod = noise[slc]
        try:
            psd[slc] = func(tod, next_fast_fft_size(block_size))
        except UnreliableFit:
            success[i] = False
        except Exception:
            raise
        acc += block_size
    return psd, success


def fit_psd_model(f, Pxx):
    """Fit a 1/f PSD model to the periodogram in log space"""
    init_params = np.array([1.0, -1.0, 0.1, 1e-5])
    final_params, pcov = curve_fit(
        _log_model,
        f[1:],
        np.log10(Pxx[1:]),
        p0=init_params,
        # bounds=([-20, -10, 0.0, 0.0], [0.0, 0.0, 20, 1]),
        maxfev=1000,
    )
    if np.linalg.cond(pcov) > 1e10:
        # TODO: what threshold to use?
        raise UnreliableFit
    return final_params


def _log_model(x, sigma, alpha, fk, f0):
    log_psd = 2 * np.log10(sigma) + np.log10(1 + ((x + f0) / fk) ** alpha)
    return log_psd


def _model(x, sigma, alpha, fk, f0):
    psd = sigma**2 * (1 + ((x + f0) / fk) ** alpha)
    return psd


# def psd_to_invntt(psd, correlation_length):
#     """Compute the inverse autocorrelation function from PSD values.
#     The result is apodized and cut at the specified correlation length.
#     """
#     invntt = np.fft.irfft(1 / psd)[..., :correlation_length]
#     window = apodization_window(correlation_length)
#     return invntt * window


# def apodization_window(size: int, kind: str = 'chebwin'):
#     if kind == 'gaussian':
#         q_apo = 3  # apodization factor: cut happens at q_apo * sigma in the Gaussian window
#         window_type = ('general_gaussian', 1, 1 / q_apo * size)
#     elif kind == 'chebwin':
#         at = 150  # attenuation level (dB)
#         window_type = ('chebwin', at)
#     else:
#         raise RuntimeError(f'Apodization window {kind!r} is not supported.')

#     window = np.array(get_window(window_type, 2 * size))
#     window = np.fft.ifftshift(window)[:size]
#     return window
