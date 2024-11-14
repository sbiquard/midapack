import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from scipy.signal import get_window, welch
from toast.data import Data as ToastData
from toast.ops.memory_counter import MemoryCounter
from toast.timing import Timer as ToastTimer
from toast.utils import Logger, memreport

from .. import wrapper as lib


def log_time_memory(
    data: ToastData,
    timer: ToastTimer = None,
    timer_msg: str | None = None,
    mem_msg: str | None = None,
    full_mem: bool = False,
    prefix: str = '',
) -> None:
    """(This function is taken from madam_utils.py)"""
    log = Logger.get()
    if (comm := data.comm.comm_world) is not None:
        comm.barrier()
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


def next_fast_fft_size(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


def interpolate_psd(
    freq: npt.NDArray, psd: npt.NDArray, fft_size: int, rate: float = 1.0
) -> npt.NDArray:
    """Perform a logarithmic interpolation of PSD values"""
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


def estimate_psd(
    noise: npt.NDArray[lib.SIGNAL_TYPE],
    block_sizes: npt.NDArray[lib.INDEX_TYPE],
    fft_size: int,
    rate: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Estimate the PSD for each block of a noise timestream using Welch's method"""

    def func(tod):
        # average the periodogram estimate over 10 minute segments
        nperseg = int(600 * rate)
        f, Pxx = welch(tod, fs=rate, nperseg=nperseg)
        # fit and compute full size PSD from fitted parameters
        params = fit_psd_model(f, Pxx)
        freq = np.fft.rfftfreq(fft_size, 1 / rate)
        psd = _model(freq, *params)
        # zero out the DC component
        psd[0] = 0
        return psd

    psd = np.empty((len(block_sizes), fft_size))
    success = np.ones_like(block_sizes, dtype=bool)
    acc = 0
    for i, block_size in enumerate(block_sizes):
        try:
            psd[i] = func(noise[acc : acc + block_size])
        except UnreliableFit:
            success[i] = False
        except Exception:
            raise
        acc += block_size
    return psd, success


class UnreliableFit(Exception):
    """Raised when the covariance matrix of the fit is poorly conditioned"""


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


def psd_to_invntt(psd: npt.NDArray, correlation_length: int) -> npt.NDArray[lib.INVTT_TYPE]:
    """Compute the inverse autocorrelation function from PSD values.
    The result is apodized and cut at the specified correlation length.
    """
    invntt = np.asarray(np.fft.irfft(1 / psd), dtype=lib.INVTT_TYPE)[..., :correlation_length]
    return apodize(invntt)


def psd_to_ntt(psd: npt.NDArray, correlation_length: int) -> npt.NDArray[lib.INVTT_TYPE]:
    """Compute the autocorrelation function from PSD values.
    The result is apodized and cut at the specified correlation length.
    """
    ntt = np.asarray(np.fft.irfft(psd), dtype=lib.INVTT_TYPE)[..., :correlation_length]
    return apodize(ntt)


def apodize(a):
    window = apodization_window(a.shape[-1])
    return a * window


def apodization_window(size: int, kind: str = 'chebwin') -> npt.NDArray:
    if kind == 'gaussian':
        q_apo = 3  # apodization factor: cut happens at q_apo * sigma in the Gaussian window
        window_type = ('general_gaussian', 1, 1 / q_apo * size)
    elif kind == 'chebwin':
        at = 150  # attenuation level (dB)
        window_type = ('chebwin', at)
    else:
        raise RuntimeError(f'Apodization window {kind!r} is not supported.')

    window = np.array(get_window(window_type, 2 * size))
    window = np.fft.ifftshift(window)[:size]
    return window


def folded_psd(inv_n_tt: npt.NDArray[lib.INVTT_TYPE], fft_size: int) -> npt.NDArray:
    """Returns the folded Power Spectral Density of a one-dimensional vector.

    Args:
        inv_n_tt: The inverse autocorrelation function of the vector.
        fft_size: The size of the FFT to use (at least twice the size of ``inv_n_tt``).
    """
    kernel = _get_kernel(inv_n_tt, fft_size)
    psd = 1 / np.abs(np.fft.rfft(kernel, n=fft_size))
    # zero out DC value
    psd[0] = 0
    return psd


def _get_kernel(n_tt: npt.NDArray[lib.INVTT_TYPE], size: int) -> npt.NDArray[lib.INVTT_TYPE]:
    lagmax = n_tt.size - 1
    padding_size = size - (2 * lagmax + 1)
    if padding_size < 0:
        msg = f'The maximum lag ({lagmax}) is too large for the required kernel size ({size}).'
        raise ValueError(msg)
    kernel = np.concatenate((n_tt, np.zeros(padding_size), n_tt[-1:0:-1]))
    return kernel


def effective_ntt(
    invntt: npt.NDArray[lib.INVTT_TYPE], fft_size: int
) -> npt.NDArray[lib.INVTT_TYPE]:
    func = np.vectorize(folded_psd, signature='(m),()->(n)')
    effective_psd = func(invntt, fft_size)
    lagmax = invntt.shape[-1]
    ntt_eff = psd_to_ntt(effective_psd, lagmax)
    return ntt_eff
