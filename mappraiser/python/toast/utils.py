# This script contains a list of routines to set mappraiser parameters, and
# apply the Mappraiser operator during a TOD2MAP TOAST(3) pipeline

import os

import numpy as np
import scipy.signal
from astropy import units as u
from scipy.optimize import curve_fit

from toast.ops.memory_counter import MemoryCounter
from toast.utils import Logger, dtype_to_aligned, memreport


# Here are some helper functions adapted from toast/src/ops/madam_utils.py
def log_time_memory(
    data, timer=None, timer_msg=None, mem_msg=None, full_mem=False, prefix=""
):
    """(This function is taken from madam_utils.py)"""
    log = Logger.get()
    data.comm.comm_world.barrier()
    restart = False

    if timer is not None:
        if timer.is_running():
            timer.stop()
            restart = True

        if data.comm.world_rank == 0:
            msg = f"{prefix} {timer_msg}: {timer.seconds():0.1f} s"
            log.debug(msg)

    if mem_msg is not None:
        # Dump toast memory use
        mem_count = MemoryCounter(silent=True)
        mem_count.total_bytes = 0
        toast_bytes = mem_count.apply(data)

        if data.comm.group_rank == 0:
            msg = "{prefix} {mem_msg} Group {data.comm.group} memory = {toast_bytes / 1024**2:0.2f} GB"
            log.debug(msg)
        if full_mem:
            _ = memreport(
                msg="{} {}".format(prefix, mem_msg), comm=data.comm.comm_world
            )
    if restart and timer is not None:
        timer.start()


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def stage_local(
    data,
    nsamp,
    dets,
    detdata_name,
    mappraiser_buffer,
    interval_starts,
    nnz,
    det_mask,
    shared_flags,
    shared_mask,
    det_flags,
    det_flag_mask,
    do_purge=False,
    operator=None,
    n_repeat=1,
    pair_diff=False,
    pair_skip=False,
    select_qu=False,
):
    """Helper function to fill a mappraiser buffer from a local detdata key.
    (This function is taken from madam_utils.py)
    """
    do_flags = False
    if shared_flags is not None or det_flags is not None:
        do_flags = True
        # Flagging should only be enabled when we are processing the pixel indices
        # (which is how mappraiser effectively implements flagging).  So we will set
        # all flagged samples to "-1" below.
    if pair_diff and pair_skip:
        raise RuntimeError("pair_diff and pair_skip in stage_local are incompatible.")

    for iobs, ob in enumerate(data.obs):
        local_dets = set(ob.select_local_detectors(flagmask=det_mask))
        offset = interval_starts[iobs]
        if pair_diff or pair_skip:
            for idet, pair in enumerate(pairwise(dets)):
                if set(pair).isdisjoint(local_dets):
                    # nothing to do
                    continue
                if not (set(pair).issubset(local_dets)):
                    msg = f"Incomplete {pair=} ({ob.uid=}, {local_dets=}"
                    raise RuntimeError(msg)
                if operator is not None:
                    # Synthesize data for staging
                    obs_data = data.select(obs_uid=ob.uid)
                    operator.apply(obs_data, detectors=list(pair))

                obs_samples = ob.n_local_samples
                flags = None
                if do_flags:
                    # Using flags
                    flags = np.zeros(obs_samples, dtype=np.uint8)
                if shared_flags is not None:
                    flags |= ob.shared["flags"][:] & shared_mask

                slc = slice(
                    (idet * nsamp + offset) * nnz,
                    (idet * nsamp + offset + obs_samples) * nnz,
                    1,
                )
                det_a, det_b = pair
                if detdata_name is not None:
                    if select_qu:
                        mappraiser_buffer[slc] = np.repeat(
                            ob.detdata[detdata_name][det_a][..., 1:].flatten(),
                            n_repeat,
                        )
                    else:
                        mappraiser_buffer[slc] = np.repeat(
                            ob.detdata[detdata_name][det_a].flatten(),
                            n_repeat,
                        )
                    if pair_diff:
                        # We are staging signal or noise
                        # Take the half difference
                        mappraiser_buffer[slc] = 0.5 * (
                            mappraiser_buffer[slc]
                            - np.repeat(
                                ob.detdata[detdata_name][det_b].flatten(),
                                n_repeat,
                            )
                        )
                else:
                    # Noiseless cases (noise_name=None).
                    mappraiser_buffer[slc] = 0.0

                if do_flags:
                    if det_flags is None:
                        detflags = flags
                    else:
                        detflags = np.copy(flags)
                        detflags |= ob.detdata[det_flags][det_a] & det_flag_mask
                        detflags |= ob.detdata[det_flags][det_b] & det_flag_mask
                    # mappraiser's pixels buffer has nnz=3, not nnz=1
                    repeated_flags = np.repeat(detflags, n_repeat)
                    mappraiser_buffer[slc][repeated_flags != 0] = -1
        else:
            for idet, det in enumerate(dets):
                if det not in local_dets:
                    continue
                if operator is not None:
                    # Synthesize data for staging
                    obs_data = data.select(obs_uid=ob.uid)
                    operator.apply(obs_data, detectors=[det])

                obs_samples = ob.n_local_samples

                flags = None
                if do_flags:
                    # Using flags
                    flags = np.zeros(obs_samples, dtype=np.uint8)
                if shared_flags is not None:
                    flags |= ob.shared["flags"][:] & shared_mask

                slc = slice(
                    (idet * nsamp + offset) * nnz,
                    (idet * nsamp + offset + obs_samples) * nnz,
                    1,
                )
                if detdata_name is not None:
                    mappraiser_buffer[slc] = np.repeat(
                        ob.detdata[detdata_name][det].flatten(),
                        n_repeat,
                    )
                else:
                    # Noiseless cases (noise_name=None).
                    mappraiser_buffer[slc] = 0.0

                if do_flags:
                    if det_flags is None:
                        detflags = flags
                    else:
                        detflags = np.copy(flags)
                        detflags |= ob.detdata[det_flags][det] & det_flag_mask
                    # mappraiser's pixels buffer has nnz=3, not nnz=1
                    repeated_flags = np.repeat(detflags, n_repeat)
                    mappraiser_buffer[slc][repeated_flags != 0] = -1
        if do_purge:
            del ob.detdata[detdata_name]
    return


def stage_in_turns(
    data,
    nodecomm,
    n_copy_groups,
    nsamp,
    dets,
    detdata_name,
    mappraiser_dtype,
    interval_starts,
    nnz,
    det_mask,
    shared_flags,
    shared_mask,
    det_flags,
    det_flag_mask,
    operator=None,
    n_repeat=1,
    pair_diff=False,
    pair_skip=False,
    select_qu=False,
):
    """When purging data, take turns staging it.
    (This function is taken from madam_utils.py)
    """
    raw = None
    wrapped = None
    for copying in range(n_copy_groups):
        if nodecomm.rank % n_copy_groups == copying:
            # Our turn to copy data
            storage, _ = dtype_to_aligned(mappraiser_dtype)
            if pair_diff:
                raw = storage.zeros(nsamp * (len(dets) // 2) * nnz)
            else:
                raw = storage.zeros(nsamp * len(dets) * nnz)
            wrapped = raw.array()
            stage_local(
                data,
                nsamp,
                dets,
                detdata_name,
                wrapped,
                interval_starts,
                nnz,
                det_mask,
                shared_flags,
                shared_mask,
                det_flags,
                det_flag_mask,
                do_purge=True,
                operator=operator,
                n_repeat=n_repeat,
                pair_diff=pair_diff,
                pair_skip=pair_skip,
                select_qu=select_qu,
            )
        nodecomm.barrier()
    return raw, wrapped


def apo_window(lambd: int, kind: str = "chebwin") -> np.ndarray:
    if kind == "gaussian":
        # Apodization factor: cut happens at q_apo * sigma in the Gaussian window
        q_apo = 3
        window = ("general_gaussian", 1, 1 / q_apo * lambd)
    elif kind == "chebwin":
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.chebwin.html#scipy.signal.windows.chebwin
        # Attenuation level (dB)
        at = 150
        window = ("chebwin", at)
    else:
        raise RuntimeError(f"Apodisation window '{kind}' is not supported.")

    window = scipy.signal.get_window(window, 2 * lambd)
    window = np.fft.ifftshift(window)[:lambd]
    return window


def compute_autocorrelations(
    ob_uids,
    dets,
    mappraiser_noise,
    local_block_sizes,
    lambda_,
    fsamp,
    buffer_inv_tt,
    buffer_tt,
    invtt_dtype,
    print_info=False,
    save_psd=False,
    save_dir="",
    apod_window_type="chebwin",
):
    """Compute the first lines of the blocks of the banded noise covariance and store them in the provided buffer."""
    offset = 0
    nobs = len(ob_uids)
    for iob, uid in enumerate(ob_uids):
        for idet, det in enumerate(dets):
            blocksize = local_block_sizes[idet * nobs + iob]
            nsetod = mappraiser_noise[offset : offset + blocksize]
            slc = slice(
                (idet * nobs + iob) * lambda_,
                (idet * nobs + iob) * lambda_ + lambda_,
                1,
            )
            buffer_inv_tt[slc], _ = noise_autocorrelation(
                nsetod,
                blocksize,
                lambda_,
                fsamp,
                idet,
                invtt_dtype,
                apod_window_type,
                verbose=(print_info and (idet == 0) and (iob == 0)),
                save_psd=save_psd,
                fname=os.path.join(save_dir, f"noise_fit_{uid}_{det}"),
            )
            buffer_tt[slc] = compute_autocorr(
                1 / compute_psd_eff(buffer_inv_tt[slc], blocksize), lambda_
            )
            offset += blocksize
    return


def psd_model(f, sigma, alpha, fknee, fmin):
    return sigma * (1 + ((f + fmin) / fknee) ** alpha)


def logpsd_model(f, a, alpha, fknee, fmin):
    return a + np.log10(1 + ((f + fmin) / fknee) ** alpha)


def inversepsd_model(f, sigma, alpha, fknee, fmin):
    return sigma * 1.0 / (1 + ((f + fmin) / fknee) ** alpha)


def inverselogpsd_model(f, a, alpha, fknee, fmin):
    return a - np.log10(1 + ((f + fmin) / fknee) ** alpha)


def noise_autocorrelation(
    nsetod,
    nn,
    lambda_,
    fsamp,
    idet,
    invtt_dtype,
    apod_window_type,
    verbose=False,
    save_psd=False,
    fname="",
):
    """Computes a periodogram from a noise timestream, and fits a PSD model
    to it, which is then used to build the first row of a Toeplitz block.
    """
    # remove unit of fsamp to avoid problems when computing periodogram
    try:
        f_unit = fsamp.unit
        fsamp = float(fsamp / (1.0 * f_unit))
    except AttributeError:
        pass

    # Estimate psd from noise timestream

    # Average over 10 minute segments
    nperseg = int(600 * fsamp)

    # Compute a periodogram with Welch's method
    f, psd = scipy.signal.welch(nsetod, fsamp, nperseg=nperseg)

    # Fit the psd model to the periodogram (in log scale)
    popt, pcov = curve_fit(
        logpsd_model,
        f[1:],
        np.log10(psd[1:]),
        p0=np.array([-7, -1.0, 0.1, 1e-6]),
        bounds=([-20, -10, 0.0, 0.0], [0.0, 0.0, 20, 1]),
        maxfev=1000,
    )

    if verbose:
        print(
            "\n[det "
            + str(idet)
            + "]: PSD fit log(sigma2) = %1.2f, alpha = %1.2f, fknee = %1.2f, fmin = %1.2e\n"
            % tuple(popt),
            flush=True,
        )
        print("[det {}]: PSD fit covariance: \n{}\n".format(idet, pcov), flush=True)

    # Initialize full size inverse PSD in frequency domain
    f_full = np.fft.rfftfreq(nn, 1.0 / fsamp)

    # Compute inverse noise psd from fit and extrapolate (if needed) to lowest frequencies
    ipsd_fit = inversepsd_model(f_full, 10 ** (-popt[0]), *popt[1:])
    psd_fit = psd_model(f_full, 10 ** (popt[0]), *popt[1:])
    psd_fit[0] = 0.0

    # Compute inverse noise auto-correlation functions
    inv_tt = np.fft.irfft(ipsd_fit, n=nn)
    tt = np.fft.irfft(psd_fit, n=nn)

    # Define apodization window
    # Only allow max lambda = nn//2
    if lambda_ > nn // 2:
        msg = f"Bandwidth cannot be larger than timestream (lambda={lambda_}, {nn=})."
        raise RuntimeError(msg)

    window = apo_window(lambda_, kind=apod_window_type)

    # Apply window
    inv_tt_w = np.multiply(window, inv_tt[:lambda_], dtype=invtt_dtype)
    tt_w = np.multiply(window, tt[:lambda_], dtype=invtt_dtype)

    # Optionally save some PSDs for plots
    if save_psd:
        noise_fit = {
            "nn": nn,
            "popt": popt,
            "pcov": pcov,
            "freqs": f,
            "periodogram": psd,
        }
        np.savez(fname, **noise_fit)

    return inv_tt_w, tt_w


def compute_psd_eff(tt, fftlen) -> np.ndarray:
    """
    Computes the power spectral density from a given autocorrelation function.

    :param tt: Input autocorrelation
    :param m: FFT size

    :return: The PSD (size = fft_size // 2 + 1 because we are using np.fft.rfft)
    """
    # Form the cylic autocorrelation
    lag = len(tt)
    circ_t = np.pad(tt, (0, fftlen - lag), "constant")
    if lag > 1:
        circ_t[-lag + 1 :] = np.flip(tt[1:], 0)

    # FFT
    psd = np.abs(np.fft.rfft(circ_t, n=fftlen))
    return psd


def compute_autocorr(psd, lambda_: int, apo=True) -> np.ndarray:
    """
    Computes the autocorrelation function from a given power spectral density.

    :param psd: Input PSD
    :param lambd: Assumed noise correlation length
    :param apo: if True, apodize the autocorrelation function

    :return: The autocorrelation function apodised and cut after `lambda_` terms
    """

    # Compute the inverse FFT
    autocorr = np.fft.irfft(psd)[:lambda_]

    # Apodisation
    if apo:
        window = apo_window(lambda_)
        autocorr *= window

    return autocorr
