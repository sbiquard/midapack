import ctypes as ct
import ctypes.util as ctu

import numpy as np
import numpy.ctypeslib as npc
from mpi4py import MPI

from .types import INDEX_TYPE, INVTT_TYPE, META_ID_TYPE, SIGNAL_TYPE, WEIGHT_TYPE

__all__ = [
    'MLmap',
    'remove_baseline',
    'sim_constrained_block',
    'sim_noise_tod',
]

_mappraiser = None
try:
    _mappraiser = ct.CDLL('libmappraiser.so')
except OSError as e:
    path = ctu.find_library('mappraiser')
    if path is None:
        # Mappraiser was not found in the system
        msg = 'Mappraiser library not found'
        raise ImportError(msg) from e
    _mappraiser = ct.CDLL(path)

MPI_Comm = ct.c_int if MPI._sizeof(MPI.Comm) == ct.sizeof(ct.c_int) else ct.c_void_p


############################################################
# MLmap routine
############################################################

_mappraiser.MLmap.restype = None
_mappraiser.MLmap.argtypes = [
    MPI_Comm,  # comm
    ct.c_char_p,  # outpath
    ct.c_char_p,  # ref
    ct.c_int,  # solver
    ct.c_int,  # precond
    ct.c_int,  # Z_2lvl
    ct.c_int,  # pointing_commflag
    ct.c_double,  # tol
    ct.c_int,  # maxIter
    ct.c_int,  # enl_fac
    ct.c_int,  # ortho_alg
    ct.c_int,  # bs_red
    ct.c_int,  # nside
    ct.c_int,  # gap_strategy
    ct.c_int,  # nested_maxiter
    ct.c_bool,  # do_gap_filling
    ct.c_bool,  # mirror
    ct.c_uint64,  # realization
    npc.ndpointer(dtype=INDEX_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # data_size_proc
    ct.c_int,  # nb_blocks_loc
    npc.ndpointer(dtype=INDEX_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # local_blocks_sizes
    ct.c_double,  # sample_rate
    npc.ndpointer(dtype=META_ID_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # detindxs
    npc.ndpointer(dtype=META_ID_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # obsindxs
    npc.ndpointer(dtype=META_ID_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # telescopes
    ct.c_int,  # Nnz
    npc.ndpointer(dtype=INDEX_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # pixels
    npc.ndpointer(dtype=WEIGHT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # pixweights
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # signal
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # noise
    ct.c_int,  # lambda
    npc.ndpointer(dtype=INVTT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # inv_tt
    npc.ndpointer(dtype=INVTT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # tt
    ct.c_double,  # rcond_threshold
]


def MLmap(
    comm,
    params,
    data_size_proc,
    local_blocks_sizes,
    detindxs,
    obsindxs,
    telescopes,
    nnz,
    pixels,
    pixweights,
    signal,
    noise,
    inv_tt,
    tt,
):
    if _mappraiser is None:
        msg = 'No libmappraiser available, cannot reconstruct the map'
        raise RuntimeError(msg)

    outpath = params['output_dir'].encode('ascii')
    ref = params['ref'].encode('ascii')

    comm.Barrier()

    # https://github.com/mpi4py/mpi4py/blob/master/demo/wrap-ctypes/helloworld.py
    comm_c = MPI_Comm(comm.handle)

    _mappraiser.MLmap(
        comm_c,
        outpath,
        ref,
        params['solver'],
        params['precond'],
        params['Z_2lvl'],
        params['ptcomm_flag'],
        params['tol'],
        params['maxiter'],
        params['enl_fac'],
        params['ortho_alg'],
        params['bs_red'],
        params['nside'],
        params['gap_strategy'],
        params['nested_maxiter'],
        params['fill_gaps'],
        params['mirror'],
        params['realization'],
        data_size_proc,
        len(local_blocks_sizes),
        local_blocks_sizes,
        params['fsample'],
        detindxs,
        obsindxs,
        telescopes,
        nnz,
        pixels,
        pixweights,
        signal,
        noise,
        params['lambda'],
        inv_tt,
        tt,
        params['rcond_threshold'],
    )


############################################################
# Timestream generation routine
############################################################

_mappraiser.sim_noise_tod.restype = None
_mappraiser.sim_noise_tod.argtypes = [
    ct.c_int,  # samples
    ct.c_int,  # lambda
    npc.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # tt
    npc.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # buf
    ct.c_uint64,  # realization
    ct.c_uint64,  # detindx
    ct.c_uint64,  # obsindx
    ct.c_uint64,  # telescope
    ct.c_double,  # sample_rate
]


def sim_noise_tod(
    samples,
    lambda_,
    tt,
    buf,
    realization,
    detindx,
    obsindx,
    telescope,
    sample_rate,
):
    if _mappraiser is None:
        msg = 'No libmappraiser available, cannot reconstruct the map'
        raise RuntimeError(msg)

    _mappraiser.sim_noise_tod(
        samples,
        lambda_,
        tt,
        buf,
        realization,
        detindx,
        obsindx,
        telescope,
        sample_rate,
    )


############################################################
# Baseline computation routine
############################################################

_mappraiser.remove_baseline.restype = None
_mappraiser.remove_baseline.argtypes = [
    ct.c_int,  # samples
    npc.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # buf
    npc.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),  # baseline
    npc.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),  # valid
    ct.c_int,  # w0
    ct.c_bool,  # rm
]


def remove_baseline(
    samples,
    buf,
    baseline,
    valid,
    w0,
    rm,
):
    if _mappraiser is None:
        msg = 'No libmappraiser available, cannot reconstruct the map'
        raise RuntimeError(msg)

    _mappraiser.remove_baseline(
        samples,
        buf,
        baseline,
        valid,
        w0,
        rm,
    )


############################################################
# Single block constrained realization
############################################################

_mappraiser.sim_constrained_block.restype = None
_mappraiser.sim_constrained_block.argtypes = [
    ct.c_bool,
    ct.c_bool,
    ct.c_int,  # samples
    ct.c_int,  # lambda
    ct.c_int,  # w0
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # tt
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # inv_tt
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # noise
    npc.ndpointer(dtype=INDEX_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # pix
    ct.c_uint64,  # realization
    ct.c_uint64,  # detindx
    ct.c_uint64,  # obsindx
    ct.c_uint64,  # telescope
    ct.c_double,  # sample_rate
]


def sim_constrained_block(
    init,
    finalize,
    samples,
    lambda_,
    w0,
    tt,
    inv_tt,
    noise,
    pix,
    realization,
    detindx,
    obsindx,
    telescope,
    sample_rate,
):
    if _mappraiser is None:
        msg = 'No libmappraiser available, cannot reconstruct the map'
        raise RuntimeError(msg)

    _mappraiser.sim_constrained_block(
        init,
        finalize,
        samples,
        lambda_,
        w0,
        tt,
        inv_tt,
        noise,
        pix,
        realization,
        detindx,
        obsindx,
        telescope,
        sample_rate,
    )
