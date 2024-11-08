import ctypes as ct
import ctypes.util as ctu

import numpy as np
import numpy.ctypeslib as npc
from mpi4py import MPI

from .types import INVTT_TYPE, PIXEL_TYPE, SIGNAL_TYPE, WEIGHT_TYPE

__all__ = [
    'MLmap',
    'available',
]


_mappraiser = None
try:
    _mappraiser = ct.CDLL('libmappraiser.so')
except OSError:
    path = ctu.find_library('mappraiser')
    if path is not None:
        _mappraiser = ct.CDLL(path)

available = _mappraiser is not None

try:
    if MPI._sizeof(MPI.Comm) == ct.sizeof(ct.c_int):
        MPI_Comm = ct.c_int
    else:
        MPI_Comm = ct.c_void_p
except Exception:
    print('Failed to set the portable MPI comunicator datatype')
    raise


def _encode_comm(comm):
    comm_ptr = MPI._addressof(comm)
    return MPI_Comm.from_address(comm_ptr)


############################################################
# MLmap routine
############################################################

_mappraiser.MLmap.restype = None  # pyright: ignore
_mappraiser.MLmap.argtypes = [  # pyright: ignore
    MPI_Comm,  # comm
    ct.c_char_p,  # outpath
    ct.c_char_p,  # ref
    ct.c_int,  # solver
    ct.c_int,  # precond
    ct.c_int,  # Z_2lvl
    ct.c_int,  # pointing_commflag
    ct.c_double,  # tol
    ct.c_int,  # maxIter
    ct.c_int,  # enlFac
    ct.c_int,  # ortho_alg
    ct.c_int,  # bs_red
    ct.c_int,  # nside
    ct.c_int,  # gap_stgy
    ct.c_bool,  # do_gap_filling
    ct.c_uint64,  # realization
    npc.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # data_size_proc
    ct.c_int,  # nb_blocks_loc
    npc.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # local_blocks_sizes
    ct.c_double,  # sample_rate
    npc.ndpointer(dtype=np.uint64, ndim=1, flags='C_CONTIGUOUS'),  # detindxs
    npc.ndpointer(dtype=np.uint64, ndim=1, flags='C_CONTIGUOUS'),  # obsindxs
    npc.ndpointer(dtype=np.uint64, ndim=1, flags='C_CONTIGUOUS'),  # telescopes
    ct.c_int,  # Nnz
    npc.ndpointer(dtype=PIXEL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # pixels
    npc.ndpointer(dtype=WEIGHT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # pixweights
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # signal
    npc.ndpointer(dtype=SIGNAL_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # noise
    ct.c_int,  # lambda
    npc.ndpointer(dtype=INVTT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # inv_tt
    npc.ndpointer(dtype=INVTT_TYPE, ndim=1, flags='C_CONTIGUOUS'),  # tt
]


def MLmap(
    comm,
    params,
    data_size_proc,
    nb_blocks_loc,
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
    if not available:
        raise RuntimeError('No libmappraiser available, cannot reconstruct the map')

    outpath = params['output_dir'].encode('ascii')
    ref = params['ref'].encode('ascii')

    comm.Barrier()

    _mappraiser.MLmap(  # pyright: ignore
        _encode_comm(comm),
        outpath,
        ref,
        params['solver'],
        params['precond'],
        params['Z_2lvl'],
        params['ptcomm_flag'],
        params['tol'],
        params['maxiter'],
        params['enlFac'],
        params['ortho_alg'],
        params['bs_red'],
        params['nside'],
        params['gap_stgy'],
        params['fill_gaps'],
        params['realization'],
        data_size_proc,
        nb_blocks_loc,
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
    )

    return
