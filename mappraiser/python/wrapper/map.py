import ctypes as ct
import ctypes.util as ctu

import numpy.ctypeslib as npc
from mpi4py import MPI

from .types import INDEX_TYPE, INVTT_TYPE, META_ID_TYPE, SIGNAL_TYPE, WEIGHT_TYPE

__all__ = [
    'MLmap',
]

_mappraiser = None
try:
    _mappraiser = ct.CDLL('libmappraiser.so')
except OSError:
    path = ctu.find_library('mappraiser')
    if path is None:
        # Mappraiser was not found in the system
        raise ImportError('Mappraiser library not found')
    _mappraiser = ct.CDLL(path)

if MPI._sizeof(MPI.Comm) == ct.sizeof(ct.c_int):
    MPI_Comm = ct.c_int
else:
    MPI_Comm = ct.c_void_p


############################################################
# MLmap routine
############################################################

_mappraiser.MLmap.restype = None  # pyright: ignore[reportOptionalMemberAccess]
_mappraiser.MLmap.argtypes = [  # pyright: ignore[reportOptionalMemberAccess]
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
        raise RuntimeError('No libmappraiser available, cannot reconstruct the map')

    outpath = params['output_dir'].encode('ascii')
    ref = params['ref'].encode('ascii')

    comm.Barrier()

    # https://github.com/mpi4py/mpi4py/blob/master/demo/wrap-ctypes/helloworld.py
    comm_c = MPI_Comm(comm.handle)

    _mappraiser.MLmap(  # pyright: ignore[reportOptionalMemberAccess]
        comm_c,
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
    )

    return
