//
// Created by sbiquard on 5/17/23.
//

#ifndef MAPPRAISER_MAP_H
#define MAPPRAISER_MAP_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

void MLmap(MPI_Comm comm, char *outpath, char *ref, int solver, int precond,
           int Z_2lvl, int pointing_commflag, double tol, int maxiter,
           int enl_fac, int ortho_alg, int bs_red, int nside, int gap_stgy,
           bool do_gap_filling, uint64_t realization, int *data_size_proc,
           int nb_blocks_loc, int *local_blocks_sizes, double sample_rate,
           uint64_t *detindxs, uint64_t *obsindxs, uint64_t *telescopes,
           int Nnz, int *pix, double *pixweights, double *signal, double *noise,
           int lambda, double *inv_tt, double *tt);

#ifdef __cplusplus
}
#endif

#endif // MAPPRAISER_MAP_H
