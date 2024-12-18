/**
 * @file mappraiser.c
 * @brief Process pointing, signal and noise data arrays to produce maps in FITS
 * format
 * @authors Hamza El Bouhargani
 * @date May 2019
 * @update June 2020 by Aygul Jamal
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mappraiser/create_toeplitz.h>
#include <mappraiser/gap_filling.h>
#include <mappraiser/map.h>
#include <mappraiser/mapping.h>
#include <mappraiser/outputs.h>
#include <mappraiser/pcg_true.h>
#include <mappraiser/precond.h>
#include <mappraiser/weight.h>
#include <memutils.h>

#ifdef WITH_ECG
#include <mappraiser/ecg.h>
#endif

void MLmap(MPI_Comm comm, char *outpath, char *ref, int solver, int precond,
           int Z_2lvl, int pointing_commflag, double tol, int maxiter,
           int enl_fac, int ortho_alg, int bs_red, int nside, int gap_stgy,
           bool do_gap_filling, uint64_t realization, int *data_size_proc,
           int nb_blocks_loc, int *local_blocks_sizes, double sample_rate,
           uint64_t *detindxs, uint64_t *obsindxs, uint64_t *telescopes,
           int nnz, int *pix, double *pixweights, double *signal, double *noise,
           int lambda, double *inv_tt, double *tt) {
    int64_t M;        // global number of rows of the pointing matrix
    int64_t gif;      // global index for the first local line
    int m;            // local number of rows of the pointing matrix
    int n_blocks_tot; // global number of stationary intervals (data blocks)

    Mat A;                    // pointing matrix structure
    int nbr_valid_pixels;     // nbr of valid pixel indices
    int nbr_extra_pixels;     // nbr of extra pixel indices
    Gap Gaps;                 // timestream gaps structure
    double *x, *rcond = NULL; // pixel domain vectors
    int *lhits = NULL;

    int rank, world_size;
    MPI_Status status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (rank == 0) {
        printf("\n############# MAPPRAISER : MidAPack PaRAllel Iterative Sky "
               "EstimatoR vDev, May 2019 "
               "################\n");
        printf("Last compiled on %s at %s\n", __DATE__, __TIME__);
        printf("[MPI info] rank = %d, size = %d\n", rank, world_size);
        puts("##### Initialization ####################");
        fflush(stdout);
    }

    // total length of the time domain signal
    M = 0;
    for (int i = 0; i < world_size; i++) {
        M += data_size_proc[i];
    }

    // compute distribution indexes over the processes
    m = data_size_proc[rank];
    gif = 0;
    for (int i = 0; i < rank; i++) {
        gif += data_size_proc[i];
    }

    // Print information on data distribution
    MPI_Allreduce(&nb_blocks_loc, &n_blocks_tot, 1, MPI_INT, MPI_SUM, comm);
    if (rank == 0) {
        printf("[Data] global M = %ld (%d intervals)\n", M, n_blocks_tot);
        printf("[Data] local  m = %d (%d intervals)\n", m, nb_blocks_loc);
        fflush(stdout);
    }

    GapStrategy gs = gap_stgy;

    // Set flag to ignore extra pixels when not marginalizing
    A.flag_ignore_extra = !(gs == MARG_LOCAL_SCAN || gs == MARG_PROC);

    // Create extra pixels according to the chosen strategy
    create_extra_pix(pix, pixweights, nnz, nb_blocks_loc, local_blocks_sizes,
                     gs);

    if (rank == 0) {
        printf("[Gaps] strategy: ");
        print_gap_stgy(gs);
        fflush(stdout);
    }

    // ____________________________________________________________
    // Pointing matrix initialization + mapping

    MPI_Barrier(comm);
    double st = MPI_Wtime();

    MatInit(&A, m, nnz, pix, pixweights, pointing_commflag, comm);
    Gaps.ngap = build_pixel_to_time_domain_mapping(&A);

    MPI_Barrier(comm);
    double elapsed = MPI_Wtime() - st;

    nbr_extra_pixels = A.trash_pix * A.nnz;
    nbr_valid_pixels = A.lcount - nbr_extra_pixels;

    if (rank == 0) {
        printf("Initialized pointing matrix in %lf s\n", elapsed);
        printf("[proc %d] sky pixels = %d", rank, A.lcount / A.nnz);
        printf(" (%d valid + %d extra)\n", nbr_valid_pixels / A.nnz,
               nbr_extra_pixels / A.nnz);
        printf("[proc %d] local timestream gaps = %d\n", rank, Gaps.ngap);
        fflush(stdout);
    }

    // ____________________________________________________________
    // Map objects memory allocation

    // Size of map that will be estimated by the solver
    int solver_map_size = get_actual_map_size(&A);

    rcond = SAFEMALLOC(sizeof *rcond * solver_map_size / A.nnz);
    lhits = SAFEMALLOC(sizeof *lhits * solver_map_size / A.nnz);

    // ____________________________________________________________
    // Create piecewise Toeplitz matrix

    // specifics parameters:
    int nb_blocks_tot = n_blocks_tot;
    int lambda_block_avg = lambda;

    // flags for Toeplitz product strategy
    Flag flag_stgy;
    flag_stgy_init_auto(&flag_stgy);

    // skip build gappy blocks
    flag_stgy.flag_skip_build_gappy_blocks = 1;

    // to print something on screen
    // flag_stgy.flag_verbose = 1;

    // define Toeplitz blocks list and structure for Nm1
    Block *tpltzblocks;
    Tpltz Nm1;

    // dependants parameters:
    int64_t nrow = M;
    int mcol = 1;

    int64_t id0 = gif;
    int local_V_size = m;

    // Block definition
    tpltzblocks = SAFEMALLOC(sizeof *tpltzblocks * nb_blocks_loc);
    defineBlocks_avg(tpltzblocks, inv_tt, nb_blocks_loc, local_blocks_sizes,
                     lambda_block_avg, id0);
    defineTpltz_avg(&Nm1, nrow, 1, mcol, tpltzblocks, nb_blocks_loc,
                    nb_blocks_tot, id0, local_V_size, flag_stgy, comm);

    // define the noise covariance matrix
    Tpltz N;
    Block *tpltzblocks_N = SAFEMALLOC(sizeof *tpltzblocks_N * nb_blocks_loc);
    defineBlocks_avg(tpltzblocks_N, tt, nb_blocks_loc, local_blocks_sizes,
                     lambda_block_avg, id0);
    defineTpltz_avg(&N, nrow, 1, mcol, tpltzblocks_N, nb_blocks_loc,
                    nb_blocks_tot, id0, local_V_size, flag_stgy, comm);

    // print Toeplitz parameters for information
    if (rank == 0) {
        printf("Noise model: Banded block Toeplitz");
        printf(" (half bandwidth = %d)\n", lambda_block_avg);
        fflush(stdout);
    }

    // ____________________________________________________________
    // Compute the system preconditioner

    MPI_Barrier(comm);
    if (rank == 0) {
        puts("##### Preconditioner ####################");
        fflush(stdout);
    }

    st = MPI_Wtime();

    // first build the BJ preconditioner
    Precond *P = newPrecondBJ(&A, &Nm1, rcond, lhits, gs, &Gaps, gif,
                              local_blocks_sizes);

    // Allocate memory for the map with the right number of pixels
    x = SAFECALLOC(P->n, sizeof *x);

    MPI_Barrier(comm);
    elapsed = MPI_Wtime() - st;

    if (rank == 0) {
        printf("Block Jacobi preconditioner built for %d sky pixels (%d valid "
               "+ %d extra)\n",
               P->n / A.nnz, P->n_valid / A.nnz, P->n_extra / A.nnz);
        printf("Total time = %lf s\n", elapsed);
        fflush(stdout);
    }

    // Guard against using ECG with extra pixels to estimate
    if (P->n_extra > 0 && solver == 1) {
        if (rank == 0) {
            fprintf(stderr,
                    "ECG solver does not support solving for extra pixels. "
                    "Choose another gap strategy, or use solver=0 (PCG).\n");
        }
        exit(EXIT_FAILURE);
    }

    // Gap treatment can happen now

    MPI_Barrier(comm);
    if (rank == 0) {
        puts("##### Gap treatment ####################");
        fflush(stdout);
    }

    WeightStgy ws = createFromGapStrategy(
        &Gaps, &A, &Nm1, &N, gs, signal, noise, do_gap_filling, realization,
        detindxs, obsindxs, telescopes, sample_rate);

    // final weighting operator
    WeightMatrix W = createWeightMatrix(&Nm1, &N, &Gaps, ws);

    // ____________________________________________________________
    // Now build the 2lvl part of the preconditioner if needed

    if (precond != BJ) {
        MPI_Barrier(comm);
        if (rank == 0) {
            puts("##### 2lvl preconditioner ####################");
            fflush(stdout);
        }

        st = MPI_Wtime();

        P->ptype = precond;
        P->Zn = Z_2lvl == 0 ? world_size : Z_2lvl;
        buildPrecond2lvl(P, &A, &W, x, signal);

        MPI_Barrier(comm);
        elapsed = MPI_Wtime() - st;

        if (rank == 0) {
            printf("2lvl preconditioner construction took %lf s\n", elapsed);
            fflush(stdout);
        }
    }

    // ____________________________________________________________
    // Solve the system

    MPI_Barrier(comm);
    if (rank == 0) {
        puts("##### Main solver ####################");
        fflush(stdout);
    }

    if (solver == 0) {
        // set up SolverInfo structure
        SolverInfo si;
        solverinfo_set_defaults(&si);
        si.store_hist = true;
        si.print = rank == 0;
        si.rel_res_reduct = tol;
        si.max_steps = maxiter;
        si.use_exact_residual = true;

        // solve the equation
        PCG_maxL(&A, P, &W, x, signal, &si);

        // Write PCG residuals to disk
        if (rank == 0) {
            char fname[FILENAME_MAX];
            sprintf(fname, "%s/residuals_%s.dat", outpath, ref);
            int info = solverinfo_write(&si, fname);
            if (info != 0) {
                fputs("Problem writing residuals to file", stderr);
            }
        }

        // Free SolverInfo structure
        solverinfo_free(&si);

    } else if (solver == 1) {
#ifdef WITH_ECG
        ECG_GLS(outpath, ref, &A, &Nm1, &(P->BJ_inv), P->pixpond, x, signal,
                noise, tol, maxiter, enl_fac, ortho_alg, bs_red);
#else
        if (rank == 0)
            fprintf(stderr,
                    "The choice of solver is 1 (=ECG), but the ECG source "
                    "file has not been compiled.\n");
        exit(EXIT_FAILURE);
#endif

    } else {
        if (rank == 0) {
            char msg[] =
                "Incorrect solver parameter. Reminder: solver=0 -> PCG, "
                "solver=1 -> ECG.";
            fputs(msg, stderr);
        }
        exit(EXIT_FAILURE);
    }

    // free tpltz blocks
    FREE(tpltzblocks);
    FREE(tpltzblocks_N);

    // free Gap structure
    FREE(Gaps.id0gap);
    FREE(Gaps.lgap);

    // ____________________________________________________________
    // Write output to fits files

    MPI_Barrier(comm);
    if (rank == 0) {
        puts("##### Write products ####################");
        fflush(stdout);
    }

    st = MPI_Wtime();

    // throw away estimated extra pixels if there are any

    int map_size = get_valid_map_size(&A);
    int extra = get_actual_map_size(&A) - map_size;

    // just copy the preconditioner values and then free it
    double *bj_map = SAFEMALLOC(sizeof *bj_map * map_size * nnz);
    memcpy(bj_map, P->BJ_inv.values + extra * nnz,
           sizeof *bj_map * map_size * nnz);

    PrecondFree(P);

    if (extra > 0) {
#ifdef DEBUG
        double *extra_map = SAFEMALLOC(sizeof *extra_map * extra);
        memcpy(extra_map, x, extra * sizeof(double));
        if (rank == 0) {
            printf("extra map with %d pixels (T only)\n {", extra / nnz);
            for (int j = 0; j < extra; j += nnz) {
                printf(" %e", extra_map[j]);
            }
            puts(" }");
        }
        fflush(stdout);
        FREE(extra_map);
#endif
        // valid map
        memmove(x, x + extra, sizeof *x * map_size);
        memmove(lhits, lhits + extra / nnz, sizeof *lhits * map_size / nnz);
        memmove(rcond, rcond + extra / nnz, sizeof *rcond * map_size / nnz);
        x = SAFEREALLOC(x, sizeof *x * map_size);
        lhits = SAFEREALLOC(lhits, sizeof *lhits * map_size / nnz);
        rcond = SAFEREALLOC(rcond, sizeof *rcond * map_size / nnz);
    }

    // get maps from all processes and combine them

    int *lstid = SAFEMALLOC(sizeof *lstid * map_size);
    for (int i = 0; i < map_size; i++) {
        lstid[i] = A.lindices[i + nnz * A.trash_pix];
    }

    if (rank != 0) {
        MPI_Send(&map_size, 1, MPI_INT, 0, 0, comm);
        MPI_Send(lstid, map_size, MPI_INT, 0, 1, comm);
        MPI_Send(x, map_size, MPI_DOUBLE, 0, 2, comm);
        MPI_Send(rcond, map_size / nnz, MPI_DOUBLE, 0, 3, comm);
        MPI_Send(lhits, map_size / nnz, MPI_INT, 0, 4, comm);
        MPI_Send(bj_map, map_size * nnz, MPI_DOUBLE, 0, 5, comm);
    }

    if (rank == 0) {
        int oldsize;

        MappraiserOutputs outputs;
        initMappraiserOutputs(&outputs, nside, nnz, outpath, ref);

        for (int proc = 0; proc < world_size; proc++) {
            if (proc != 0) {
                // Receive data from other processes
                oldsize = map_size;
                MPI_Recv(&map_size, 1, MPI_INT, proc, 0, comm, &status);
                if (oldsize != map_size) {
                    lstid = SAFEREALLOC(lstid, sizeof *lstid * map_size);
                    x = SAFEREALLOC(x, sizeof *x * map_size);
                    rcond = SAFEREALLOC(rcond, sizeof *rcond * map_size / nnz);
                    lhits = SAFEREALLOC(lhits, sizeof *lhits * map_size / nnz);
                    bj_map =
                        SAFEREALLOC(bj_map, sizeof *bj_map * map_size * nnz);
                }
                MPI_Recv(lstid, map_size, MPI_INT, proc, 1, comm, &status);
                MPI_Recv(x, map_size, MPI_DOUBLE, proc, 2, comm, &status);
                MPI_Recv(rcond, map_size / nnz, MPI_DOUBLE, proc, 3, comm,
                         &status);
                MPI_Recv(lhits, map_size / nnz, MPI_INT, proc, 4, comm,
                         &status);
                MPI_Recv(bj_map, map_size * nnz, MPI_DOUBLE, proc, 5, comm,
                         &status);
            }
            populateMappraiserOutputs(&outputs, x, lstid, rcond, lhits, bj_map,
                                      map_size, nnz);
        }

        // Remove old files before writing new ones
        puts("Checking output directory... old files will be overwritten");
        int remove_info = clearFiles(&outputs);

        if (remove_info == 0) {
            printf("Writing HEALPix maps FITS files to %s...\n", outpath);
            writeFiles(&outputs);
        } else {
            fprintf(stderr, "IO Error: Could not overwrite old files, map "
                            "results will not be stored ;(\n");
        }

        freeMappraiserOutputs(&outputs);
    }

    elapsed = MPI_Wtime() - st;
    if (rank == 0) {
        printf("Total time = %lf s\n", elapsed);
        fflush(stdout);
    }

    // free memory
    FREE(x);
    FREE(rcond);
    FREE(lhits);

    MatFree(&A);
    A.indices = NULL;
    A.values = NULL;
    FREE(lstid);
}
