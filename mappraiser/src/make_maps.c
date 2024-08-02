/**
 * @file make_maps.c
 * @brief Master function to perform the map-making process
 * @authors Hamza El Bouhargani, Aygul Jamal, Simon Biquard
 * @date August 2024
 */

#include <fitsio.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mappraiser/create_toeplitz.h>
#include <mappraiser/gap_filling.h>
#include <mappraiser/iofiles.h>
#include <mappraiser/make_maps.h>
#include <mappraiser/mapping.h>
#include <mappraiser/pcg_true.h>
#include <mappraiser/precond.h>
#include <mappraiser/weight.h>
#include <midapack/memutils.h>

#ifdef WITH_ECG
#include <mappraiser/ecg.h>
#endif

#ifdef HAVE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

WeightStgy handle_gaps(Gap *Gaps, Mat *A, Tpltz *Nm1, Tpltz *N, GapStrategy gs,
                       double *b, const double *noise, bool do_gap_filling,
                       uint64_t realization, const uint64_t *detindxs,
                       const uint64_t *obsindxs, const uint64_t *telescopes,
                       double sample_rate);

void x2map_pol(double *mapI, double *mapQ, double *mapU, double *Cond,
               int *hits, const double *x, const int *lstid, const double *cond,
               const int *lhits, int xsize, int nnz);

void init_templates(int store_hwp, TemplateClass *X, double *B, int **az_binned,
                    int *hwp_bins, double ***hwp_mod, int *ces_length,
                    Tpltz *Nm1, int npoly, int nhwp, double delta_t, int ground,
                    int n_sss_bins, int **sweeptstamps, int *nsweeps,
                    double **az, double *az_min, double *az_max,
                    double **hwp_angle, int nces, int nb_blocks_loc,
                    void *local_blocks_sizes, double sample_rate, int rank);

void make_maps(MPI_Comm comm, int method, char *outpath, char *ref, int solver,
               int precond, int Z_2lvl, int pointing_commflag, double tol,
               int maxiter, int enlFac, int ortho_alg, int bs_red, int nside,
               int gap_stgy, bool do_gap_filling, uint64_t realization,
               void *data_size_proc, int nb_blocks_loc,
               void *local_blocks_sizes, double sample_rate, uint64_t *detindxs,
               uint64_t *obsindxs, uint64_t *telescopes, int Nnz, void *pix,
               void *pixweights, void *signal, double *noise, int lambda,
               double *inv_tt, double *tt, int npoly, int nhwp, double delta_t,
               int ground, int n_sss_bins, int **sweeptstamps, int *nsweeps,
               double **az, double *az_min, double *az_max, double **hwp_angle,
               int nces) {
    int64_t M;          // Global number of rows
    int m;              // local number of rows of the pointing matrix A
    int Nb_t_Intervals; // nbr of stationary intervals
    int64_t gif;        // global indice for the first local line

    Mat A;                // pointing matrix structure
    int nbr_valid_pixels; // nbr of valid pixel indices
    int nbr_extra_pixels; // nbr of extra pixel indices
    Gap Gaps;             // timestream gaps structure

    double *x, *cond = NULL; // pixel domain vectors
    int *lhits = NULL;
    int rank, size;
    MPI_Status status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("\n############# MAPPRAISER : MidAPack PaRAllel Iterative Sky "
               "EstimatoR v2.1, July 2024 "
               "################\n");
        printf("Last compiled on %s at %s\n", __DATE__, __TIME__);
        printf("[MPI info] rank = %d, size = %d\n", rank, size);
        puts("##### Initialization ####################");
        fflush(stdout);
    }

    // total length of the time domain signal
    M = 0;
    for (int i = 0; i < size; i++) {
        M += ((int *)data_size_proc)[i];
    }

    // compute distribution indexes over the processes
    m = ((int *)data_size_proc)[rank];
    gif = 0;
    for (int i = 0; i < rank; i++) {
        gif += ((int *)data_size_proc)[i];
    }

    // Print information on data distribution
    int Nb_t_Intervals_loc = nb_blocks_loc;
    MPI_Allreduce(&nb_blocks_loc, &Nb_t_Intervals, 1, MPI_INT, MPI_SUM, comm);
    if (rank == 0) {
        printf("[Data] global M = %ld (%d intervals)\n", M, Nb_t_Intervals);
        printf("[Data] local  m = %d (%d intervals)\n", m, Nb_t_Intervals_loc);
        fflush(stdout);
    }

    GapStrategy gs = gap_stgy;

    // Set flag to ignore extra pixels when not marginalizing
    A.flag_ignore_extra = !(gs == MARG_LOCAL_SCAN || gs == MARG_PROC);

    // Create extra pixels according to the chosen strategy
    create_extra_pix(pix, pixweights, Nnz, nb_blocks_loc, local_blocks_sizes,
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

    MatInit(&A, m, Nnz, pix, pixweights, pointing_commflag, comm);
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

    cond = SAFEMALLOC(sizeof *cond * solver_map_size / A.nnz);
    lhits = SAFEMALLOC(sizeof *lhits * solver_map_size / A.nnz);

    // ____________________________________________________________
    // Create piecewise Toeplitz matrix

    // specifics parameters:
    int nb_blocks_tot = Nb_t_Intervals;
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
    Precond *P =
        newPrecondBJ(&A, &Nm1, cond, lhits, gs, &Gaps, gif, local_blocks_sizes);

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

    WeightStgy ws =
        handle_gaps(&Gaps, &A, &Nm1, &N, gs, signal, noise, do_gap_filling,
                    realization, detindxs, obsindxs, telescopes, sample_rate);

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
        P->Zn = Z_2lvl == 0 ? size : Z_2lvl;
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

    switch (method) {
    case 0:

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
                    noise, tol, maxiter, enlFac, ortho_alg, bs_red);
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

        break;

    case 1:
        // ____________________________________________________________
        // Templates initialization

        MPI_Barrier(comm);
        if (rank == 0) {
            puts("##### Templates ####################");
            fflush(stdout);
        }

        st = MPI_Wtime();

        // Initialize templates
        int store_hwp = 0;
        TemplateClass *X = NULL;
        double *B = NULL;
        int **az_binned = NULL;
        int *hwp_bins = NULL;
        double ***hwp_mod = NULL;
        int *ces_length = NULL;
        init_templates(store_hwp, X, B, az_binned, hwp_bins, hwp_mod,
                       ces_length, &Nm1, npoly, nhwp, delta_t, ground,
                       n_sss_bins, sweeptstamps, nsweeps, az, az_min, az_max,
                       hwp_angle, nces, nb_blocks_loc, local_blocks_sizes,
                       sample_rate, rank);

        MPI_Barrier(comm);
        elapsed = MPI_Wtime() - st;

        if (rank == 0) {
            printf("Total time building Templates classes and inverse "
                   "kernel blocks = %lf s\n",
                   elapsed);
            fflush(stdout);
        }

        MPI_Barrier(comm);
        if (rank == 0) {
            puts("##### Start PCG ####################");
            fflush(stdout);
        }
        st = MPI_Wtime();

        // Conjugate Gradient
        if (solver == 0)
            PCG_GLS_templates(outpath, ref, &A, P, &W, X, B, sweeptstamps,
                              npoly, ground, nhwp, nsweeps, az_binned,
                              n_sss_bins, hwp_bins, hwp_mod, delta_t, store_hwp,
                              nces, ces_length, nb_blocks_loc, x, signal, noise,
                              tol, maxiter, sample_rate);
        else {
            fprintf(
                stderr,
                "ECG unavailable at this stage please choose the PCG solver: "
                "solver=0\n");
            exit(1);
        }

        MPI_Barrier(comm);
        elapsed = MPI_Wtime() - st;

        if (rank == 0) {
            puts("##### End PCG ####################");
            printf("Total PCG time = %lf s\n", elapsed);
        }
        fflush(stdout);

        break;
    } // end switch(method)

    // free tpltz blocks
    FREE(tpltzblocks);
    FREE(tpltzblocks_N);

    // free Gap structure
    FREE(Gaps.id0gap);
    FREE(Gaps.lgap);

    // free memory allocated for preconditioner
    PrecondFree(P);

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

    if (extra > 0) {
#ifdef DEBUG
        double *extra_map = SAFEMALLOC(sizeof *extra_map * extra);
        memcpy(extra_map, x, extra * sizeof(double));
        if (rank == 0) {
            printf("extra map with %d pixels (T only)\n {", extra / Nnz);
            for (int j = 0; j < extra; j += Nnz) {
                printf(" %e", extra_map[j]);
            }
            puts(" }");
        }
        fflush(stdout);
        FREE(extra_map);
#endif
        // valid map
        memmove(x, x + extra, sizeof *x * map_size);
        memmove(lhits, lhits + extra / Nnz, sizeof *lhits * map_size / Nnz);
        memmove(cond, cond + extra / Nnz, sizeof *cond * map_size / Nnz);
        x = SAFEREALLOC(x, sizeof *x * map_size);
        lhits = SAFEREALLOC(lhits, sizeof *lhits * map_size / Nnz);
        cond = SAFEREALLOC(cond, sizeof *cond * map_size / Nnz);
    }

    // get maps from all processes and combine them

    int *lstid = SAFEMALLOC(sizeof *lstid * map_size);
    for (int i = 0; i < map_size; i++) {
        lstid[i] = A.lindices[i + Nnz * A.trash_pix];
    }

    if (rank != 0) {
        MPI_Send(&map_size, 1, MPI_INT, 0, 0, comm);
        MPI_Send(lstid, map_size, MPI_INT, 0, 1, comm);
        MPI_Send(x, map_size, MPI_DOUBLE, 0, 2, comm);
        MPI_Send(cond, map_size / Nnz, MPI_DOUBLE, 0, 3, comm);
        MPI_Send(lhits, map_size / Nnz, MPI_INT, 0, 4, comm);
    }

    if (rank == 0) {
        int npix = 12 * nside * nside;
        int oldsize;

        double *mapI = NULL;
        if (Nnz == 3) {
            mapI = SAFECALLOC(npix, sizeof *mapI);
        }
        double *mapQ = SAFECALLOC(npix, sizeof *mapQ);
        double *mapU = SAFECALLOC(npix, sizeof *mapU);
        int *hits = SAFECALLOC(npix, sizeof *hits);
        double *Cond = SAFECALLOC(npix, sizeof *Cond);

        for (int i = 0; i < size; i++) {
            if (i != 0) {
                oldsize = map_size;
                MPI_Recv(&map_size, 1, MPI_INT, i, 0, comm, &status);
                if (oldsize != map_size) {
                    lstid = SAFEREALLOC(lstid, sizeof *lstid * map_size);
                    x = SAFEREALLOC(x, sizeof *x * map_size);
                    cond = SAFEREALLOC(cond, sizeof *cond * map_size);
                    lhits = SAFEREALLOC(lhits, sizeof *lhits * map_size);
                }
                MPI_Recv(lstid, map_size, MPI_INT, i, 1, comm, &status);
                MPI_Recv(x, map_size, MPI_DOUBLE, i, 2, comm, &status);
                MPI_Recv(cond, map_size / Nnz, MPI_DOUBLE, i, 3, comm, &status);
                MPI_Recv(lhits, map_size / Nnz, MPI_INT, i, 4, comm, &status);
            }
            x2map_pol(mapI, mapQ, mapU, Cond, hits, x, lstid, cond, lhits,
                      map_size, Nnz);
        }
        puts("Checking output directory... old files will be overwritten");
        char Imap_name[FILENAME_MAX];
        char Qmap_name[FILENAME_MAX];
        char Umap_name[FILENAME_MAX];
        char Condmap_name[FILENAME_MAX];
        char Hitsmap_name[FILENAME_MAX];
        char nest = 1;
        char *cordsys = "C";
        int ret, w = 1;

        if (Nnz == 3) {
            sprintf(Imap_name, "%s/mapI_%s.fits", outpath, ref);
            if (access(Imap_name, F_OK) != -1) {
                ret = remove(Imap_name);
                if (ret != 0) {
                    printf("Error: unable to delete the file %s\n", Imap_name);
                    w = 0;
                }
            }
        }

        sprintf(Qmap_name, "%s/mapQ_%s.fits", outpath, ref);
        sprintf(Umap_name, "%s/mapU_%s.fits", outpath, ref);
        sprintf(Condmap_name, "%s/Cond_%s.fits", outpath, ref);
        sprintf(Hitsmap_name, "%s/Hits_%s.fits", outpath, ref);

        if (access(Qmap_name, F_OK) != -1) {
            ret = remove(Qmap_name);
            if (ret != 0) {
                printf("Error: unable to delete the file %s\n", Qmap_name);
                w = 0;
            }
        }

        if (access(Umap_name, F_OK) != -1) {
            ret = remove(Umap_name);
            if (ret != 0) {
                printf("Error: unable to delete the file %s\n", Umap_name);
                w = 0;
            }
        }

        if (access(Condmap_name, F_OK) != -1) {
            ret = remove(Condmap_name);
            if (ret != 0) {
                printf("Error: unable to delete the file %s\n", Condmap_name);
                w = 0;
            }
        }

        if (access(Hitsmap_name, F_OK) != -1) {
            ret = remove(Hitsmap_name);
            if (ret != 0) {
                printf("Error: unable to delete the file %s\n", Hitsmap_name);
                w = 0;
            }
        }

        if (w == 1) {
            printf("Writing HEALPix maps FITS files to %s...\n", outpath);
            if (Nnz == 3) {
                write_map(mapI, TDOUBLE, nside, Imap_name, nest, cordsys);
            }
            write_map(mapQ, TDOUBLE, nside, Qmap_name, nest, cordsys);
            write_map(mapU, TDOUBLE, nside, Umap_name, nest, cordsys);
            write_map(Cond, TDOUBLE, nside, Condmap_name, nest, cordsys);
            write_map(hits, TINT, nside, Hitsmap_name, nest, cordsys);
        } else {
            fprintf(stderr, "IO Error: Could not overwrite old files, map "
                            "results will not be stored ;(\n");
        }

        FREE(mapI);
        FREE(mapQ);
        FREE(mapU);
        FREE(Cond);
        FREE(hits);
    }

    elapsed = MPI_Wtime() - st;
    if (rank == 0) {
        printf("Total time = %lf s\n", elapsed);
        fflush(stdout);
    }

    // free memory
    FREE(x);
    FREE(cond);
    FREE(lhits);

    MatFree(&A);
    A.indices = NULL;
    A.values = NULL;
    FREE(lstid);

    // MPI_Finalize();
}

void init_templates(int store_hwp, TemplateClass *X, double *B, int **az_binned,
                    int *hwp_bins, double ***hwp_mod, int *ces_length,
                    Tpltz *Nm1, int npoly, int nhwp, double delta_t, int ground,
                    int n_sss_bins, int **sweeptstamps, int *nsweeps,
                    double **az, double *az_min, double *az_max,
                    double **hwp_angle, int nces, int nb_blocks_loc,
                    void *local_blocks_sizes, double sample_rate, int rank) {
    int n_class = 0;
    hwpss_w hwpss_wghts;
    int ndet = nb_blocks_loc / nces;
    int **detnsweeps = SAFEMALLOC(sizeof *detnsweeps * nces);
    ces_length = SAFEMALLOC(sizeof *ces_length * nces);
    hwp_bins = SAFEMALLOC(sizeof *hwp_bins * nces);

    // Polynomial templates metadata
    for (int i = 0; i < nces; i++) {
        detnsweeps[i] = SAFECALLOC(ndet, sizeof *detnsweeps[i]);
        for (int j = 0; j < ndet; j++)
            detnsweeps[i][j] = nsweeps[i];
        ces_length[i] = sweeptstamps[i][nsweeps[i]];
    }

    // Binned boresight azimuth
    az_binned =
        bin_az(az, az_min, az_max, ces_length, ground, n_sss_bins, nces);

    // HWP harmonics
    hwp_mod = SAFEMALLOC(sizeof *hwp_mod * nces);
    if (nhwp) {
        for (int i = 0; i < nces; i++) {
            hwp_mod[i] = SAFEMALLOC(sizeof *hwp_mod[i] * 2);
            hwp_mod[i][0] =
                SAFECALLOC(ces_length[i], sizeof *hwp_mod[i][0]); // hwp_cos[i];
            hwp_mod[i][1] =
                SAFECALLOC(ces_length[i], sizeof *hwp_mod[i][1]); // hwp_sin[i];
            for (int j = 0; j < ces_length[i]; j++) {
                // hwp_angle_bis = (double)(2*M_PI*hwp_f*j)/sample_rate;
                hwp_mod[i][0][j] = cos(hwp_angle[i][j]);
                hwp_mod[i][1][j] = sin(hwp_angle[i][j]);
            }
        }
    }

    // Set number of template classes
    n_class = npoly + ground + 2 * nhwp;

    // Allocate memory to the templates classes instances
    X = SAFEMALLOC(sizeof *X * n_class * nb_blocks_loc);

    // Initialize templates classes list
    Tlist_init(X, ndet, nces, (int *)local_blocks_sizes, detnsweeps, ces_length,
               sweeptstamps, n_sss_bins, az_binned, sample_rate, npoly, ground,
               nhwp, delta_t, store_hwp, hwp_mod);

    // Free memory
    for (int i = 0; i < nces; i++)
        FREE(detnsweeps[i]);
    FREE(detnsweeps);

    // Allocate memory for the list of kernel blocks and inv block container
    int global_size_kernel = 0;
    int id_kernelblock = 0;
    int ksize = 0;
    for (int i = 0; i < nces; i++) {
        ksize =
            npoly * nsweeps[i] + ground * n_sss_bins + 2 * nhwp * hwp_bins[i];
        hwp_bins[i] = (int)ceil(ces_length[i] / (delta_t * sample_rate));
        global_size_kernel += ndet * ksize * ksize;
    }
    B = SAFECALLOC(global_size_kernel, sizeof *B);
    ksize = npoly * nsweeps[0] + ground * n_sss_bins + 2 * nhwp * hwp_bins[0];
    double *Binv = SAFECALLOC(ksize * ksize, sizeof *Binv);

    // Build the list of inverse kernel blocks
    for (int i = 0; i < nces; i++) {
        if (store_hwp == 0)
            build_hwpss_w(&hwpss_wghts, hwp_mod[i], ces_length[i], nhwp, i);

        ksize =
            npoly * nsweeps[i] + ground * n_sss_bins + 2 * nhwp * hwp_bins[i];
        if (i != 0) {
            Binv = SAFEREALLOC(Binv, sizeof *Binv * ksize * ksize);
            // init to zero
            for (int k = 0; k < ksize * ksize; k++)
                Binv[k] = 0;
        }

        // Processing detector blocks
        for (int j = 0; j < ndet; j++) {
            BuildKernel(
                X + (i * ndet + j) * n_class, n_class, B + id_kernelblock,
                Nm1->tpltzblocks[i * ndet + j].T_block[0], sweeptstamps[i],
                az_binned[i], hwpss_wghts, delta_t, sample_rate);

            int rank_eff = InvKernel(B + id_kernelblock, ksize, Binv);
            if (rank == 0) {
                printf(
                    "[rank %d] Effective rank of local kernel block %d = %d\n",
                    rank, i * ndet + j, rank_eff);
                fflush(stdout);
            }
            for (int k = 0; k < ksize * ksize; k++) {
                B[id_kernelblock + k] = Binv[k];
                Binv[k] = 0;
            }
            id_kernelblock += ksize * ksize;
        }
        if (store_hwp == 0)
            free_hwpss_w(&hwpss_wghts, nhwp);
    }
    FREE(Binv);
}

void x2map_pol(double *mapI, double *mapQ, double *mapU, double *Cond,
               int *hits, const double *x, const int *lstid, const double *cond,
               const int *lhits, int xsize, int nnz) {
    for (int i = 0; i < xsize; i++) {
        if (nnz == 3) {
            // I, Q and U maps
            if (i % nnz == 0) {
                mapI[(int)(lstid[i] / nnz)] = x[i];
                hits[(int)(lstid[i] / nnz)] = lhits[(int)(i / nnz)];
                Cond[(int)(lstid[i] / nnz)] = cond[(int)(i / nnz)];
            } else if (i % nnz == 1) {
                mapQ[(int)(lstid[i] / nnz)] = x[i];
            } else {
                mapU[(int)(lstid[i] / nnz)] = x[i];
            }
        } else {
            // only Q and U maps are estimated
            if (i % nnz == 0) {
                mapQ[(int)(lstid[i] / nnz)] = x[i];
                hits[(int)(lstid[i] / nnz)] = lhits[(int)(i / nnz)];
                Cond[(int)(lstid[i] / nnz)] = cond[(int)(i / nnz)];
            } else {
                mapU[(int)(lstid[i] / nnz)] = x[i];
            }
        }
    }
}

WeightStgy handle_gaps(Gap *Gaps, Mat *A, Tpltz *Nm1, Tpltz *N, GapStrategy gs,
                       double *b, const double *noise, bool do_gap_filling,
                       uint64_t realization, const uint64_t *detindxs,
                       const uint64_t *obsindxs, const uint64_t *telescopes,
                       double sample_rate) {
    int my_rank;
    MPI_Comm_rank(A->comm, &my_rank);

    compute_gaps_per_block(Gaps, Nm1->nb_blocks_loc, Nm1->tpltzblocks);
    copy_gap_info(Nm1->nb_blocks_loc, Nm1->tpltzblocks, N->tpltzblocks);

#if 0
    if (my_rank == 0) {
        puts("gap informations");
        for (int i = 0; i < Nm1->nb_blocks_loc; ++i) {
            printf("block %d: first %d last %d\n", i,
                   Nm1->tpltzblocks[i].first_gap, Nm1->tpltzblocks[i].last_gap);
        }
        fflush(stdout);
    }

    MPI_Barrier(A->comm);
#endif

    WeightStgy ws;

    // When not doing gap-filling, set signal in the gaps to zero
    const bool reset_signal_in_gaps = !do_gap_filling;

    switch (gs) {

    case COND:
        // set noise weighting strategy
        ws = BASIC;

        if (my_rank == 0) {
            puts("[Gaps/conditioning] weighting strategy = BASIC");
        }

        // set signal in all gaps to zero
        if (reset_signal_in_gaps) {
            reset_relevant_gaps(b, Nm1, Gaps);
        }

        // this is not needed any more
        // condition_extra_pix_zero(A);

        // recombine signal and noise
        for (int i = 0; i < A->m; ++i) {
            b[i] += noise[i];
        }

        if (do_gap_filling) {
            perform_gap_filling(A->comm, N, Nm1, b, Gaps, realization, detindxs,
                                obsindxs, telescopes, sample_rate, true);
        } else {
            // perfect noise reconstruction
            if (my_rank == 0) {
                puts("[Gaps/conditioning] perfect noise reconstruction");
            }
        }

        break;

    case MARG_LOCAL_SCAN:
        // set noise weighting strategy
        ws = BASIC;

        if (my_rank == 0) {
            puts("[Gaps/marginalization] weighting strategy = BASIC");
        }

        // set signal in all gaps to zero
        if (reset_signal_in_gaps) {
            reset_relevant_gaps(b, Nm1, Gaps);
        }

        // recombine signal and noise
        for (int i = 0; i < A->m; ++i) {
            b[i] += noise[i];
        }

        if (do_gap_filling) {
            perform_gap_filling(A->comm, N, Nm1, b, Gaps, realization, detindxs,
                                obsindxs, telescopes, sample_rate, true);
        } else {
            // perfect noise reconstruction
            if (my_rank == 0) {
                puts("[Gaps/marginalization] perfect noise reconstruction");
            }
        }

        break;

    case NESTED_PCG:
        // set noise weighting strategy
        ws = ITER;

        if (my_rank == 0) {
            puts("[Gaps/nested] weighting strategy = ITER");
        }

        // recombine signal and noise
        for (int i = 0; i < A->m; ++i) {
            b[i] += noise[i];
        }

        break;

    case NESTED_PCG_NO_GAPS:
        // set noise weighting strategy
        ws = ITER_IGNORE;

        if (my_rank == 0) {
            puts("[Gaps/nested-ignore] weighting strategy = ITER_IGNORE");
        }

        // set signal in all gaps to zero
        if (reset_signal_in_gaps) {
            reset_relevant_gaps(b, Nm1, Gaps);
        }

        // recombine signal and noise
        for (int i = 0; i < A->m; ++i) {
            b[i] += noise[i];
        }

        if (do_gap_filling) {
            perform_gap_filling(A->comm, N, Nm1, b, Gaps, realization, detindxs,
                                obsindxs, telescopes, sample_rate, true);
        } else {
            // perfect noise reconstruction
            if (my_rank == 0) {
                puts("[Gaps/nested-ignore] perfect noise reconstruction");
            }
        }

        break;

    case MARG_PROC:
        // set noise weighting strategy
        ws = BASIC;

        if (my_rank == 0) {
            puts("[Gaps/marginalization] weighting strategy = BASIC");
        }

        // set signal in all gaps to zero
        if (reset_signal_in_gaps) {
            reset_relevant_gaps(b, Nm1, Gaps);
        }

        // recombine signal and noise
        for (int i = 0; i < A->m; ++i) {
            b[i] += noise[i];
        }

        if (do_gap_filling) {
            perform_gap_filling(A->comm, N, Nm1, b, Gaps, realization, detindxs,
                                obsindxs, telescopes, sample_rate, true);
        } else {
            // perfect noise reconstruction
            if (my_rank == 0) {
                puts("[Gaps/marginalization] perfect noise reconstruction");
            }
        }

        break;
    }
    fflush(stdout);

    return ws;
}
