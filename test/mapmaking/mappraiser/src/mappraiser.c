// MAPPRAISER vdev
// Massively parallel iterative map-making code using the Midapack library v1.2b, Nov 2012
// The routine processes expanded pointing, signal and noise data arrays and produces maps in fits files

/** @file   mappraiser.c
    @author Hamza El Bouhargani
    @date   May 2019 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "fitsio.h"
#include "midapack.h"
#include "mappraiser.h"
#include <mkl.h>

int x2map_pol( double *mapI, double *mapQ, double *mapU, double *Cond, int * hits, int npix, double *x, int *lstid, double *cond, int *lhits, int xsize);

double rand_gen();

double normalRandom();

void MLmap(MPI_Comm comm, char *outpath, char *ref, int solver, int pointing_commflag, double tol, int maxiter, int enlFac, int ortho_alg, int bs_red, int nside, void *data_size_proc, int nb_blocks_loc, void *local_blocks_sizes, int Nnz, void *pix, void *pixweights, void *signal, double *noise, int lambda, double *invtt)
{
  int64_t	M;       //Global number of rows
  int		m, Nb_t_Intervals;  //local number of rows of the pointing matrix A, nbr of stationary intervals
  int64_t	gif;			//global indice for the first local line
  int		i, j, k;
  // int          K;	  //maximum number of iteration for the PCG
  // double 	tol;			//residual tolerence for the PCG
  Mat	A;			        //pointing matrix structure
  int 		*id0pix, *ll;
  // int 		pointing_commflag;	//option for the communication scheme for the pointing matrix
  double	*x;	//pixel domain vectors
  double	st, t;		 	//timer, start time
  int 		rank, size;
  MPI_Status status;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if(rank==0){
    printf("\n############# MAPPRAISER : MidAPack PaRAllel Iterative Sky EstimatoR vDev, May 2019 ################\n");
    printf("rank = %d, size = %d\n",rank,size);
  }
  fflush(stdout);

//communication scheme for the pointing matrix  (to move in .h)
  // pointing_commflag=6; //2==BUTTERFLY - 1==RING - 6==MPI_Allreduce

//PCG parameters
  // tol=pow(10,-6);
  // K=500;

//total length of the time domaine signal
  M = 0;
  for(i=0;i<size;i++){
    M += ((int *)data_size_proc)[i];
  }
  if(rank==0){
    printf("[rank %d] M=%ld\n", rank, M);
  }
  fflush(stdout);

//compute distribution indexes over the processes
  m = ((int *)data_size_proc)[rank];
  gif = 0;
  for(i=0;i<rank;i++)
    gif += ((int *)data_size_proc)[i];

//Print information on data distribution
  int Nb_t_Intervals_loc = nb_blocks_loc;
  MPI_Allreduce(&nb_blocks_loc, &Nb_t_Intervals, 1, MPI_INT, MPI_SUM, comm);
  int nb_proc_shared_one_interval = 1; //max(1, size/Nb_t_Intervals );
  if(rank==0){
    printf("[rank %d] nb_proc_shared_one_interval=%d\n", rank, nb_proc_shared_one_interval );
    // int t_Interval_length_loc = t_Interval_length/nb_proc_shared_one_interval;
    printf("[rank %d] size=%d \t m=%d \t Nb_t_Intervals=%d \n", rank, size, m, Nb_t_Intervals);
    printf("[rank %d] Nb_t_Intervals_loc=%d \n", rank, Nb_t_Intervals_loc );
    fflush(stdout);
  }


//Pointing matrix init
  st=MPI_Wtime();
  A.trash_pix =0;
  MatInit( &A, m, Nnz, pix, pixweights, pointing_commflag, comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Initializing pointing matrix time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  // printf("A.lcount = %d\n", A.lcount);

//Build pixel-to-time domain mapping
  st=MPI_Wtime();
  id0pix = (int *) malloc(A.lcount/(A.nnz) * sizeof(int)); //index of the last time sample pointing to each pixel
  ll = (int *) malloc(m * sizeof(int)); //linked list of time samples indexes

  //initialize the mapping arrays to -1
  for(i=0; i<m; i++){
    ll[i] = -1;
  }
  for(j=0; j<A.lcount/(A.nnz); j++){
    id0pix[j] = -1;
  }
  //build the linked list chain of time samples corresponding to each pixel
  for(i=0;i<m;i++){
    if(id0pix[A.indices[i*A.nnz]/(A.nnz)] == -1)
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
    else{
      ll[i] = id0pix[A.indices[i*A.nnz]/(A.nnz)];
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
    }
  }
  A.id0pix = id0pix;
  A.ll = ll;
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Total pixel-to-time domain mapping time=%lf \n", rank, t-st);
    fflush(stdout);
  }

//PCG beginning vector input definition for the pixel domain map (MatInit gives A.lcount)
  int *lhits;
  double *cond;
  x   = (double *) malloc(A.lcount*sizeof(double));
  cond = (double *) malloc((int)(A.lcount/3)*sizeof(double));
  lhits = (int *) malloc((int)(A.lcount/3) * sizeof(int));
  for(j=0; j<A.lcount; j++){
    x[j] = 0.;
    if(j%3 == 0){
      lhits[(int)(j/3)] = 0;
      cond[(int)(j/3)] = 0.;
    }
  }
  // printf("[rank %d] npix = %d\n",rank,A.lcount);
//Create piecewise Toeplitz matrix
//specifics parameters:
  int nb_blocks_tot = Nb_t_Intervals;
  // int n_block_avg = M/nb_blocks_tot;  //should be equal to t_Intervals_length in the current config
  //                                     //because we dont have flotting blocks
  int lambda_block_avg = lambda;

//flags for Toeplitz product strategy
  Flag flag_stgy;
  flag_stgy_init_auto(&flag_stgy);

//to print something on screen
  flag_stgy.flag_verbose=1;

//define Toeplitz blocks list and structure for Nm1
  Block *tpltzblocks;
  Tpltz Nm1;

//dependants parameters:
  int64_t nrow = M;
  int mcol = 1;

  int64_t id0 = gif;
  int local_V_size = m;

  // int nb_blocks_loc;
  // nb_blocks_loc = ceil( local_V_size*1.0/n_block_avg );

  // double nb_blocks_loc_part =  (local_V_size*1.0)/(n_block_avg) ;

// // check special cases to have exact number of local blocks
//   if ((id0/n_block_avg + nb_blocks_loc) * n_block_avg < (id0+local_V_size))
//     nb_blocks_loc=nb_blocks_loc+1;

  if (rank==0 | rank==1) {
    printf("M=%ld, m=%d \n", M, m);
    printf("gif = %ld \n", gif);
  }

  // int nb_proc_shared_a_block = ceil( size*1.0/nb_blocks_tot );
  // int nb_comm = (nb_proc_shared_a_block)-1 ;

//Block definition
  tpltzblocks = (Block *) malloc(nb_blocks_loc * sizeof(Block));
  defineBlocks_avg(tpltzblocks, invtt, nb_blocks_loc, local_blocks_sizes, lambda_block_avg, id0 );
  defineTpltz_avg( &Nm1, nrow, 1, mcol, tpltzblocks, nb_blocks_loc, nb_blocks_tot, id0, local_V_size, flag_stgy, comm);

//print Toeplitz parameters for information
  if (rank==0 | rank==1) {
    printf("[rank %d] size=%d, nrow=%ld, local_V_size=%d, id0=%ld \n", rank, size, nrow, local_V_size, id0);
    printf("[rank %d] nb_blocks_tot=%d, nb_blocks_loc=%d, lambda_block_avg=%d \n", rank, nb_blocks_tot, nb_blocks_loc, lambda_block_avg);
    // printf("[rank %d] nb_proc_shared_a_block=%d, nb_comm=%d \n", rank, nb_proc_shared_a_block, nb_comm);
  }

  MPI_Barrier(comm);
   if(rank==0)
 printf("##### Start PCG ####################\n");
 fflush(stdout);

  //Hard coded parameters (To be removed)
  // int solver = 1;
  // int maxIter = 250;
  // int enlFac = 16;
  // int ortho_alg = 1;
  // int bs_red = 0;

  st=MPI_Wtime();
// Conjugate Gradient
  if(solver==0)
    PCG_GLS_true(outpath, ref, &A, Nm1, x, signal, noise, cond, lhits, tol, maxiter);
  else
    ECG_GLS(outpath, ref, &A, Nm1, x, signal, noise, cond, lhits, tol, maxiter, enlFac, ortho_alg, bs_red);
  MPI_Barrier(comm);
  t=MPI_Wtime();
   if(rank==0)
 printf("##### End PCG ####################\n");
  if (rank==0) {
    printf("[rank %d] Total PCG time=%lf \n", rank, t-st);
  }
  fflush(stdout);

//write output to fits files:
  st=MPI_Wtime();
  int mapsize = A.lcount-(A.nnz)*(A.trash_pix);
  int map_id = rank;

  int *lstid;
  lstid = (int *) calloc(mapsize, sizeof(int));
  for(i=0; i< mapsize; i++){
    lstid[i] = A.lindices[i+(A.nnz)*(A.trash_pix)];
  }

  if (rank!=0){
    MPI_Send(&mapsize, 1, MPI_INT, 0, 0, comm);
    MPI_Send(lstid, mapsize, MPI_INT, 0, 1, comm);
    MPI_Send(x, mapsize, MPI_DOUBLE, 0, 2, comm);
    MPI_Send(cond, mapsize/Nnz, MPI_DOUBLE, 0, 3, comm);
    MPI_Send(lhits, mapsize/Nnz, MPI_INT, 0, 4, comm);
  }

  if (rank==0){
    int npix = 12*pow(nside,2);
    int oldsize;

    double *mapI;
    mapI    = (double *) calloc(npix, sizeof(double));
    double *mapQ;
    mapQ    = (double *) calloc(npix, sizeof(double));
    double *mapU;
    mapU    = (double *) calloc(npix, sizeof(double));
    int *hits;
    hits = (int *) calloc(npix, sizeof(int));
    double *Cond;
    Cond = (double *) calloc(npix, sizeof(double));

    for (i=0;i<size;i++){
      if (i!=0){
        oldsize = mapsize;
        MPI_Recv(&mapsize, 1, MPI_INT, i, 0, comm, &status);
        if (oldsize!=mapsize){
          lstid = (int *) realloc(lstid, mapsize*sizeof(int));
          x = (double *) realloc(x, mapsize*sizeof(double));
          cond = (double *) realloc(cond, mapsize*sizeof(double));
          lhits = (int *) realloc(lhits, mapsize*sizeof(int));
        }
        MPI_Recv(lstid, mapsize, MPI_INT, i, 1, comm, &status);
        MPI_Recv(x, mapsize, MPI_DOUBLE, i, 2, comm, &status);
        MPI_Recv(cond, mapsize/Nnz, MPI_DOUBLE, i, 3, comm, &status);
        MPI_Recv(lhits, mapsize/Nnz, MPI_INT, i, 4, comm, &status);
      }
      x2map_pol(mapI, mapQ, mapU, Cond, hits, npix, x, lstid, cond, lhits, mapsize);
    }
    printf("Checking output directory ... old files will be overwritten\n");
    char Imap_name[256];
    char Qmap_name[256];
    char Umap_name[256];
    char Condmap_name[256];
    char Hitsmap_name[256];
    char nest = 1;
    char *cordsys = "C";
    int ret,w=1;

    // sprintf(Imap_name,"/global/cscratch1/sd/elbouha/data_TOAST/output/mapI_%s.fits",outpath);
    // sprintf(Qmap_name,"/global/cscratch1/sd/elbouha/data_TOAST/output/mapQ_%s.fits",outpath);
    // sprintf(Umap_name,"/global/cscratch1/sd/elbouha/data_TOAST/output/mapU_%s.fits",outpath);
    // sprintf(Condmap_name,"/global/cscratch1/sd/elbouha/data_TOAST/output/Cond_%s.fits",outpath);
    // sprintf(Hitsmap_name,"/global/cscratch1/sd/elbouha/data_TOAST/output/Hits_%s.fits",outpath);
    sprintf(Imap_name,"%s/mapI_%s.fits", outpath, ref);
    sprintf(Qmap_name,"%s/mapQ_%s.fits", outpath, ref);
    sprintf(Umap_name,"%s/mapU_%s.fits", outpath, ref);
    sprintf(Condmap_name,"%s/Cond_%s.fits", outpath, ref);
    sprintf(Hitsmap_name,"%s/Hits_%s.fits", outpath, ref);


    if( access( Imap_name, F_OK ) != -1 ) {
      ret = remove(Imap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Imap_name);
        w = 0;
      }
    }

    if( access( Qmap_name, F_OK ) != -1 ) {
      ret = remove(Qmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Qmap_name);
        w = 0;
      }
    }

    if( access( Umap_name, F_OK ) != -1 ) {
      ret = remove(Umap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Umap_name);
        w = 0;
      }
    }

    if( access( Condmap_name, F_OK ) != -1 ) {
      ret = remove(Condmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Condmap_name);
        w = 0;
      }
    }

    if( access( Hitsmap_name, F_OK ) != -1 ) {
      ret = remove(Hitsmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Hitsmap_name);
        w = 0;
      }
    }

    if(w==1){
      printf("Writing HEALPix maps FITS files ...\n");
      write_map(mapI, TDOUBLE, nside, Imap_name, nest, cordsys);
      write_map(mapQ, TDOUBLE, nside, Qmap_name, nest, cordsys);
      write_map(mapU, TDOUBLE, nside, Umap_name, nest, cordsys);
      write_map(Cond, TDOUBLE, nside, Condmap_name, nest, cordsys);
      write_map(hits, TINT, nside, Hitsmap_name, nest, cordsys);
    }
    else{
      printf("IO Error: Could not overwrite old files, map results will not be stored ;(\n");
    }

  }

  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Write output files time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  st=MPI_Wtime();

  MatFree(&A);
  A.indices = NULL;
  A.values = NULL;                                                //free memory
  free(x);
  free(cond);
  free(lhits);
  free(tpltzblocks);
  MPI_Barrier(comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Free memory time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  // MPI_Finalize();
}

void MTmap(MPI_Comm comm, char *outpath, char *ref, int solver, int pointing_commflag, int npoly, int nhwp, double delta_t, int ground, int n_sss_bins, double tol, int maxiter, int enlFac, int ortho_alg, int bs_red, int nside, int **sweeptstamps, int *nsweeps, double **az, double *az_min, double *az_max, double **hwp_angle, int nces, void *data_size_proc, int nb_blocks_loc, void *local_blocks_sizes, int Nnz, void *pix, void *pixweights, void *signal, double *noise, double sampling_freq, double hwp_rpm, double *invtt)
{
  int64_t	M;       //Global number of rows
  int		m, Nb_t_Intervals;  //local number of rows of the pointing matrix A, nbr of stationary intervals
  int64_t	gif;			//global indice for the first local line
  int		i, j, k, l;
  Mat	A;			        //pointing matrix structure
  int 		*id0pix, *ll;
  double	*x;	//pixel domain vectors
  double	st, t;		 	//timer, start time
  int 		rank, size;
  MPI_Status status;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  mkl_set_num_threads(1); // required for some obscure reason when working with mkl routines
  if(rank==0){
    printf("\n############# MAPPRAISER : MidAPack PaRAllel Iterative Sky EstimatoR vDev, May 2019 ################\n");
    printf("rank = %d, size = %d\n",rank,size);
  }
  fflush(stdout);

//total length of the time domaine signal
  M = 0;
  for(i=0;i<size;i++){
    M += ((int *)data_size_proc)[i];
  }
  if(rank==0){
    printf("[rank %d] M=%ld\n", rank, M);
  }
  fflush(stdout);

//compute distribution indexes over the processes
  m = ((int *)data_size_proc)[rank];
  gif = 0;
  for(i=0;i<rank;i++)
    gif += ((int *)data_size_proc)[i];

//Print information on data distribution
  int Nb_t_Intervals_loc = nb_blocks_loc;
  MPI_Allreduce(&nb_blocks_loc, &Nb_t_Intervals, 1, MPI_INT, MPI_SUM, comm);
  int nb_proc_shared_one_interval = 1; //max(1, size/Nb_t_Intervals );
  if(rank==0){
    printf("[rank %d] nb_proc_shared_one_interval=%d\n", rank, nb_proc_shared_one_interval );
    // int t_Interval_length_loc = t_Interval_length/nb_proc_shared_one_interval;
    printf("[rank %d] size=%d \t m=%d \t Nb_t_Intervals=%d \n", rank, size, m, Nb_t_Intervals);
    printf("[rank %d] Nb_t_Intervals_loc=%d \n", rank, Nb_t_Intervals_loc );
    fflush(stdout);
  }

//Pointing matrix init
  st=MPI_Wtime();
  A.trash_pix =0;
  MatInit( &A, m, Nnz, pix, pixweights, pointing_commflag, comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Initializing pointing matrix time=%lf \n", rank, t-st);
    fflush(stdout);
  }

//Build pixel-to-time domain mapping
  st=MPI_Wtime();
  id0pix = (int *) malloc(A.lcount/(A.nnz) * sizeof(int)); //index of the last time sample pointing to each pixel
  ll = (int *) malloc(m * sizeof(int)); //linked list of time samples indexes

  //initialize the mapping arrays to -1
  for(i=0; i<m; i++){
    ll[i] = -1;
  }
  for(j=0; j<A.lcount/(A.nnz); j++){
    id0pix[j] = -1;
  }
  //build the linked list chain of time samples corresponding to each pixel
  for(i=0;i<m;i++){
    if(id0pix[A.indices[i*A.nnz]/(A.nnz)] == -1)
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
    else{
      ll[i] = id0pix[A.indices[i*A.nnz]/(A.nnz)];
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
    }
  }
  A.id0pix = id0pix;
  A.ll = ll;
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Total pixel-to-time domain mapping time=%lf \n", rank, t-st);
    fflush(stdout);
  }

//PCG beginning vector input definition for the pixel domain map (MatInit gives A.lcount)
  int *lhits;
  double *cond;
  x   = (double *) malloc(A.lcount*sizeof(double));
  cond = (double *) malloc((int)(A.lcount/3)*sizeof(double));
  lhits = (int *) malloc((int)(A.lcount/3) * sizeof(int));
  for(j=0; j<A.lcount; j++){
    x[j] = 0.;
    if(j%3 == 0){
      lhits[(int)(j/3)] = 0;
      cond[(int)(j/3)] = 0.;
    }
  }
  // printf("[rank %d] npix = %d\n",rank,A.lcount);
//Create piecewise Toeplitz matrix
//specifics parameters:
  int nb_blocks_tot = Nb_t_Intervals;
  // int n_block_avg = M/nb_blocks_tot;  //should be equal to t_Intervals_length in the current config
  //                                     //because we dont have flotting blocks
  int lambda_block_avg = 1;

//flags for Toeplitz product strategy
  Flag flag_stgy;
  flag_stgy_init_auto(&flag_stgy);

//to print something on screen
  flag_stgy.flag_verbose=1;

//define Toeplitz blocks list and structure for Nm1
  Block *tpltzblocks;
  Tpltz Nm1;

//dependants parameters:
  int64_t nrow = M;
  int mcol = 1;

  int64_t id0 = gif;
  int local_V_size = m;

  // int nb_blocks_loc;
  // nb_blocks_loc = ceil( local_V_size*1.0/n_block_avg );

  // double nb_blocks_loc_part =  (local_V_size*1.0)/(n_block_avg) ;

// // check special cases to have exact number of local blocks
//   if ((id0/n_block_avg + nb_blocks_loc) * n_block_avg < (id0+local_V_size))
//     nb_blocks_loc=nb_blocks_loc+1;

  if (rank==0 | rank==1) {
    printf("M=%ld, m=%d \n", M, m);
    printf("gif = %ld \n", gif);
  }

  // int nb_proc_shared_a_block = ceil( size*1.0/nb_blocks_tot );
  // int nb_comm = (nb_proc_shared_a_block)-1 ;

//Block definition
  tpltzblocks = (Block *) malloc(nb_blocks_loc * sizeof(Block));
  defineBlocks_avg(tpltzblocks, invtt, nb_blocks_loc, local_blocks_sizes, lambda_block_avg, id0 );
  defineTpltz_avg( &Nm1, nrow, 1, mcol, tpltzblocks, nb_blocks_loc, nb_blocks_tot, id0, local_V_size, flag_stgy, comm);

//print Toeplitz parameters for information
  if (rank==0 | rank==1) {
    printf("[rank %d] size=%d, nrow=%ld, local_V_size=%d, id0=%ld \n", rank, size, nrow, local_V_size, id0);
    printf("[rank %d] nb_blocks_tot=%d, nb_blocks_loc=%d, lambda_block_avg=%d \n", rank, nb_blocks_tot, nb_blocks_loc, lambda_block_avg);
    // printf("[rank %d] nb_proc_shared_a_block=%d, nb_comm=%d \n", rank, nb_proc_shared_a_block, nb_comm);
  }

  // //Sanity checks
  // if(rank == 0){
  //   printf("nces = %d\n",nces);
  //   for(i=0;i<nces;i++){
  //     printf("nsweeps[%d] = %d\n",i,nsweeps[i]);
  //     for(j=0;j<nsweeps[i];j++){
  //       printf("sweeptstamps[%d][%d] = %d\n",i,j,sweeptstamps[i][j]);
  //     }
  //   }
  // }
  //Templates classes initialization
  st=MPI_Wtime();
  //hardcoded parameters to be changed later on
  // int npoly = 3;
  // int ground = 0;
  // int n_sss_bins = 20;
  // int nhwp = 0;
  // double delta_t = 10.0;
  int store_hwp = 0;
  int n_class = 0;
  double sigma2;
  hwpss_w hwpss_wghts;
  int ndet = nb_blocks_loc / nces;
  int **detnsweeps = (int **) malloc(nces * sizeof(int*));
  int *ces_length = (int *) malloc(nces * sizeof(int));
  int *hwp_bins = (int *) malloc(nces * sizeof(int));
  for(i=0;i<nces;i++){
    detnsweeps[i] = (int *) malloc(ndet * sizeof(int));
    for(j=0;j<ndet;j++)
      detnsweeps[i][j] = nsweeps[i];
      ces_length[i] = sweeptstamps[i][nsweeps[i]];
  }
  int **az_binned = NULL;
  az_binned = bin_az(az, az_min, az_max, ces_length, ground, n_sss_bins, nces);

  double ***hwp_mod = (double ***) malloc(nces * sizeof(double **));
  // double hwp_f = (double) hwp_rpm / 60 ;
  //double hwp_angle_bis = 0.0;
  if(nhwp){
    for(i=0;i<nces;i++){
      hwp_mod[i] = (double **) malloc(2 * sizeof(double *));
      hwp_mod[i][0] = (double *) calloc(ces_length[i], sizeof(double));//hwp_cos[i];
      hwp_mod[i][1] = (double *) calloc(ces_length[i], sizeof(double));//hwp_sin[i];
      for(j=0;j<ces_length[i];j++){
        //hwp_angle_bis = (double)(2*M_PI*hwp_f*j)/sampling_freq;
        hwp_mod[i][0][j] = cos(hwp_angle[i][j]);
        hwp_mod[i][1][j] = sin(hwp_angle[i][j]);
      }
    }
  }

  // //Simulate HWPSS signal (just harmonics of hwp angle)
  // double hwpss_center = 0.002;
  // double hwpss_dispersion = 0.01;
  // int hwpss_order = 6;
  // double **hwp_amp = (double **) malloc(hwpss_order * sizeof(double*));
  // for(l=0;l<hwpss_order;l++){
  //   hwp_amp[l] = (double *) calloc(nb_blocks_loc, sizeof(double));
  //   for(i=0;i<nb_blocks_loc;i++){
  //     hwp_amp[l][i] = hwpss_center * (1 + normalRandom()*hwpss_dispersion);
  //   }
  // }
  // int id = 0;
  // for(i=0;i<nces;i++){
  //   for(j=0;j<ndet;j++){
  //     for(k=0;k<ces_length[i];k++){
  //       for(l=1;l<=hwpss_order;l++){
  //         noise[id+k] += hwp_amp[l-1][i*ndet + j]*(1+0.0001*k/sampling_freq)*(cos(l*hwp_angle[i][k]) + sin(l*hwp_angle[i][k]));
  //       }
  //     }
  //     id += ces_length[i];
  //   }
  // }


  n_class = npoly + ground + 2*nhwp;
  //Allocate memory to the templates classes instances (polynomials + SSS template)
  TemplateClass *X = (TemplateClass *) malloc(n_class * nb_blocks_loc * sizeof(TemplateClass));
  //Initialize templates classes list
  Tlist_init(X, ndet, nces, (int *)local_blocks_sizes, detnsweeps, ces_length,
  sweeptstamps, n_sss_bins, az_binned, sampling_freq, npoly, ground, nhwp, delta_t,
  store_hwp, hwp_mod);

  //Allocate memory for the list of kernel blocks and inv block container
  int global_size_kernel = 0;
  int id_kernelblock = 0;
  for(i=0;i<nces;i++){
      hwp_bins[i] = (int)ceil(ces_length[i]/(delta_t*sampling_freq));
      // printf("[rank %d] hwp_bins = %d\n", rank, hwp_bins[i]);
    global_size_kernel += ndet * (npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i]) * (npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i]);
  }
  double *B = (double *) calloc(global_size_kernel, sizeof(double));
  double *Binv = (double *) calloc((npoly*nsweeps[0] + ground*n_sss_bins + 2*nhwp*hwp_bins[0])*(npoly*nsweeps[0] + ground*n_sss_bins + 2*nhwp*hwp_bins[0]), sizeof(double));

  //Build the list of inverse kernel blocks
  for(i=0;i<nces;i++){
    if(store_hwp == 0)
      build_hwpss_w(&hwpss_wghts, hwp_mod[i], ces_length[i], nhwp, i);
    // double sum  = 0.;
    // for(j=0;j<100;j++){
    //   sum += Nm1.tpltzblocks[0].T_block[0]*hwpss_wghts.hwpcos[0][j]*hwpss_wghts.hwpsin[0][j];
    //   // printf("hwp_cos[%d] = %e, cos[%d]= %e\n",j,hwpss_wghts.hwpcos[2][j],j,cos(3*2*M_PI*2*j/sampling_freq));
    //   // printf("hwp_sin[%d] = %e, sin[%d]= %e\n",j,hwpss_wghts.hwpsin[2][j],j,sin(3*2*M_PI*2*j/sampling_freq));
    // }
    // printf("sum = %e\n",sum);

    if(i!=0){
      Binv = (double *) realloc(Binv, (npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i])*(npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i])*sizeof(double));
      // init to zero
      for(k=0;k<(npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i])*(npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i]);k++)
        Binv[k] = 0;
    }
    // Processing detector blocks
    for(j=0;j<ndet;j++){
      // fflush(stdout);
      BuildKernel(X+(i*ndet+j)*n_class, n_class, B+id_kernelblock, Nm1.tpltzblocks[i*ndet+j].T_block[0], sweeptstamps[i], az_binned[i], hwpss_wghts, delta_t, sampling_freq);

      // if((i==0) && (j==0))
      //   sigma2 = B[0];
      // printf("i=%d, af kernel\n",i);
      // fflush(stdout);
      // if(j==0)
      //   for( l = 0; l < npoly*nsweeps[i] + ground*n_sss_bins+2*nhwp*hwp_bins[i]; l++ ) {
      //     for(k=0;k < npoly*nsweeps[i] + ground*n_sss_bins+2*nhwp*hwp_bins[i]; k++){
      //       if((B+id_kernelblock)[l*(npoly*nsweeps[i] + ground*n_sss_bins+2*nhwp*hwp_bins[i])+k] > 1e-6 && l!=k)
      //         printf("B[%d,%d]=%e\n",l,k,(B+id_kernelblock)[l*(npoly*nsweeps[i] + ground*n_sss_bins+2*nhwp*hwp_bins[i])+k]);
      //     }
      //   }
      // if(i==0 && rank ==0){
      // for( l = 84*0+0; l < 84*0+5; l++ ) {
      //         for( k = 84*2+0; k < 84*2+5; k++ ) printf( " %6.12f", (B+id_kernelblock)[l*(npoly*nsweeps[i] + ground*n_sss_bins+2*nhwp*hwp_bins[i])+k] );
      //         printf( "\n" );
      // }}
      // printf("i=%d, af init\n",i);
      // printf("bf: nbins = %d\n",(X+1)->nbins);
      // printf("bf: nbinMin = %d\n",(X+1)->nbinMin);
      // printf("bf: nbinMax = %d\n",(X+1)->nbinMax);
      // printf("bf: nsamples = %d\n",(X+1)->nsamples);
      // printf("bf: flagw = %s\n",(X+1)->flag_w);
      // fflush(stdout);
      // for(k=0;k<(npoly*nsweeps[i] + n_sss_bins);k++){
      //   // printf("B[%d] = %f\n",j,(B+i*(npoly*nsweeps)*(npoly*nsweeps))[j]);
      //   B[id_kernelblock + k*(npoly*nsweeps[i] + n_sss_bins) + k] += sigma2;
      // }

      // printf("[rank %d] Effective rank of local kernel block %d = %d\n",rank, i, inverse_svd(npoly*nsweeps, npoly*nsweeps, npoly*nsweeps,  B+i*(npoly*nsweeps)*(npoly*nsweeps)));
      if(rank==0)
        printf("[rank %d] Effective rank of local kernel block %d = %d\n",rank, i*ndet+j, InvKernel(B+id_kernelblock, npoly*nsweeps[i] + ground*n_sss_bins + 2*nhwp*hwp_bins[i], Binv));
      else
        InvKernel(B+id_kernelblock, npoly*nsweeps[i] + ground*n_sss_bins +  2*nhwp*hwp_bins[i], Binv);
      // printf("af: nbins = %d\n",(X+1)->nbins);
      // printf("af: nbinMin = %d\n",(X+1)->nbinMin);
      // printf("af: nbinMax = %d\n",(X+1)->nbinMax);
      // printf("af: nsamples = %d\n",(X+1)->nsamples);
      // printf("af: flagw = %s\n",(X+1)->flag_w);
      fflush(stdout);

      // printf("%d rank kernel = %d, access block = %f\n",i, rang, block[npoly*nsweeps*77+77]);
      // printf("%d rank kernel = %d\n",i, InvKernel(block, npoly*nsweeps, Binv));
      for(k=0;k<(npoly*nsweeps[i] + ground*n_sss_bins +  2*nhwp*hwp_bins[i])*(npoly*nsweeps[i] + ground*n_sss_bins +  2*nhwp*hwp_bins[i]);k++){
      //   // printf("B[%d] = %f\n",j,(B+i*(npoly*nsweeps)*(npoly*nsweeps))[j]);
        B[id_kernelblock+k] = Binv[k];
        Binv[k] = 0;
      }
      id_kernelblock += (npoly*nsweeps[i] + ground*n_sss_bins +  2*nhwp*hwp_bins[i])*(npoly*nsweeps[i] + ground*n_sss_bins +  2*nhwp*hwp_bins[i]);
    }
    if(store_hwp == 0)
      free_hwpss_w(&hwpss_wghts, nhwp);
  }

  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Total time building Templates classes and inverse kernel blocks=%lf \n", rank, t-st);
    fflush(stdout);
  }

  MPI_Barrier(comm);
   if(rank==0)
 printf("##### Start PCG ####################\n");
 fflush(stdout);

  st=MPI_Wtime();
// Conjugate Gradient
  if(solver==0)
    PCG_GLS_templates(outpath, ref, &A, Nm1, X, B, sweeptstamps, npoly, ground, nhwp, nsweeps, az_binned, n_sss_bins, hwp_bins, hwp_mod, delta_t, store_hwp, nces, ces_length, nb_blocks_loc, x, signal, noise, cond, lhits, tol, maxiter, sampling_freq);
  else{
    printf("ECG unavailable at this stage please choose the PCG solver: solver=0\n");
    exit(1);
  }
    // ECG_GLS(outpath, ref, &A, Nm1, x, signal, noise, cond, lhits, tol, maxiter, enlFac, ortho_alg, bs_red);
  MPI_Barrier(comm);
  t=MPI_Wtime();
   if(rank==0)
 printf("##### End PCG ####################\n");
  if (rank==0) {
    printf("[rank %d] Total PCG time=%lf \n", rank, t-st);
  }
  fflush(stdout);

//write output to fits files:
  st=MPI_Wtime();
  int mapsize = A.lcount-(A.nnz)*(A.trash_pix);
  int map_id = rank;

  int *lstid;
  lstid = (int *) calloc(mapsize, sizeof(int));
  for(i=0; i< mapsize; i++){
    lstid[i] = A.lindices[i+(A.nnz)*(A.trash_pix)];
  }

  if (rank!=0){
    MPI_Send(&mapsize, 1, MPI_INT, 0, 0, comm);
    MPI_Send(lstid, mapsize, MPI_INT, 0, 1, comm);
    MPI_Send(x, mapsize, MPI_DOUBLE, 0, 2, comm);
    MPI_Send(cond, mapsize/Nnz, MPI_DOUBLE, 0, 3, comm);
    MPI_Send(lhits, mapsize/Nnz, MPI_INT, 0, 4, comm);
  }

  if (rank==0){
    int npix = 12*pow(nside,2);
    int oldsize;

    double *mapI;
    mapI    = (double *) calloc(npix, sizeof(double));
    double *mapQ;
    mapQ    = (double *) calloc(npix, sizeof(double));
    double *mapU;
    mapU    = (double *) calloc(npix, sizeof(double));
    int *hits;
    hits = (int *) calloc(npix, sizeof(int));
    double *Cond;
    Cond = (double *) calloc(npix, sizeof(double));

    for (i=0;i<size;i++){
      if (i!=0){
        oldsize = mapsize;
        MPI_Recv(&mapsize, 1, MPI_INT, i, 0, comm, &status);
        if (oldsize!=mapsize){
          lstid = (int *) realloc(lstid, mapsize*sizeof(int));
          x = (double *) realloc(x, mapsize*sizeof(double));
          cond = (double *) realloc(cond, mapsize*sizeof(double));
          lhits = (int *) realloc(lhits, mapsize*sizeof(int));
        }
        MPI_Recv(lstid, mapsize, MPI_INT, i, 1, comm, &status);
        MPI_Recv(x, mapsize, MPI_DOUBLE, i, 2, comm, &status);
        MPI_Recv(cond, mapsize/Nnz, MPI_DOUBLE, i, 3, comm, &status);
        MPI_Recv(lhits, mapsize/Nnz, MPI_INT, i, 4, comm, &status);
      }
      x2map_pol(mapI, mapQ, mapU, Cond, hits, npix, x, lstid, cond, lhits, mapsize);
    }
    printf("Checking output directory ... old files will be overwritten\n");
    char Imap_name[256];
    char Qmap_name[256];
    char Umap_name[256];
    char Condmap_name[256];
    char Hitsmap_name[256];
    char nest = 1;
    char *cordsys = "C";
    int ret,w=1;

    sprintf(Imap_name,"%s/mapI_%s.fits", outpath, ref);
    sprintf(Qmap_name,"%s/mapQ_%s.fits", outpath, ref);
    sprintf(Umap_name,"%s/mapU_%s.fits", outpath, ref);
    sprintf(Condmap_name,"%s/Cond_%s.fits", outpath, ref);
    sprintf(Hitsmap_name,"%s/Hits_%s.fits", outpath, ref);


    if( access( Imap_name, F_OK ) != -1 ) {
      ret = remove(Imap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Imap_name);
        w = 0;
      }
    }

    if( access( Qmap_name, F_OK ) != -1 ) {
      ret = remove(Qmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Qmap_name);
        w = 0;
      }
    }

    if( access( Umap_name, F_OK ) != -1 ) {
      ret = remove(Umap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Umap_name);
        w = 0;
      }
    }

    if( access( Condmap_name, F_OK ) != -1 ) {
      ret = remove(Condmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Condmap_name);
        w = 0;
      }
    }

    if( access( Hitsmap_name, F_OK ) != -1 ) {
      ret = remove(Hitsmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Hitsmap_name);
        w = 0;
      }
    }

    if(w==1){
      printf("Writing HEALPix maps FITS files ...\n");
      write_map(mapI, TDOUBLE, nside, Imap_name, nest, cordsys);
      write_map(mapQ, TDOUBLE, nside, Qmap_name, nest, cordsys);
      write_map(mapU, TDOUBLE, nside, Umap_name, nest, cordsys);
      write_map(Cond, TDOUBLE, nside, Condmap_name, nest, cordsys);
      write_map(hits, TINT, nside, Hitsmap_name, nest, cordsys);
    }
    else{
      printf("IO Error: Could not overwrite old files, map results will not be stored ;(\n");
    }

  }

  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Write output files time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  st=MPI_Wtime();

  MatFree(&A);
  A.indices = NULL;
  A.values = NULL;                                                //free memory
  free(x);
  free(cond);
  free(lhits);
  free(tpltzblocks);
  MPI_Barrier(comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Free memory time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  // MPI_Finalize();
}

int x2map_pol( double *mapI, double *mapQ, double *mapU, double *Cond, int * hits, int npix, double *x, int *lstid, double *cond, int *lhits, int xsize)
{

  int i;

  for(i=0; i<xsize; i++){
    if(i%3 == 0){
      mapI[(int)(lstid[i]/3)]= x[i];
      hits[(int)(lstid[i]/3)]= lhits[(int)(i/3)];
      Cond[(int)(lstid[i]/3)]= cond[(int)(i/3)];
    }
    else if (i%3 == 1)
      mapQ[(int)(lstid[i]/3)]= x[i];
    else
      mapU[(int)(lstid[i]/3)]= x[i];
  }

  return 0;
}

double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}
double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}
