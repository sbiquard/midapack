 /** @file s2hat_tools.h
    @brief <b> Declaration of the backbone routines of the Wiener-filtering extension of MAPPRAISER.</b>
    @author Magdy Morshed
    @date November 2022 */
 
 /* Full documentation for S2HAT here : https://apc.u-paris.fr/APC_CS/Recherche/Adamis/MIDAS09/software/s2hat/s2hat/docs/S2HATdocs.html */ 
    
// #include <mpi.h>
// #include "s2hat.h"
// #include "midapack.h"


#ifndef DBL_MAX
#define DBL_MAX            1.79769313486231470e+308
#endif

// typedef struct S2HAT_GLOBAL_parameters S2HAT_GLOBAL_parameters;
typedef struct S2HAT_GLOBAL_parameters{
    /* Global parameters of S2HAT, to give to all processors */
    s2hat_pixeltype pixelization_scheme;
    s2hat_scandef scan_sky_structure_pixel;
    s2hat_pixparameters pixpar;

    int nside;
    int nlmax;
    int nmmax;
} S2HAT_GLOBAL_parameters;

// typedef struct S2HAT_LOCAL_parameters S2HAT_LOCAL_parameters;
typedef struct S2HAT_LOCAL_parameters{
    /* Local parameters of S2HAT, dependent on each processor */

    int gangrank;
    int gangsize;
    int gangroot;
    MPI_Comm gangcomm;

    int nmvals;
    int first_ring;
    int last_ring;
    int map_size;
    int* mvals; // size given by nmvals

    // For butterfly communication scheme -- in RING scheme
    int first_pixel_number;
    int last_pixel_number;
    long int *pixel_numbered_ring; // Pointer to the ordered pixel in RING scheme which the process will consider for S2HAT operations


    // Tools to precompute Legendre functions, but not taken into account by s2hat if plms=0, which is the default behaviour we choose
    int plms;
    long int nplm;
} S2HAT_LOCAL_parameters;


typedef enum { false, true } bool;

typedef struct Files_path_WIENER_FILTER{
    /* Global parameters of S2HAT, to give to all processors */
    // bool use_mask_file; // Boolean  to determine if a mask is used or not ; if not, maskfile_path will not be considered
    // char *maskfile_path; // Path to mask file, of dimensions [12*nside**2, 3] in fits format (write_maps of Healpy can be used)
    
    int lmax_Wiener_Filter; // lmax which will be considered in the application of Wiener filter
    int nside; // Nside in order to read the maps
    char *c_ell_path; // Path for c_ells fits file, to construct CMB sky covariance matrix, in the form of a 1d vector of dimension [lmax,number_correlations] in column-wise ordering
    int number_correlations; // Number of correlations included in c_ell fits file, can be eitehr 4, TT, EE, BB and TE, or 6, TT, EE, BB, TE, TB and EB (in this order)
} Files_path_WIENER_FILTER;


typedef struct S2HAT_parameters{
    S2HAT_GLOBAL_parameters *Global_param_s2hat;
    S2HAT_LOCAL_parameters *Local_param_s2hat;
    Files_path_WIENER_FILTER *Files_WF_struct;

    int size_alm; // Size of the alm ararys : lmax*mmax
    int nstokes; // Number of Stokes parameters : either 1 for intensity only, 2 for polarization only, 3 for intensity and polarization
} S2HAT_parameters;


/* Create a mpi communicator subset of the initial global communicator, by taking the number_ranks_to_divide first ranks within it*/
int mpi_create_subset(int number_ranks_to_divive, MPI_Comm initcomm, MPI_Comm *subset_comm);



/* Get global s2hat structures which must be distributed to all processors*/
int get_main_s2hat_global_parameters(int nside, char *maskfile_path, s2hat_pixeltype *pixelization_scheme, s2hat_scandef *scan_sky_structure_pixel, s2hat_pixparameters *pixpar, bool use_mask_file);

/* Create wrapper structure s2hat of local parameters of s2hat, which will differ for all processors */
int init_s2hat_global_parameters(Files_path_WIENER_FILTER Files_WF_struct, int nside, int lmax, S2HAT_GLOBAL_parameters *Global_param_s2hat);

/* Initialize MPI parameters of local parameters wrapper structure of s2hat, which will differ for all processors */
int init_MPI_struct_s2hat_local_parameters(S2HAT_LOCAL_parameters *Local_param_s2hat, int number_ranks_s2hat, MPI_Comm initcomm);

/* Create wrapper structure of local parameters wrapper structure of s2hat, which will differ for all processors, and assuming MPI structure already assigned */
int init_s2hat_local_parameters_struct(S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters *Local_param_s2hat);

/* Initaization of superctrure S2HAT_parameters */
int init_s2hat_parameters_superstruct(Files_path_WIENER_FILTER *Files_WF_struct, S2HAT_parameters *S2HAT_params, int root);

/* Use s2hat routines to broadcast s2hat global structures */
void mpi_broadcast_s2hat_global_struc(S2HAT_GLOBAL_parameters *Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat);

/* Distribute full sky map in ring ordering, with convention [npix, nstokes] in column-wise order among procs, into local maps */
int distribute_full_sky_map_into_local_maps_S2HAT(double* full_sky_map, double *local_map_s2hat, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat, int nstokes);

/* Collect submap from local_maps of S2HAT, given first and last pixel of submap */
int collect_partial_map_from_pixels(double* local_map_s2hat, double *output_submap, int first_pix, int last_pix, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat, int nstokes);

/* Free covariance matrix */
void free_covariance_matrix(double ** covariance_matrix_3x3, int lmax);

/* Free wrapper structures of s2hat */
void free_s2hat_GLOBAL_parameters_struct(S2HAT_GLOBAL_parameters *Global_param_s2hat);

/* Free wrapper structures of s2hat */
void free_s2hat_LOCAL_parameters_struct(S2HAT_LOCAL_parameters *Local_param_s2hat);

/* Free superstructure around S2HAT */
void free_s2hat_parameters_struct(S2HAT_parameters *S2HAT_params);

/* Change conventions between ring and nest distribution of maps */
int convert_indices_nest2ring(double *indices_nest, double *indices_ring, long int number_of_indices);
int convert_indices_ring2nest(double *indices_ring, double *indices_nest, long int number_of_indices);

/* Change conventions between nest and ring distribution of maps */
void convert_full_map_ring2nest(double *map_ring, double *map_nest, int nside, int nstokes, int npix);
void convert_full_map_nest2ring(double *map_nest, double *map_ring, int nside, int nstokes, int npix);

/* Define file support structure for Wiener_filter extension */
void init_files_struct_WF(Files_path_WIENER_FILTER *Files_path_WF_struct, char *path_mask_file,  bool use_mask_file, int nside, int lmax_Wiener_Filter, char *c_ell_path, int number_correlations);

/* Function to read file corresponding to the mask */
void read_fits_mask(int nside, double *mask, char *path_mask_file, int col);

/* Function to read TQU maps */
void read_TQU_maps( int nside, double *map, char *infile, int nstokes);

/* Function to transform the mask into binary (composed of 0 and 1 on pixel sky)*/
void make_mask_binary(double* mask, int* mask_binary, int *f_sky, long npix);

/* Obtain c_ell array from c_ell path */
void read_fits_cells(int lmax, int number_correl, double *c_ell_array, char *path_list_file, int col);

/* Transform alm coefficients local_alm into a pixel map local_map_pix */
int apply_alm2pix(s2hat_dcomplex *local_alm, double *local_map_pix, S2HAT_parameters *S2HAT_params);

/* Transform local pixel map into local alm coefficients */
int apply_pix2alm(double *local_map_pix, s2hat_dcomplex *local_alm, S2HAT_parameters *S2HAT_params);

/* Gather all local_map to obtain a full_sky_map */
// int gather_map(double *local_map_pix, double *full_sky_map, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat);

/* Apply inverse of covariance matrix to local_alm */
int apply_inv_covariance_matrix_to_alm(s2hat_dcomplex *input_local_alm, s2hat_dcomplex *out_local_alm, double **inv_covariance_matrix, int nstokes, S2HAT_parameters *S2HAT_params);





/* Transform alm to c_ell coefficients */
int alm2cls(s2hat_dcomplex* local_alm, double *c_ell_array, int nspec, int nstokes, S2HAT_parameters *S2HAT_params);

/* General function to inverse matrix using LAPACK */
int get_inverse_matrix(int order_matrix, double* matrix_to_be_inverted);

/* Read c_ell to generate covariance matrix which will be in the form : covariance_matrix_3x3[lmax][number_correlations] with number_correlations being either 9 for [TT, TE, TB, ET, EE, EB, BT, BE, BB] (with TE=ET, TB=BT and BE=EB), 4 for 9 for [EE, EB, BE, BB] (with BE=EB) or 1 for [TT] */
int get_covariance_matrix_NxN(char *c_ell_path, int number_correl, int nstokes, double **covariance_matrix_NxN, S2HAT_GLOBAL_parameters Global_param_s2hat);

/* Function to obtain inverse of covariance matrix in harmonic domain, from given c_ells */
int get_inverse_covariance_matrix_NxN(int nstokes, S2HAT_parameters *S2HAT_params, double **inverse_covariance_matrix);


/* tmp functions for communication */
// int all_reduce_to_single_map_mappraiser(Mat *A, double* x, int nside, double* out_val, int root);

// int distribute_map_S2HAT_ordering(double* full_sky_map, double *local_map_s2hat, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat);

// int brute_force_transfer_local_maps(Mat *A, double* local_pixel_map_MAPPRAISER, double *local_pixel_map_s2hat, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat);
