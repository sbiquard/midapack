
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
// #include <mkl.h>
// #include "fitsio.h"
#include <unistd.h>
#include "s2hat_tools.h"




int apply_alm2pix(s2hat_dcomplex *local_alm, double *local_map_pix, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat){
    /* Transform alm coefficients local_alm into a pixel map local_map_pix, 
    all details here : https://apc.u-paris.fr/APC_CS/Recherche/Adamis/MIDAS09/software/s2hat/s2hat/docs/Cmanual/Calm2map.html 

    local_alm is a 4-dimensional array in the form :
        (1:nstokes,0:nlmax,0:nmvals-1,1:nmaps), if lda == nstokes;      (HEALpix convention)
        (0:nlmax,0:nmvals-1,1:nstokes,1:nmaps), if lda == nlmax;      (S2HAT convention)
    in the form I, E, B
    */

    int nmaps = 1; // We only provide 1 input set of alm coefficient
    int nstokes = 3; // We want all T, Q and U maps
    int lda = nstokes; // We choose the HEALPIX convention with local_alm in the form (1:nstokes,0:nlmax,0:nmvals-1,1:nmaps)

    s2hat_alm2map(Local_param_s2hat.plms, Global_param_s2hat.pixelization_scheme, Global_param_s2hat.scan_sky_structure_pixel, Global_param_s2hat.nlmax, Global_param_s2hat.nmmax, 
        Local_param_s2hat.nmvals, Local_param_s2hat.mvals, nmaps, nstokes, 
	    Local_param_s2hat.first_ring, Local_param_s2hat.last_ring, Local_param_s2hat.map_size, local_map_pix, lda, 
        local_alm, Local_param_s2hat.nplm, NULL, 
		Local_param_s2hat.gangsize, Local_param_s2hat.gangrank, Local_param_s2hat.gangcomm);
    // The NULL argument correspond to precomputed Legendre polynomials, only relevant if plms != 0
    return 0;
}





int apply_pix2alm(double *local_map_pix, s2hat_dcomplex *local_alm, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat){
    /* Transform pixel map into alm coefficients, all details here : https://apc.u-paris.fr/APC_CS/Recherche/Adamis/MIDAS09/software/s2hat/s2hat/docs/Cmanual/Cmap2alm.html 
     local_alm is a 4-dimensional array in the form :
        (1:nstokes,0:nlmax,0:nmvals-1,1:nmaps), if lda == nstokes;      (HEALpix convention)
        (0:nlmax,0:nmvals-1,1:nstokes,1:nmaps), if lda == nlmax;      (S2HAT convention)
    Here the HEALpix convention has been chosen
    Output will be in the form I, E, B
    */

    int nmaps = 1; // We only provide 1 input set of alm coefficient
    int nstokes = 3; // We provide all 3 T, Q and U maps
    int lda = nstokes; // We choose the HEALPIX convention with local_alm in the form (1:nstokes,0:nlmax,0:nmvals-1,1:nmaps)

    double *local_w8ring;
    int i_ring;


    // local_w8ring=(double *)calloc( nstokes*(Local_param_s2hat.last_ring-Local_param_s2hat.first_ring+1), sizeof(double));
    local_w8ring = (double *) malloc( nstokes*(Local_param_s2hat.last_ring-Local_param_s2hat.first_ring+1)*sizeof(double));
    for( i_ring=0; i_ring< nstokes*(Local_param_s2hat.last_ring-Local_param_s2hat.first_ring+1); i_ring++) {
            local_w8ring[i_ring]=1.;
        }
    
    
    s2hat_map2alm(Local_param_s2hat.plms, Global_param_s2hat.pixelization_scheme, Global_param_s2hat.scan_sky_structure_pixel, Global_param_s2hat.nlmax, Global_param_s2hat.nmmax, 
            Local_param_s2hat.nmvals, Local_param_s2hat.mvals, nmaps, nstokes, 
            Local_param_s2hat.first_ring, Local_param_s2hat.last_ring, local_w8ring, Local_param_s2hat.map_size, local_map_pix, lda, local_alm, 
            Local_param_s2hat.nplm, NULL,
            Local_param_s2hat.gangsize, Local_param_s2hat.gangrank, Local_param_s2hat.gangcomm);
        // The NULL argument correspond to precomputed Legendre polynomials, only relevant if plms != 0

    free(local_w8ring);

    return 0;
}

int apply_inv_covariance_matrix_to_alm(s2hat_dcomplex *local_alm, double **inv_covariance_matrix, S2HAT_GLOBAL_parameters Global_param_s2hat, S2HAT_LOCAL_parameters Local_param_s2hat){
    /* Apply inverse of covariance matrix to local_alm */

    // distribute_alms to apply to covariance matrix ?

    int ell_value, m_value, nstokes, line_index;
    int lmax = Global_param_s2hat.nlmax;

    int nmvals = Local_param_s2hat.nmvals; // Total number of m values
    // int *mvals = Local_param_s2hat->mvals; // Values of m the considered processor contain

    double res_real, res_imag;

    for(ell_value=0; ell_value < lmax+1; ell_value++){
        for(m_value=0; m_value < nmvals; m_value++){
            for (nstokes=0; nstokes<3; nstokes++){
                res_real, res_imag = 0;
                for (line_index=0; line_index < 3; line_index++){
                    res_real += local_alm[line_index*(lmax+1)*nmvals + ell_value*nmvals + m_value].re * inv_covariance_matrix[ell_value][nstokes*3 + line_index];
                    res_imag += local_alm[line_index*(lmax+1)*nmvals + ell_value*nmvals + m_value].im * inv_covariance_matrix[ell_value][nstokes*3 + line_index];
                }
                local_alm[nstokes*(lmax+1)*nmvals + ell_value*nmvals + m_value].re = res_real;
                local_alm[nstokes*(lmax+1)*nmvals + ell_value*nmvals + m_value].im = res_imag;
            }
            // Verify it is not applied to part where a_lm not defined !!!
        }
    }
}

