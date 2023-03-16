/**
 * @file gap_filling.cpp
 * @brief Implementation of gap_filling routines
 * @author Simon Biquard
 * @date March 2023
 */

#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <complex>

#include "mappraiser/gap_filling.h"
#include "mappraiser/rng.h"
#include "mappraiser/noise_weighting.h"

void mappraiser::psd_from_tt(int fftlen,
                             int lambda,
                             int psdlen,
                             const double *tt,
                             double *psd,
                             double rate) {
    // Normalization
    double norm = rate * static_cast <double> (psdlen - 1);

    // FFTW variables
    double *circ_t;
    fftw_complex *cplx_psd;
    fftw_plan p;

    // Allocation
    circ_t = fftw_alloc_real(fftlen);
    cplx_psd = fftw_alloc_complex(psdlen);

    // FFTW plan
    p = fftw_plan_dft_r2c_1d(fftlen, circ_t, cplx_psd, FFTW_ESTIMATE);

    // Initialization of input
    std::copy(tt, (tt + lambda), circ_t);
    std::fill((circ_t + lambda), (circ_t + fftlen - lambda), 0);
    std::reverse_copy(tt, (tt + lambda), (circ_t + fftlen - lambda));

    // Execute FFT plan
    fftw_execute(p);

    // Compute the PSD values
    for (int i = 0; i < psdlen; ++i) {
        psd[i] = norm * std::abs(std::complex<double>(cplx_psd[i][0], cplx_psd[i][1]));
    }

    // Zero out DC value
    psd[0] = 0;

    // Free allocated memory
    fftw_free(circ_t);
    fftw_free(cplx_psd);
    fftw_destroy_plan(p);
}


double mappraiser::compute_mean(int samples, double *buf, bool subtract) {
    // Compute the DC level
    double DC = 0;
    for (int i = 0; i < samples; ++i) {
        DC += buf[i];
    }
    DC /= static_cast<double> (samples);

    // Remove it if needed
    if (subtract) {
        for (int i = 0; i < samples; ++i) {
            buf[i] -= DC;
        }
    }
    return DC;
}


double mappraiser::compute_variance(int samples, const double &mean, double *buf) {
    double var = 0;
    for (int i = 0; i < samples; ++i) {
        var += std::pow(buf[i] - mean, 2);
    }
    var /= static_cast<double> (samples - 1);
    return var;
}


void mappraiser::sim_noise_tod(int samples,
                               int lambda,
                               const double *tt,
                               double *buf,
                               double var_goal) {
    // Logical size of the fft
    // this could be modified to be a power of 2, for example
    int fftlen = samples;

    double *pdata;
    fftw_plan p;

    // Allocate the input/output buffer
    pdata = fftw_alloc_real(fftlen);

    // Create a plan for in-place half-complex -> real (HC2R) transform
    p = fftw_plan_r2r_1d(fftlen, pdata, pdata, FFTW_HC2R, FFTW_ESTIMATE);

    // Generate Re/Im gaussian randoms in a half-complex array
    // (see https://fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html)
    mappraiser::rng_dist_normal(samples, 0, 0, 0, 0, pdata);

    // Compute PSD values from the auto-correlation function
    int psdlen = (fftlen / 2) + 1;
    auto *psd = static_cast<double *>(std::malloc(sizeof(double) * psdlen));
    mappraiser::psd_from_tt(fftlen, lambda, psdlen, tt, psd);

    // Multiply by the PSD
    pdata[0] *= std::sqrt(psd[0]);
    for (int i = 1; i < (fftlen / 2); ++i) {
        double psdval = std::sqrt(psd[i]);
        pdata[i] *= psdval;
        pdata[fftlen - i] *= psdval;
    }
    pdata[fftlen / 2] *= std::sqrt(psd[psdlen - 1]);

    // Execute the FFT plan
    fftw_execute(p);

    // Backward FFT: 1/N factor not included by FFTW
    for (int i = 0; i < fftlen; ++i) {
        pdata[i] /= fftlen;
    }

    // Copy as many samples as we need into our noise vector
    std::copy(pdata, (pdata + samples), buf);

    // Zero out DC level
    double DC_subtracted = mappraiser::compute_mean(samples, buf, true);

    std::cout << "subtracted DC level = " << DC_subtracted << std::endl;

    // Normalize according to desired variance for the TOD
    // In theory this should not be necessary
    double rescale = std::sqrt(var_goal / mappraiser::compute_variance(samples, 0.0, buf));
    for (int i = 0; i < samples; ++i) {
        buf[i] *= rescale;
    }

    std::cout << "rescale factor (= sigma_goal / sigma) = " << rescale << std::endl;

    // Free allocated memory
    fftw_free(pdata);
    fftw_destroy_plan(p);
    std::free(psd);
}

void mappraiser::sim_constrained_noise_block(Tpltz *N_block,
                                             Tpltz *Nm1_block,
                                             const double *noise,
                                             Gap *gaps,
                                             double *constr_block) {
    // get the number of samples and the bandwidth
    const int samples = N_block->tpltzblocks[0].n;
    const int lambda = N_block->tpltzblocks[0].lambda;

    // copy the original noise vector and remove the mean
    auto *rhs = static_cast<double *>(std::malloc(sizeof(double) * samples));
    std::copy(noise, (noise + samples), rhs);
    double mean = mappraiser::compute_mean(samples, rhs, true);
    double var = mappraiser::compute_variance(samples, 0.0, rhs);

    // generate random noise realization "xi" with correlations
    auto *xi = static_cast<double *>(std::malloc(sizeof(double) * samples));
    mappraiser::sim_noise_tod(samples, lambda, N_block->tpltzblocks[0].T_block, xi, var);

    // rhs = noise - xi
    for (int i = 0; i < samples; ++i) {
        rhs[i] -= xi[i];
    }

    // invert the system N x = (noise - xi)
    apply_weights(Nm1_block, N_block, gaps, rhs);

    // compute the unconstrained realization
    std::copy(rhs, (rhs + samples), constr_block);
    stbmmProd(N_block, constr_block);

    for (int i = 0; i < samples; ++i) {
        constr_block[i] += xi[i] + mean; // add the mean back
    }

    // Free memory
    std::free(xi);
    std::free(rhs);
}

void mappraiser::sim_constrained_noise(Tpltz *N,
                                       Tpltz *Nm1,
                                       const double *noise,
                                       Gap *gaps,
                                       double *out_constrained) {
    // Loop through toeplitz blocks
    int t_id = 0;
    Tpltz N_block, Nm1_block;
    const double *noise_block;
    double *constrained_block;

    for (int i = 0; i < N->nb_blocks_loc; ++i) {
        // define single-block Tpltz structures
        set_tpltz_struct(&N_block, N, &N->tpltzblocks[i]);
        set_tpltz_struct(&Nm1_block, Nm1, &Nm1->tpltzblocks[i]);

        // pointer to current block in the tod
        noise_block = (noise + t_id);
        constrained_block = (out_constrained + t_id);

        // compute a constrained noise realization for the current block
        mappraiser::sim_constrained_noise_block(&N_block, &Nm1_block, noise_block,
                                                gaps, constrained_block);

        t_id += N->tpltzblocks[i].n;
    }
}

void sim_constrained_noise(Tpltz *N,
                           Tpltz *Nm1,
                           const double *noise,
                           Gap *gaps,
                           double *out_constrained) {
    mappraiser::sim_constrained_noise(N, Nm1, noise, gaps, out_constrained);
}
