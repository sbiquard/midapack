#ifndef MAPPRAISER_GAP_FILLING_H
#define MAPPRAISER_GAP_FILLING_H

#include <midapack.h>

#ifdef __cplusplus

#include <chrono>
#include <string>
#include <vector>

namespace mappraiser {

    class GapFillInfo {
    public:
        GapFillInfo(int, int);

        void   timer_start();
        void   timer_stop();
        double elapsed_seconds();

        double get_mean_iterations();
        double get_mean_seconds();

        void print_recap();
        void print_recap(const std::string &pref);

        void set_current_block(int i) { current_block = i; }

        void store_nbIterations(int n) { nbIterations[current_block] = n; }
        void store_time() { times[current_block] = elapsed_seconds(); }
        void store_validFrac(double f) { validFracs[current_block] = f; }

    private:
        int n_blocks;      // number of data blocks to treat
        int current_block; // current block index

        std::vector<int>    nbIterations; // number of PCG iterations for each block
        std::vector<double> times;        // total time for each block
        std::vector<double> validFracs;   // proportion of valid samples in each block

        // timepoints
        std::chrono::time_point<std::chrono::system_clock> _start;
        std::chrono::time_point<std::chrono::system_clock> _end;

    public:
        const int n_gaps; // number of local timestream gaps
    };

    void psd_from_tt(int fftlen, int lambda, int psdlen, const double *tt, std::vector<double> &psd,
                     double rate = 200.0);

    double compute_mean(int samples, double *buf, bool subtract);

    double compute_mean_good(int samples, double *buf, const bool *valid, bool subtract);

    double compute_variance(int samples, double mean, const double *buf);

    double compute_variance_good(int samples, double mean, const double *buf, const bool *valid);

    int find_valid_samples(Gap *gaps, size_t id0, std::vector<bool> &valid);

    void remove_baseline(std::vector<double> &buf, std::vector<double> &baseline, const std::vector<bool> &valid,
                         bool rm);

    void sim_noise_tod(int samples, int lambda, const double *tt, std::vector<double> &buf, u_int64_t realization,
                       u_int64_t detindx, u_int64_t obsindx, u_int64_t telescope, double var_goal = 1.0,
                       bool verbose = false);

    void sim_constrained_noise_block(GapFillInfo &gfi, Tpltz *N_block, Tpltz *Nm1_block, double *noise, Gap *gaps,
                                     u_int64_t realization, u_int64_t detindx, u_int64_t obsindx, u_int64_t telescope);

    void sim_constrained_noise(GapFillInfo &gfi, Tpltz *N, Tpltz *Nm1, double *noise, Gap *gaps, u_int64_t realization,
                               const u_int64_t *detindxs, const u_int64_t *obsindxs, const u_int64_t *telescopes);
} // namespace mappraiser
#endif

// Here the routine destined to be called by C code
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void perform_gap_filling(MPI_Comm comm, Tpltz *N, Tpltz *Nm1, double *noise, Gap *gaps, u_int64_t realization,
                         const u_int64_t *detindxs, const u_int64_t *obsindxs, const u_int64_t *telescopes,
                         bool verbose);

// test routine to call from python
void gap_filling(MPI_Comm comm, const int *data_size_proc, int nb_blocks_loc, int *local_blocks_sizes, int nnz,
                 double *tt, double *inv_tt, int lambda, double *noise, int *indices, u_int64_t realization,
                 const u_int64_t *detindxs, const u_int64_t *obsindxs, const u_int64_t *telescopes);

#ifdef __cplusplus
}
#endif

#endif // MAPPRAISER_GAP_FILLING_H
