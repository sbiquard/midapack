/**
 * @file mapping.c
 * @brief Implementation of routines for pixel to time-domain mapping and gap
 * measurement
 * @author Simon Biquard
 * @date Nov 2022
 */

#include "mapmat/mapmat.h"
#include <stdbool.h>

#ifndef NDEBUG
#include <assert.h>
#endif

#include <mappraiser/mapping.h>
#include <memutils.h>

void print_gap_stgy(GapStrategy gs) {
    switch (gs) {
    case COND:
        puts("conditioning");
        break;
    case MARG_LOCAL_SCAN:
        puts("marginalization (1 extra pixel/scan)");
        break;
    case NESTED_PCG:
        puts("nested PCG");
        break;
    case NESTED_PCG_NO_GAPS:
        puts("nested PCG ignoring gaps");
        break;
    case MARG_PROC:
        puts("marginalization (1 extra pixel/proc)");
        break;
    }
}

int get_actual_map_size(const Mat *A) {
    if (A->flag_ignore_extra)
        return A->lcount - A->trash_pix * A->nnz;
    else
        return A->lcount;
}

int get_valid_map_size(const Mat *A) {
    return A->lcount - A->trash_pix * A->nnz;
}

int create_extra_pix(int *indices, double *weights, int nnz, int nb_blocks_loc,
                     const int *local_blocks_sizes, GapStrategy gs) {
    bool extra_pixels_polarized = false;

    switch (gs) {
    case MARG_LOCAL_SCAN: {
        int offset = 0; // position in the data
        int lsize;      // size of data block

        // loop through local blocks
        for (int i = 0; i < nb_blocks_loc; ++i) {
            lsize = local_blocks_sizes[i];
            for (int j = offset; j < offset + lsize; j++) {
                int jnnz = j * nnz;
                if (indices[jnnz] < 0) {
                    // set a negative index corresponding to local scan number
                    // don't forget the nnz multiplicity of the indices
                    for (int k = 0; k < nnz; ++k) {
                        indices[jnnz + k] = -(i + 1) * nnz + k;
                        // indices[jnnz + k] = -(nb_blocks_loc - i) * nnz + k;

                        if (!extra_pixels_polarized) {
                            // make the extra pixels not polarized
                            if (k > 0) {
                                weights[jnnz + k] = 0.0;
                            }
                        }
                    }
                }
            }
            offset += lsize;
        }
        break;
    }

    case MARG_PROC: {
        int offset = 0; // position in the data
        int lsize;      // size of data block

        // loop through local blocks
        for (int i = 0; i < nb_blocks_loc; ++i) {
            lsize = local_blocks_sizes[i];
            for (int j = offset; j < offset + lsize; j++) {
                int jnnz = j * nnz;
                if (indices[jnnz] < 0) {
                    for (int k = 0; k < nnz; ++k) {
                        // no dependence on the block index
                        indices[jnnz + k] = -nnz + k;

                        if (!extra_pixels_polarized) {
                            // make the extra pixels not polarized
                            if (k > 0) {
                                weights[jnnz + k] = 0.0;
                            }
                        }
                    }
                }
            }
            offset += lsize;
        }
        break;
    }

    default:
        /* nothing to do */
        break;
    }
    return 0;
}

/**
 * @brief Build the pixel-to-time-domain mapping, i.e.
 * i) A->id_last_pix which contains the indexes of the last samples pointing to
 * each pixel ii) A->ll which is a linked list of time sample indexes
 *
 * @param A the pointing matrix structure
 * @return int the number of timestream gaps found
 */
int build_pixel_to_time_domain_mapping(Mat *A) {
    int i, j;
    int ipix;
    int ngap, lengap;

    // index of last sample pointing to each pixel
    A->id_last_pix = SAFEMALLOC(sizeof A->id_last_pix * A->lcount / A->nnz);

    // linked list of time samples indexes
    A->ll = SAFEMALLOC(sizeof A->ll * A->m);

    // initialize the mapping arrays to -1
    for (i = 0; i < A->m; i++) {
        A->ll[i] = -1;
    }
    for (j = 0; j < A->lcount / A->nnz; j++) {
        A->id_last_pix[j] = -1;
    }

    // build the linked list chain of time samples corresponding to each pixel
    // and compute number of timestream gaps
    ngap = 0;
    lengap = 0;
    for (i = 0; i < A->m; i++) {
        ipix = A->indices[i * A->nnz] / A->nnz;
        if (A->id_last_pix[ipix] == -1) {
            A->id_last_pix[ipix] = i;
        } else {
            A->ll[i] = A->id_last_pix[ipix];
            A->id_last_pix[ipix] = i;
        }

        if (A->trash_pix == 0) {
            // skip the computation of ngap
            continue;
        }

        // compute the number of gaps in the timestream
        if (A->indices[i * A->nnz] >= A->trash_pix * A->nnz) {
            // valid sample: reset gap length
            lengap = 0;
        } else {
            // flagged sample -> gap
            if (lengap == 0) {
                // new gap: increment gap count
                ++ngap;
            }

            // increment current gap size
            ++lengap;
        }
    }
    return ngap;
}

int argmax(const int *array, int size) {
    int max = array[0];
    int argmax = 0;
    for (int i = 1; i < size; ++i) {
        if (array[i] > max) {
            max = array[i];
            argmax = i;
        }
    }
    return argmax;
}

/**
 * @brief Build the gap structure for the local samples.
 * @param gif global row index offset of the local data
 * @param gaps Gap structure (Gaps->ngap must already be computed!)
 * @param A pointing matrix structure
 */
void build_gap_struct(int64_t gif, Gap *gaps, Mat *A) {
    // allocate the arrays
    gaps->id0gap = SAFEMALLOC(sizeof gaps->id0gap * gaps->ngap);
    gaps->lgap = SAFEMALLOC(sizeof gaps->lgap * gaps->ngap);

    if (gaps->ngap == 0)
        return;

    // follow linked time samples for all extra pixels simultaneously
    int *tab_j = SAFEMALLOC(sizeof tab_j * A->trash_pix);

    // initialize with the last sample pointing to each extra pixel
    for (int p = 0; p < A->trash_pix; p++) {
        tab_j[p] = A->id_last_pix[p];
    }

    // current index in the tab_j array
    int pj = argmax(tab_j, A->trash_pix);

    int i = gaps->ngap - 1; // index of the gap being computed
    int lengap = 1;         // length of the current gap
    int j = tab_j[pj];      // index to go through linked time samples
    int gap_start = j;      // index of the first sample of the gap

    // go through the time samples
    while (j != -1) {
        // go to previous flagged sample
        tab_j[pj] = A->ll[tab_j[pj]];
        pj = argmax(tab_j, A->trash_pix);
        j = tab_j[pj];

        if (j != -1 && gap_start - j == 1) {
            // same gap, and there are flagged samples left
            ++lengap;
        } else {
            // different gap, or no flagged samples remaining
            gaps->id0gap[i] = gif + gap_start; // global row index
            gaps->lgap[i] = lengap;
            lengap = 1;
            --i;
        }
        gap_start = j;
    }

    FREE(tab_j);
}

bool gap_overlaps_with_block(Gap *gaps, int i_gap, Block *block) {
    if (i_gap < 0 || i_gap > gaps->ngap - 1)
        return false;
    int64_t id0g = gaps->id0gap[i_gap];
    int64_t idv = block->idv;
    int lg = gaps->lgap[i_gap];
    int n = block->n;
    return (idv < id0g + lg) && (id0g < idv + n);
}

void compute_gaps_per_block(Gap *gaps, int nb_blocks, Block *blocks) {
    // number of local gaps
    int ng = gaps->ngap;

    // initialize everything to -1
    for (int i = 0; i < nb_blocks; ++i) {
        blocks[i].first_gap = -1;
        blocks[i].last_gap = -1;
    }

    if (ng == 0) {
        // no local gaps, nothing to do
        return;
    }

    int ig;           // gap index
    int ig_reset = 0; // index of last gap matched with a block

    for (int i = 0; i < nb_blocks; ++i) {
        // reset gap index
        ig = ig_reset;

        // current block
        Block *b = &(blocks[i]);

        // find the first relevant gap
        while (ig < ng && !gap_overlaps_with_block(gaps, ig, b)) {
            ++ig;
        }

        if (ig == ng) {
            // no relevant gaps found for this block
            // leave first_gap and last_gap fields untouched
            continue;
        }

        // store the first relevant gap
        b->first_gap = ig;

        // go through next gaps
        while (gap_overlaps_with_block(gaps, ig + 1, b)) {
            ++ig;
        }

        // store the index of the last relevant gap
        b->last_gap = ig;
        ig_reset = ig;
    }
}

void copy_gap_info(int nb_blocks, Block *src, Block *dest) {
    for (int i = 0; i < nb_blocks; ++i) {
        dest[i].first_gap = src[i].first_gap;
        dest[i].last_gap = src[i].last_gap;
    }
}

int compute_global_gap_count(MPI_Comm comm, Gap *gaps) {
    int gap_count = gaps->ngap;
    MPI_Allreduce(MPI_IN_PLACE, &gap_count, 1, MPI_INT, MPI_SUM, comm);
    return gap_count;
}

void fill_gap_with_zero(double *tod, int n, int64_t idv, int64_t id0g, int lg) {
#ifndef NDEBUG
    // assert that gap is relevant for the given data block
    assert(idv < id0g + lg && id0g < idv + n);
#endif
    // set intersection of tod and gap to zero
    for (int64_t j = id0g; j < id0g + lg; ++j) {
        if (idv <= j && j < n + idv)
            tod[j - idv] = 0;
    }
}

/**
 * Fill all timestream gaps of a vector with zeros. Warning! This routine
 * assumes that relevant gaps have been determined for each data block (e.g.
 * through a call to compute_gaps_per_block).
 * @param tod pointer to the data vector
 * @param tmat pointer to a Tpltz matrix containing information about the data
 * blocks
 * @param gaps pointer to the gaps structure
 */
void reset_relevant_gaps(double *tod, Tpltz *tmat, Gap *gaps) {
    // loop over data blocks
    Block *b;
    double *tod_block;
    int pos = 0;
    for (int i = 0; i < tmat->nb_blocks_loc; ++i) {
        b = &(tmat->tpltzblocks[i]);
        // if there are no relevant gaps, skip this block
        if (b->first_gap < 0) {
            continue;
        }

        tod_block = (tod + pos);
        // loop over the relevant gaps for this block
        for (int j = b->first_gap; j <= b->last_gap; ++j) {
            fill_gap_with_zero(tod_block, b->n, b->idv, gaps->id0gap[j],
                               gaps->lgap[j]);
        }
        pos += b->n;
    }
}

void condition_extra_pix_zero(Mat *A) {
    // number of extra pixels in the pointing matrix
    int extra = A->trash_pix * A->nnz;
    int nnz = A->nnz;

    // if no extra pixels, there is nothing to do
    if (extra == 0)
        return;

    // set pointing weights for extra pixels to zero
    for (int i = 0; i < A->m * nnz; i += nnz) {
        if (A->indices[i] < extra) {
            for (int j = 0; j < nnz; ++j) {
                A->values[i + j] = 0;
            }
        }
    }
}

void point_pixel_to_trash(Mat *A, int ipix) {
    // last index of time sample pointing to pixel
    int j = A->id_last_pix[ipix];
    int nnz = A->nnz;

    while (j != -1) {
        // point sample to trash pixel
        for (int k = 0; k < nnz; k++) {
            A->indices[j * nnz + k] = k - nnz;
        }
        j = A->ll[j];
    }
}

void get_pixshare_pond(Mat *A, double *pixpond) {
    // number of local pixels
    int n = get_actual_map_size(A);
    int n_extra = n - get_valid_map_size(A);

    // create an eyes local vector
    for (int i = 0; i < n; i++)
        pixpond[i] = 1.;

    // communicate with the others processes to have the global reduce
    // only communicate shared pixels (i.e. valid)
    greedyreduce(A, pixpond + n_extra);

    // compute the inverse vector
    for (int i = n_extra; i < n; i++)
        pixpond[i] = 1. / pixpond[i];
}

__attribute__((unused)) void print_gap_info(Gap *gaps) {
    printf("Local Gap structure\n");
    printf("  ngap   = %d\n", gaps->ngap);
    printf("  lgap   = ");
    int n = gaps->ngap;
    printf("[");
    for (int i = 0; i < n; ++i) {
        printf((i + 1 < n) ? "%d, " : "%d]", gaps->lgap[i]);
    }
    printf("\n");
    printf("  id0gap = ");
    printf("[");
    for (int i = 0; i < n; ++i) {
        printf((i + 1 < n) ? "%ld, " : "%ld]", gaps->id0gap[i]);
    }
    printf("\n");
}
