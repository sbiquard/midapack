#include <assert.h>
#include <errno.h>
#include <fitsio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mappraiser/iofiles.h>
#include <mappraiser/outputs.h>
#include <memutils.h>

void initMappraiserOutputs(MappraiserOutputs *o, int nside, int nnz,
                           bool mirror) {
    // Do we have mirror pixels?
    o->mirror = mirror;

    // Number of pixels of the HEALPix map
    o->nside = nside;
    int npix = nside * nside * 12;
    int size = npix * (o->mirror ? 2 : 1);

    // Initialize all maps to NULL
    o->hits = NULL;
    o->mapI = NULL;
    o->mapQ = NULL;
    o->mapU = NULL;
    o->rcond = NULL;
    o->precII = NULL;
    o->precIQ = NULL;
    o->precIU = NULL;
    o->precQQ = NULL;
    o->precUU = NULL;
    o->precQU = NULL;

    // Allocate memory for the maps
    if (nnz != 2) {
        // We have I
        o->mapI = SAFECALLOC(size, sizeof *o->mapI);
        o->precII = SAFECALLOC(size, sizeof *o->precII);
    }
    if (nnz > 1) {
        // We have Q, U and Q x U
        o->mapQ = SAFECALLOC(size, sizeof *o->mapQ);
        o->mapU = SAFECALLOC(size, sizeof *o->mapU);
        o->precQQ = SAFECALLOC(size, sizeof *o->precQQ);
        o->precUU = SAFECALLOC(size, sizeof *o->precUU);
        o->precQU = SAFECALLOC(size, sizeof *o->precQU);
    }
    if (nnz == 3) {
        // We have I x {Q,U} cross-terms
        o->precIQ = SAFECALLOC(size, sizeof *o->precIQ);
        o->precIU = SAFECALLOC(size, sizeof *o->precIU);
    }
    o->rcond = SAFECALLOC(size, sizeof *o->rcond);
    o->hits = SAFECALLOC(size, sizeof *o->hits);
}

void freeMappraiserOutputs(MappraiserOutputs *o) {
    FREE(o->mapI);
    FREE(o->mapQ);
    FREE(o->mapU);
    FREE(o->rcond);

    FREE(o->precII);
    FREE(o->precIQ);
    FREE(o->precIU);
    FREE(o->precQQ);
    FREE(o->precUU);
    FREE(o->precQU);

    FREE(o->hits);
}

void populateMappraiserOutputs(MappraiserOutputs *o, const double *x,
                               const int *lstid, const double *rcond,
                               const int *lhits, const double *bj_map,
                               int xsize, int nnz) {
    for (int i = 0; i < xsize / nnz; i++) {
        int hpi = lstid[i * nnz] / nnz; // HEALPix index
        switch (nnz) {
        case 1: // I map
            o->hits[hpi] = lhits[i];
            o->rcond[hpi] = rcond[i];
            o->mapI[hpi] = x[i * nnz];
            o->precII[hpi] = bj_map[i * nnz * nnz];
            break;
        case 2: // Q and U maps
            o->hits[hpi] = lhits[i];
            o->rcond[hpi] = rcond[i];
            o->mapQ[hpi] = x[i * nnz];
            o->mapU[hpi] = x[i * nnz + 1];
            o->precQQ[hpi] = bj_map[i * nnz * nnz];
            o->precQU[hpi] = bj_map[i * nnz * nnz + 1];
            o->precUU[hpi] = bj_map[i * nnz * nnz + 3];
            break;
        case 3: // I, Q and U maps
            o->hits[hpi] = lhits[i];
            o->rcond[hpi] = rcond[i];
            o->mapI[hpi] = x[i * nnz];
            o->mapQ[hpi] = x[i * nnz + 1];
            o->mapU[hpi] = x[i * nnz + 2];
            o->precII[hpi] = bj_map[i * nnz * nnz];
            o->precIQ[hpi] = bj_map[i * nnz * nnz + 1];
            o->precIU[hpi] = bj_map[i * nnz * nnz + 2];
            o->precQQ[hpi] = bj_map[i * nnz * nnz + 4];
            o->precQU[hpi] = bj_map[i * nnz * nnz + 5];
            o->precUU[hpi] = bj_map[i * nnz * nnz + 8];
            break;
        default:
            fprintf(stderr, "Error: unsupported nnz: %d\n", nnz);
            exit(1);
        }
    }
}

int check_and_write(void *content, int type, int nside, const char *fname) {
    // check access to the file
    if (access(fname, F_OK) == 0) {
        fprintf(stderr,
                "Warning: file %s already exists and will be overwritten.\n",
                fname);
        if (remove(fname) != 0) {
            fprintf(stderr, "Error removing file %s: %s\n", fname,
                    strerror(errno));
            return 1;
        }
    } else if (errno != ENOENT) {
        // ENOENT means the file does not exist, which is fine
        fprintf(stderr, "Error checking file %s: %s\n", fname, strerror(errno));
        return 1;
    }

    // write the map
    write_map(content, type, nside, fname, NEST, COORDSYS);
    return 0;
}

int writeFiles(MappraiserOutputs *o, const char *outpath, const char *ref) {
    int info = 0;
    int npix = o->nside * o->nside * 12;

    // First, handle hits map separately
    if (o->hits != NULL) {
        char fname[FILENAME_MAX];
        snprintf(fname, FILENAME_MAX, "%s/Hits_%s", outpath, ref);

        if (o->mirror) {
            char fname_mirror[FILENAME_MAX];
            strcpy(fname_mirror, fname);
            strcat(fname_mirror, "_mirror.fits");
            // write the second part of the map
            check_and_write(o->hits + npix, TINT, o->nside, fname_mirror);
        }
        strcat(fname, ".fits");
        check_and_write(o->hits, TINT, o->nside, fname);
    }

    // Handle all double maps
    for (int i = 0; i < NUM_MAPS; ++i) {
        char fname[FILENAME_MAX];
        double *map = NULL;

        if (i == 0) {
            map = o->mapI;
            snprintf(fname, FILENAME_MAX, "%s/mapI_%s", outpath, ref);
        } else if (i == 1) {
            map = o->mapQ;
            snprintf(fname, FILENAME_MAX, "%s/mapQ_%s", outpath, ref);
        } else if (i == 2) {
            map = o->mapU;
            snprintf(fname, FILENAME_MAX, "%s/mapU_%s", outpath, ref);
        } else if (i == 3) {
            map = o->rcond;
            snprintf(fname, FILENAME_MAX, "%s/Cond_%s", outpath, ref);
        } else if (i == 4) {
            map = o->precII;
            snprintf(fname, FILENAME_MAX, "%s/precII_%s", outpath, ref);
        } else if (i == 5) {
            map = o->precIQ;
            snprintf(fname, FILENAME_MAX, "%s/precIQ_%s", outpath, ref);
        } else if (i == 6) {
            map = o->precIU;
            snprintf(fname, FILENAME_MAX, "%s/precIU_%s", outpath, ref);
        } else if (i == 7) {
            map = o->precQQ;
            snprintf(fname, FILENAME_MAX, "%s/precQQ_%s", outpath, ref);
        } else if (i == 8) {
            map = o->precUU;
            snprintf(fname, FILENAME_MAX, "%s/precUU_%s", outpath, ref);
        } else if (i == 9) {
            map = o->precQU;
            snprintf(fname, FILENAME_MAX, "%s/precQU_%s", outpath, ref);
        }

        if (map == NULL)
            continue;

        if (o->mirror) {
            char fname_mirror[FILENAME_MAX];
            strcpy(fname_mirror, fname);
            strcat(fname_mirror, "_mirror.fits");
            // write the second part of the map
            check_and_write(map + npix, TDOUBLE, o->nside, fname_mirror);
        }
        strcat(fname, ".fits");
        check_and_write(map, TDOUBLE, o->nside, fname);
    }
    return info;
}
