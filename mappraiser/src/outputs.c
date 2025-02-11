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
                           const char *outpath, const char *ref) {
    // Number of pixels of the HEALPix map
    int npix = nside * nside * 12;
    o->nside = nside;

    // Initialize all maps to NULL
    o->mapI = NULL;
    o->mapQ = NULL;
    o->mapU = NULL;
    o->rcond = NULL;
    o->hits = NULL;
    o->precII = NULL;
    o->precIQ = NULL;
    o->precIU = NULL;
    o->precQQ = NULL;
    o->precUU = NULL;
    o->precQU = NULL;

    // Allocate memory for the maps
    int count = 0;
    if (nnz != 2) {
        // We have I
        o->mapI = SAFECALLOC(npix, sizeof *o->mapI);
        o->precII = SAFECALLOC(npix, sizeof *o->precII);
        count += 2;
    }
    if (nnz > 1) {
        // We have Q, U and Q x U
        o->mapQ = SAFECALLOC(npix, sizeof *o->mapQ);
        o->mapU = SAFECALLOC(npix, sizeof *o->mapU);
        o->precQQ = SAFECALLOC(npix, sizeof *o->precQQ);
        o->precUU = SAFECALLOC(npix, sizeof *o->precUU);
        o->precQU = SAFECALLOC(npix, sizeof *o->precQU);
        count += 5;
    }
    if (nnz == 3) {
        // We have I x {Q,U} cross-terms
        o->precIQ = SAFECALLOC(npix, sizeof *o->precIQ);
        o->precIU = SAFECALLOC(npix, sizeof *o->precIU);
        count += 2;
    }
    o->rcond = SAFECALLOC(npix, sizeof *o->rcond);
    o->hits = SAFECALLOC(npix, sizeof *o->hits);
    count += 2;

    // Define output file names
    const char *template_names[11] = {
        nnz != 2 ? "%s/mapI_%s.fits" : NULL,
        nnz > 1 ? "%s/mapQ_%s.fits" : NULL,
        nnz > 1 ? "%s/mapU_%s.fits" : NULL,
        nnz != 2 ? "%s/precII_%s.fits" : NULL,
        nnz == 3 ? "%s/precIQ_%s.fits" : NULL,
        nnz == 3 ? "%s/precIU_%s.fits" : NULL,
        nnz > 1 ? "%s/precQQ_%s.fits" : NULL,
        nnz > 1 ? "%s/precUU_%s.fits" : NULL,
        nnz > 1 ? "%s/precQU_%s.fits" : NULL,
        "%s/Cond_%s.fits",
        "%s/Hits_%s.fits",
    };
    o->max_count = sizeof(template_names) / sizeof(template_names[0]);
    o->real_count = count;
    o->files = SAFEMALLOC(sizeof *o->files * count);

    int c = 0;
    for (int i = 0; i < o->max_count; ++i) {
        if (template_names[i] == NULL)
            continue;
        o->files[c] = SAFEMALLOC(FILENAME_MAX * sizeof(char));
        sprintf(o->files[c], template_names[i], outpath, ref);
        ++c;
    }
    assert(c == o->real_count);
}

void freeMappraiserOutputs(MappraiserOutputs *o) {
    FREE(o->mapI);
    FREE(o->mapQ);
    FREE(o->mapU);
    FREE(o->precII);
    FREE(o->precIQ);
    FREE(o->precIU);
    FREE(o->precQQ);
    FREE(o->precQU);
    FREE(o->precUU);
    FREE(o->rcond);
    FREE(o->hits);
    for (int i = 0; i < o->real_count; ++i)
        FREE(o->files[i]);
    o->real_count = 0;
    FREE(o->files);
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

int clearFiles(MappraiserOutputs *o) {
    for (int i = 0; i < o->real_count; ++i) {
        if (access(o->files[i], F_OK) == -1)
            // file does not exist
            continue;
        int ret = remove(o->files[i]);
        if (ret != 0) {
            fprintf(stderr, "Error: unable to delete the file %s: %s\n",
                    o->files[i], strerror(errno));
            return 1;
        }
    }
    return 0;
}

void writeFiles(MappraiserOutputs *o) {
    char nest = 1;
    char *cordsys = "C";
    int c = 0;
    for (int i = 0; i < o->max_count; ++i) {
        void *map = NULL;
        int type = TDOUBLE;
        if (i == 0) {
            map = o->mapI;
        } else if (i == 1) {
            map = o->mapQ;
        } else if (i == 2) {
            map = o->mapU;
        } else if (i == 3) {
            map = o->precII;
        } else if (i == 4) {
            map = o->precIQ;
        } else if (i == 5) {
            map = o->precIU;
        } else if (i == 6) {
            map = o->precQQ;
        } else if (i == 7) {
            map = o->precUU;
        } else if (i == 8) {
            map = o->precQU;
        } else if (i == 9) {
            map = o->rcond;
        } else if (i == 10) {
            map = o->hits;
            type = TINT;
        }
        if (map == NULL)
            continue;
        write_map(map, type, o->nside, o->files[c], nest, cordsys);
        ++c;
    }
    assert(c == o->real_count);
}
