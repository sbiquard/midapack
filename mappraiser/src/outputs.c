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

    // Allocate memory for the maps
    int count = 2;
    if (nnz != 2) {
        o->mapI = SAFECALLOC(npix, sizeof *o->mapI);
        count++;
    }
    if (nnz > 1) {
        o->mapQ = SAFECALLOC(npix, sizeof *o->mapQ);
        o->mapU = SAFECALLOC(npix, sizeof *o->mapU);
        count += 2;
    }
    o->rcond = SAFECALLOC(npix, sizeof *o->rcond);
    o->hits = SAFECALLOC(npix, sizeof *o->hits);

    // Define output file names
    o->count = count;
    o->files = SAFEMALLOC(sizeof *o->files * count);

    const char *template_names[] = {
        nnz != 2 ? "%s/mapI_%s.fits" : NULL,
        nnz > 1 ? "%s/mapQ_%s.fits" : NULL,
        nnz > 1 ? "%s/mapU_%s.fits" : NULL,
        "%s/Cond_%s.fits",
        "%s/Hits_%s.fits",
    };

    int c = 0;
    for (int i = 0; i < 5; ++i) {
        if (template_names[i] == NULL)
            continue;
        o->files[c] = SAFEMALLOC(FILENAME_MAX * sizeof(char));
        sprintf(o->files[c], template_names[i], outpath, ref);
        ++c;
    }

    assert(c == count);
}

void freeMappraiserOutputs(MappraiserOutputs *o) {
    FREE(o->mapI);
    FREE(o->mapQ);
    FREE(o->mapU);
    FREE(o->rcond);
    FREE(o->hits);
    for (int i = 0; i < o->count; ++i)
        FREE(o->files[i]);
    o->count = 0;
    FREE(o->files);
}

int clearFiles(MappraiserOutputs *o) {
    for (int i = 0; i < o->count; ++i) {
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
    for (int i = 0; i < 5; ++i) {
        void *map = NULL;
        int type = TDOUBLE;
        if (i == 0) {
            map = o->mapI;
        } else if (i == 1) {
            map = o->mapQ;
        } else if (i == 2) {
            map = o->mapU;
        } else if (i == 3) {
            map = o->rcond;
        } else if (i == 4) {
            map = o->hits;
            type = TINT;
        }
        if (map == NULL)
            continue;
        write_map(map, type, o->nside, o->files[c], nest, cordsys);
        ++c;
    }
    assert(c == o->count);
}
