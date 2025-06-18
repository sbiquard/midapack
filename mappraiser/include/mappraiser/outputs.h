#ifndef MAPPRAISER_OUTPUTS_H
#define MAPPRAISER_OUTPUTS_H

#include <stdbool.h>

#define NUM_MAPS 10
#define NEST 1
#define COORDSYS "C"

typedef struct {
    // output objects
    int nside;
    int *hits;
    double *maps[NUM_MAPS];
    double *rcond;
    double *mapI;
    double *mapQ;
    double *mapU;

    // mirror maps
    bool mirror;
    double *mirror_mapI;
    double *mirror_mapQ;
    double *mirror_mapU;

    // inverse preconditioner (symmetric nnzxnnz matrix for each pixel)
    double *precII;
    double *precIQ;
    double *precIU;
    double *precQQ;
    double *precQU;
    double *precUU;
} MappraiserOutputs;

void initMappraiserOutputs(MappraiserOutputs *o, int size, int nnz,
                           bool mirror);

void freeMappraiserOutputs(MappraiserOutputs *o);

void populateMappraiserOutputs(MappraiserOutputs *o, const double *x,
                               const int *lstid, const double *rcond,
                               const int *lhits, const double *bj_map,
                               int xsize, int nnz);

int writeFiles(MappraiserOutputs *o, const char *outpath, const char *ref);

#endif // MAPPRAISER_OUTPUTS_H
