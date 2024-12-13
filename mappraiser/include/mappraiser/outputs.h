#ifndef MAPPRAISER_OUTPUTS_H
#define MAPPRAISER_OUTPUTS_H

typedef struct {
    // output objects
    int nside;
    int *hits;
    double *rcond;
    double *mapI;
    double *mapQ;
    double *mapU;

    // inverse preconditioner (symmetric nnzxnnz matrix for each pixel)
    double *precII;
    double *precIQ;
    double *precIU;
    double *precQQ;
    double *precQU;
    double *precUU;

    // files
    int max_count;
    int real_count;
    char **files;
} MappraiserOutputs;

void initMappraiserOutputs(MappraiserOutputs *o, int size, int nnz,
                           const char *outpath, const char *ref);

void freeMappraiserOutputs(MappraiserOutputs *o);

void populateMappraiserOutputs(MappraiserOutputs *o, const double *x,
                               const int *lstid, const double *rcond,
                               const int *lhits, const double *bj_map,
                               int xsize, int nnz);

int clearFiles(MappraiserOutputs *o);

void writeFiles(MappraiserOutputs *o);

#endif // MAPPRAISER_OUTPUTS_H
