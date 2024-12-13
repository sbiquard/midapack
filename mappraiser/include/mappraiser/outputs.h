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

    // files
    int count;
    char **files;
} MappraiserOutputs;

void initMappraiserOutputs(MappraiserOutputs *o, int size, int nnz,
                           const char *outpath, const char *ref);

void freeMappraiserOutputs(MappraiserOutputs *o);

int clearFiles(MappraiserOutputs *o);

void writeFiles(MappraiserOutputs *o);

#endif // MAPPRAISER_OUTPUTS_H
