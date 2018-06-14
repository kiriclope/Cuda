#ifndef _DEV_FUNC_PROTOS_
#define _DEV_FUNC_PROTOS_

__host__ void nbNeurons(int N, int* &Nk) ;

__host__ void CptNeurons(int N, int *Nk, int* &Cpt) ;

__host__ void CreatePath(char *&path,int N) ;

__host__ void CheckPres(char *path, int* Nk, int **nbPreSab) ;

__host__ void WritetoFile(char *path, int N, int *IdPost, int *nbPost, unsigned long int *idxPost) ;

__host__ void WriteMatrix(char *path, int N, int *IdPost, int *nbPost, unsigned long int *idxPost) ;

#endif
