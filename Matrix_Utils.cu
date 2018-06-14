#ifndef __MATRIXUTILS__
#define __MATRIXUTILS__

///////////////////////////////////////////////////////////////////    

__host__ __device__ int whichPop(unsigned long int neuronIdx) {
  
  // const double PROPORTIONS[4] = {.75,.25,0,0} ;

  int popIdx = 0 ;
  unsigned long propPop = N_NEURONS*popSize ;

  while( neuronIdx > propPop-1 ) {
    popIdx++ ;
    propPop += ( N_NEURONS * (100 - int( popSize * 100 ) ) / 100 ) / max( (nbpop-1), 1 ) ;
  }
  return popIdx ;
}

///////////////////////////////////////////////////////////////////    

__host__ int dirExists(const char *path) {
  struct stat info;

  if(stat( path, &info ) != 0)
    return 0;
  else if(info.st_mode & S_IFDIR)
    return 1;
  else
    return 0;
}

///////////////////////////////////////////////////////////////////    

__host__ void CreatePath(char *&path,int N) {
  
  char *mkdirp ;   
  char cdum[500] ;

  if(IF_SPACE) {
    char strCrec[100] ;
    if(nbpop==1) 
      sprintf(strCrec,"CrecI%.4f",Sigma[0]);
    if(nbpop==2) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4f",Sigma[0],Sigma[1]);
    if(nbpop==3) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4f",Sigma[0],Sigma[1],Sigma[2]);
    if(nbpop==4) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4fCrecX%.4f",Sigma[0],Sigma[1],Sigma[2],Sigma[3]);
    
    sprintf(cdum, "../Connectivity/%dpop/N%d/K%.0f/Gauss/%s", nbpop, (int) (N/N_NEURONS*nbpop), K, strCrec) ;
  }
  else
    sprintf(cdum, "../Connectivity/%dpop/N%d/K%.0f", nbpop, (int) (N/N_NEURONS*nbpop), K) ;

  path = (char *) malloc( strlen(cdum) ) ;
  strcpy(path,cdum) ;

  mkdirp = (char *) malloc(strlen(path)+20);
  
  strcpy(mkdirp,"mkdir -p ") ;
  strcat(mkdirp,path) ; 
  // printf("%s\n",mkdirp) ;

  const int dir_err = system(mkdirp);
  // printf("%d\n",dir_err) ;

  if(-1 == dir_err) {
    printf("error creating directory : ");
  }
  else {
    printf("Created directory : ") ;
  }
  printf("%s\n",path) ;
}

///////////////////////////////////////////////////////////////////    

__host__ void CheckPres(char *path, int* Nk, int **nbPreSab) {

  printf("Average nbPreS : ");
  const char * str ="/nbPreSab.txt" ;
  char *strPreSab ;
  strPreSab =  (char *) malloc( strlen(path) + strlen(str) ) ;

  strcpy(strPreSab,path) ;
  strcat(strPreSab,str) ;

  FILE * pFile;
  pFile = fopen (strPreSab,"w");

  for(int i=0;i<nbpop;i++) 
    for(int j=0;j<nbpop;j++) { 
      printf("%.3f ", nbPreSab[i][j]/(double)Nk[i]);
      fprintf(pFile,"%.3f ", nbPreSab[i][j]/(double)Nk[i]);
    }
  printf("\n");

  fclose(pFile);
}

///////////////////////////////////////////////////////////////////    

__host__ void WritetoFile(char *path, int N, int *IdPost, int *nbPost, unsigned long int *idxPost) {

  printf(" Writing to Files :\n") ;

  char *nbpath  ;
  char *idxpath ;
  char  *Idpath ;

  const char *strIdPost = "/nbPost.dat";
  const char *stridxPost = "/idxPost.dat";
  const char *strnbPost = "/IdPost.dat"; 

  nbpath =  (char *) malloc(strlen(path)+strlen(strnbPost)) ;
  idxpath = (char *) malloc(strlen(path)+strlen(stridxPost)) ;
  Idpath = (char *)  malloc(strlen(path)+strlen(strIdPost)) ;

  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  FILE *fIdPost, *fnbPost, *fidxPost ;
  
  fIdPost = fopen(Idpath, "wb");
  fwrite(IdPost, sizeof(*IdPost) ,  sizeof(IdPost) / sizeof(IdPost[0]) , fIdPost);
  fclose(fIdPost);

  printf("%s\n",Idpath) ;

  for(int i=1;i<N;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;

  fidxPost = fopen(idxpath, "wb") ;
  fwrite(idxPost, sizeof(*idxPost) , sizeof(idxPost) / sizeof(idxPost[0]) , fidxPost); 
  fclose(fidxPost);

  printf("%s\n",idxpath) ;

  fnbPost = fopen(nbpath, "wb") ;
  fwrite(nbPost, sizeof(*nbPost) , sizeof(nbPost) / sizeof(nbPost[0]) , fnbPost) ;
  fclose(fnbPost);

  printf("%s\n",nbpath) ;
  printf("Done\n") ;

}

///////////////////////////////////////////////////////////////////    

__host__ void WriteMatrix(char *path, int N, int *IdPost, int *nbPost, unsigned long int *idxPost) {

  printf("Writing Cij Matrix to : \n") ;
  const char* strMatrix = "/Cij_Matrix.dat";
  char *pathMatrix ;
  pathMatrix = (char *) malloc(strlen(path)+strlen(strMatrix)+10) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");
  
  int **M = (int **)malloc(N * sizeof(int *));
  for(int i=0; i<N; i++)
    M[i] = (int *) malloc(N * sizeof(int));

  for(int i=0;i<N;i++) 
    for(int l=idxPost[i]; l<idxPost[i]+nbPost[i]; l++) 
      M[IdPost[l]][i] = 1 ;
  
  for (int i=0; i<N; i++) 
    fwrite(M[i], sizeof(int), N, Out) ;
  
  fclose(Out) ;
  free(M) ;
}

__global__ void GenSparseRep(float *dev_conVec, float *dev_IdPost, float *dev_nbPost, int lChunk, int maxNeurons) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  int nbPost, counter ;

  for(i=0;i<N_NEURONS;i++) {

    nbPost = 0 ;
    counter = 0 ;

    if(dev_ConVec[i + id * N_NEURONS]) { // id-->i column to row
      IdPost[counter] = i ;
      nbPreSab[j][i]++ ;
      counter+=1 ;
    }

    dev_nbPost[i] += nbPost ;

  }
}

#endif
