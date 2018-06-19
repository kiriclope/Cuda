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

  char strCrec[100] ;
  if(nbpop==1) 
    sprintf(strCrec,"CrecI%.4f",Sigma[0]);
  if(nbpop==2) 
    sprintf(strCrec,"CrecE%.4fCrecI%.4f",Sigma[0],Sigma[1]);
  if(nbpop==3) 
    sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4f",Sigma[0],Sigma[1],Sigma[2]);
  if(nbpop==4) 
    sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4fCrecX%.4f",Sigma[0],Sigma[1],Sigma[2],Sigma[3]);
  
  if(IF_BUMP) 
    sprintf(cdum, "../Connectivity/%dpop/N%d/K%.0f/Ring/%s", nbpop, (int) (nbpop), K, strCrec) ;
  else
    if(IF_SPACE) 
      sprintf(cdum, "../Connectivity/%dpop/N%d/K%.0f/Gauss/%s", nbpop, (int) (nbpop), K, strCrec) ; 
    else 
      sprintf(cdum, "../Connectivity/%dpop/N%d/K%.0f", nbpop, (int) (nbpop), K) ;
  
  path = (char *) malloc( strlen(cdum) + 100) ;
  strcpy(path,cdum) ;

  mkdirp = (char *) malloc(strlen(path)+100);
  
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
  strPreSab =  (char *) malloc( strlen(path) + strlen(str) + 100) ;

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

  const char *strIdPost = "/IdPost.dat";
  const char *stridxPost = "/idxPost.dat";
  const char *strnbPost = "/nbPost.dat"; 

  nbpath =  (char *) malloc(strlen(path)+strlen(strnbPost) + 100 ) ;
  idxpath = (char *) malloc(strlen(path)+strlen(stridxPost) + 100 ) ;
  Idpath = (char *)  malloc(strlen(path)+strlen(strIdPost) + 100 ) ;

  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  unsigned long int nbCon = 0;
  for(int i = 0; i < N_NEURONS; i++) 
    nbCon += nbPost[i];
  
  FILE *fIdPost, *fnbPost, *fidxPost ;
  
  printf("sizeof IdPost %ld \n",  nbCon) ;

  for(int i=0;i<10;i++)
    printf("%d ",IdPost[i]);
  printf("\n"); 

  fIdPost = fopen(Idpath, "wb");
  fwrite(IdPost, sizeof(*IdPost) , nbCon , fIdPost);
  fclose(fIdPost);

  printf("%s\n",Idpath) ;

  for(int i=1;i<N;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;

  for(int i=0;i<10;i++)
    printf("%lu ",idxPost[i]);
  printf("\n"); 

  fidxPost = fopen(idxpath, "wb") ;
  fwrite(idxPost, sizeof(*idxPost) , N_NEURONS , fidxPost); 
  fclose(fidxPost);

  printf("%s\n",idxpath) ;

  for(int i=0;i<10;i++)
    printf("%d ",nbPost[i]);
  printf("\n"); 

  fnbPost = fopen(nbpath, "wb") ;
  fwrite(nbPost, sizeof(*nbPost) , N_NEURONS , fnbPost) ;
  fclose(fnbPost);

  printf("%s\n",nbpath) ;
  printf("Done\n") ;

}

///////////////////////////////////////////////////////////////////    

__host__ void WriteMatrix(char *path, float *conVec) {

  printf("Writing Cij Matrix to : \n") ;
  const char* strMatrix = "/Cij_Matrix.dat";
  char *pathMatrix ;
  pathMatrix = (char *) malloc(strlen(path)+strlen(strMatrix)+10) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");
  
  fwrite(conVec, sizeof(float), N_NEURONS * N_NEURONS , Out) ;

  fclose(Out) ;
}

__host__ void CheckSparseVec(char * path) {

  char *nbpath  ;
  char *idxpath ;
  char  *Idpath ;

  const char *strIdPost = "/IdPost.dat";
  const char *stridxPost = "/idxPost.dat";
  const char *strnbPost = "/nbPost.dat"; 

  nbpath =  (char *) malloc(strlen(path)+strlen(strnbPost) + 100 ) ;
  idxpath = (char *) malloc(strlen(path)+strlen(stridxPost) + 100 ) ;
  Idpath = (char *)  malloc(strlen(path)+strlen(strIdPost) + 100 ) ;
  
  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  int *nbPost ;
  unsigned long int *idxPost ;
  int *IdPost ;

  nbPost = new int [N_NEURONS] ;
  idxPost = new unsigned long int [N_NEURONS] ;

  FILE *fnbPost, *fidxPost, *fIdPost ;
  
  int dum ;
  
  fnbPost = fopen(nbpath, "rb") ;
  dum = fread(&nbPost[0], sizeof nbPost[0], N_NEURONS , fnbPost);  
  fclose(fnbPost);
  
  fidxPost = fopen(idxpath, "rb") ;
  dum = fread(&idxPost[0], sizeof idxPost[0], N_NEURONS , fidxPost);
  fclose(fidxPost);
  
  unsigned long int nbposttot = 0 ;
  for(int j=0 ; j<N_NEURONS; j++)
    nbposttot += nbPost[j] ;
  
  IdPost = new int [nbposttot] ;
  
  fIdPost = fopen(Idpath, "rb");
  dum = fread(&IdPost[0], sizeof IdPost[0], nbposttot , fIdPost); 
  fclose(fIdPost);
  
  printf("Writing Cij Matrix to : \n") ;
  const char* strMatrix = "/Cij_Matrix.dat";
  char *pathMatrix ;
  pathMatrix = (char *) malloc(strlen(path)+strlen(strMatrix)+10) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");
  
  int **M ;
  M = new int*[N_NEURONS] ;
  for(int i=0;i<N_NEURONS;i++) 
    M[i] = new int[N_NEURONS]() ;

  for(int i=1;i<N_NEURONS;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;
  
  for(int i=0;i<N_NEURONS;i++) 
    for(int l=idxPost[i]; l<idxPost[i]+nbPost[i]; l++) 
      M[IdPost[l]][i] = 1 ;
  
  for (int i=0; i<N_NEURONS; i++) 
    fwrite(M[i], sizeof(M[i][0]), N_NEURONS, Out) ;
  
  fclose(Out) ;
  delete [] M ;

}


///////////////////////////////////////////////////////////////////    

__global__ void GenSparseRep(float *dev_conVec, int *dev_IdPost, int *dev_nbPost, int *dev_nbPreS, int lChunk, int maxNeurons) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long int kNeuron = id + lChunk * maxNeurons;
  unsigned long int i;
  int nbPost ;

  if(id < maxNeurons & kNeuron < N_NEURONS) {

    dev_nbPost[kNeuron] = 0 ;
    dev_nbPreS[whichPop(kNeuron)] = 0 ;

    nbPost = 0 ;
      
    for(i=0;i<N_NEURONS;i++) {
      
      if(dev_conVec[id + i * maxNeurons]) { // id-->i column to row
	dev_IdPost[kNeuron + nbPost * N_NEURONS] = i ; 
	nbPost += 1 ;
      }
      
    }
    
    dev_nbPreS[whichPop(kNeuron)] += nbPost ; 
    dev_nbPost[kNeuron] += nbPost ; 
  }
  
}

#endif
