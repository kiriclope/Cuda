/* #include "GlobalVars.h" */
/* #include "librairies.h" */
#include <gsl/gsl_sort.h>

///////////////////////////////////////////////////////////////////    

__host__ int dirExists(const char *path) {
  struct stat info;

  if(stat( path, &info ) != 0)
    return 0;
  else 
    if(info.st_mode & S_IFDIR)
      return 1;
    else
      return 0;
}
///////////////////////////////////////////////////////////////////    

__host__ unsigned long fsize(FILE *file) {
  unsigned long lines = 0;
  int ch ;
  while(!feof(file))
    {
      ch = fgetc(file);
      if(ch == '\n')
	{
	  lines++;
	}
    }
  return lines ;
}
// __host__ size_t fsize(const char *filename) {
//   // struct stat st; 
  
//   // if (stat(filename, &st) == 0)
//   //   return st.st_size;
  
//   // return -1 ;
// }

///////////////////////////////////////////////////////////////////    

__host__ char* CreatePath() {
  
  char *path ;
  char *mkdirp ;   
  char cdum[500] ;
  char strCrec[100] ;

  char strAuta[100] ;
  switch(AUTA_Pop) {
  case 0 :
    if(nbpop>1)
      sprintf(strAuta,"AutaE%.2f", AUTA_p[0]) ;
    else
      sprintf(strAuta,"AutaI%.2f", AUTA_p[0]) ;
    break ;
  case 1 :
    sprintf(strAuta,"AutaI%.2f", AUTA_p[1]) ;
    break ;
  case 2 :
    sprintf(strAuta,"AutaE%.2fI%.2f", AUTA_p[0], AUTA_p[1] ) ; 
    break ;
  }

  if(IF_RING) {

    if(nbpop==1) 
      sprintf(strCrec,"CrecI%.4f",Sigma[0]) ;
    if(nbpop==2) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4f",Sigma[0],Sigma[1]);
    if(nbpop==3) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4f",Sigma[0],Sigma[1],Sigma[2]);
    if(nbpop==4) 
      sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4fCrecV%.4f",Sigma[0],Sigma[1],Sigma[2],Sigma[3]);
    
    if(IF_GAUSS)
      sprintf(cdum, "../../%s/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/Gauss/%s/Raster.txt", model, nbpop, dir, (int) (N_NEURONS/nbPref), K, g, strCrec) ; 
    else
      sprintf(cdum, "../../%s/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/Ring/%s/Raster.txt", model, nbpop, dir, (int) (N_NEURONS/nbPref), K, g, strCrec) ; 
  }
  else {
    // sprintf(cdum, "../../spktimes_p20.txt") ; 
    if(IF_AUTA)
      // sprintf(cdum, "../../%s/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/AUTA_p%.2f/Raster.txt", model, nbpop, dir, (int) (N_NEURONS/nbPref), K, g, AUTA_p) ;
      sprintf(cdum, "../../%s/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/%s/Raster.txt", model, nbpop, dir, (int) (N_NEURONS/nbPref), K, g, strAuta) ;
          // sprintf(cdum, "../../spktimes_p0.txt") ; 
    else
      sprintf(cdum, "../../%s/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/Raster.txt", model, nbpop, dir, (int) (N_NEURONS/nbPref), K, g) ;
  }

  path = (char *) malloc( strlen(cdum) + 100) ; 
  strcpy(path,cdum) ; 

  mkdirp = (char *) malloc(strlen(path)+100) ; 
  
  strcpy(mkdirp,"mkdir -p ") ;
  strcat(mkdirp,path) ; 

  int dir_err = system(mkdirp) ;
  dir_err = dirExists(path) ;
  
  if(1 == dir_err) {
    printf("error creating directory : ");
    exit(-1) ;
  }
  else {
    printf("Directory : ") ;
    printf("%s\n",path) ;
  }

  return path ;
}

///////////////////////////////////////////////////////////////////    

__host__ int dim_sort(const void *va, const void *vb) {
  float a = *(const float *)va;
  float b = *(const float *)vb;

  if(a<b) return -1 ; 
  if(a>b) return 1 ;
  return 0 ;
}

///////////////////////////////////////////////////////////////////    

__host__ void ImportSpikeTrains(unsigned long &Nfft, float** &SpkTimes, unsigned long* &nbSpk) {
  
  unsigned long i=0;
  unsigned long j=0;

  char *path ;
  path = CreatePath() ;

  FILE *file ;
  file = fopen(path,"r") ;

  if(file==NULL) {
    printf("ERROR could not open file\n") ;
    exit(-1) ;
  }

  unsigned long N =  fsize(file) ; 
  rewind(file) ;

  printf("DATA SIZE : %lu", N) ;
  if(N==0) {
    printf("ERROR empty file\n") ;
    exit(-1) ;
  }
  
  float **data ;
  data = (float **) malloc( 2 * sizeof(float*) ) ; 
  for(i=0;i<2;i++) {
    data[i] = (float*) malloc( (unsigned long) N * sizeof(float) ) ;
    for(j=0;j<N;j++) 
      data[i][j] = 0. ;
  }
  
  printf("\nRead data ") ;

  int dum ;
  for(j=0;j<N;j++) {
    // dum = fscanf(file,"%f %f\n", &tmp0, &tmp1 ) ; 
    // printf("%f Tspk %f\n", tmp0, tmp1);
    dum = fscanf(file, "%f %f\n", &data[0][j], &data[1][j] ) ; 
    // data[0][j] *= 1.0E-4 ;
    // dum = fscanf(file, "%f %f\n", &data[0][j], &data[1][j] ) ; 
    // printf("#%.3f Tspk %.3f\n", data[0][j], data[1][j]) ; 
  }
  printf("... Done \n") ;

  fclose(file) ; 

  // for(i=0;i<10;i++) 
  //   printf("# %.3f Tspk %.3f\n", data[0][i], data[1][i]) ; 
  // printf("\n") ; 

  printf("Sort Data ") ; 
  // for(int i=0;i<N;i++)
  //   qsort(data[i], N, 2*sizeof(float), dim_sort) ;
  /* gsl_sort_float(data[0], 1, N) ; */
  /* gsl_sort2_float(data[0], 1, data[1], 1, N) ; */

  size_t *p ;
  p = (size_t *) malloc( (unsigned long) N * sizeof(size_t) ) ; 
  float *tmp ; 
  tmp = (float *) malloc( (unsigned long) N * sizeof(float) ) ; 

  gsl_sort_float_index(p, data[0], 1, (size_t) N) ; 

  // for(i=0;i<10;i++)
  //   printf("%d ",p[i]) ;
  
  for(i=0;i<N;i++) 
    tmp[i] = data[0][p[i]] ;
  for(i=0;i<N;i++) 
    data[0][i] = tmp[i] ;

  for(i=0;i<N;i++) 
    tmp[i] = data[1][p[i]] ; 
  for(i=0;i<N;i++) 
    data[1][i] = tmp[i] ; 

  
  for(i=0;i<3;i++)
    printf("#%.3f Tspk %.3f\n", data[0][i], data[1][i]) ; 
  printf("... Done \n") ;

  unsigned long *CumSumSpk ; 

  nbSpk = (unsigned long *) malloc( (unsigned long) N_NEURONS * sizeof(unsigned long) ) ;
  CumSumSpk = (unsigned long *) malloc( (unsigned long) (N_NEURONS+1) * sizeof(unsigned long) ) ;

  unsigned long nbSpkMax = 0 ;
  unsigned long idxMax = 0 ;

  for(i=0;i<N_NEURONS;i++) 
    nbSpk[i] = 0 ;

  for(i=0;i<N;i++) {
    j = (unsigned long) data[0][i] ;    
    nbSpk[j] += 1 ;
    
    if(nbSpk[j]>nbSpkMax) {
	nbSpkMax = nbSpk[j] ;
	idxMax = j ;
    }
  }

  // for(i=0;i<10;i++) 
  //   if(nbSpk[i]!=0) {
  //     printf("# %lu nbSpk %lu \r", i, nbSpk[i] ) ;
  //     printf("\n") ;
  //   }
  
  CumSumSpk[0] = 0 ;
  for(i=0;i<N_NEURONS;i++)
    CumSumSpk[i+1] = CumSumSpk[i] + nbSpk[i] ;

  // printf(", nbSpkMax %lu #%lu", nbSpkMax, idxMax) ;
  
  Nfft = pow( 2, ( ceil( log2( (float) ( 2*nbSpkMax-1 ) ) ) ) ) ; 

  SpkTimes = (float **) malloc( (unsigned long) N_NEURONS * sizeof(float*) ) ; 
  for(i=0;i<N_NEURONS;i++) 
    SpkTimes[i] = (float*) malloc( (unsigned long) Nfft * sizeof(float) ) ; 
  
  for(i=0;i<N_NEURONS;i++) 
    for(j=0;j<Nfft;j++) 
      SpkTimes[i][j] = 0. ; 

  j=0 ; 
  while(j<N) { 
    SpkTimes[ (unsigned long) data[0][j] ][ (unsigned long) ( j - CumSumSpk[ (unsigned long) data[0][j] ] ) ] = data[1][j] ; 
    j++ ;
  }

  // for(i=0;i<10;i++) {
  //   printf("# %lu Tspk ", i) ;
  //   for(j=0;j<10;j++)
  //     printf("%.3f ", SpkTimes[i][j] ) ;
  //   printf("\n") ;
  // }
  // free(SpkTimes) ;

  free(data) ;
  free(CumSumSpk) ;
  
}
