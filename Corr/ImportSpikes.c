#include "GlobalVars.h"
#include "librairies.h"
#include <gsl/gsl_sort.h>

///////////////////////////////////////////////////////////////////    

int dirExists(const char *path) {
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

unsigned long fsize(FILE *file) {
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

///////////////////////////////////////////////////////////////////    

char* CreatePath() {
  
  char *path ;
  char *mkdirp ;   
  char cdum[500] ;
  char strCrec[100] ;

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
      sprintf(cdum, "../LIF/Simulations/%dpop/%s/N%d/K%.0f/Gauss/%s/Raster.txt", nbpop, dir, (int) (N_NEURONS/nbPref), K, strCrec) ; 
    else
      sprintf(cdum, "../LIF/Simulations/%dpop/%s/N%d/K%.0f/Ring/%s/Raster.txt", nbpop, dir, (int) (N_NEURONS/nbPref), K, strCrec) ; 
  }
  else
    sprintf(cdum, "../../LIF/Simulations/%dpop/%s/N%d/K%.0f/g%.2f/Raster.txt", nbpop, dir, (int) (N_NEURONS/nbPref), K, g) ;
  
  path = (char *) malloc( strlen(cdum) + 100) ;
  strcpy(path,cdum) ;

  mkdirp = (char *) malloc(strlen(path)+100);
  
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

int dim_sort(const void *va, const void *vb) {
  float a = *(const float *)va;
  float b = *(const float *)vb;

  if(a<b) return -1 ; 
  if(a>b) return 1 ;
  return 0 ;
}

///////////////////////////////////////////////////////////////////    

/* void ImportSpikeTrains(unsigned long &Nfft, float** &SpkTimes, unsigned long* &nbSpk) { */
int main(int argc, char *argv[]) {

  unsigned long Nfft ;
  float **SpkTimes ;
  unsigned long *nbSpk = NULL ;
  
  unsigned long i=0;
  unsigned long j=0;

  char *path ;
  path = CreatePath() ;

  FILE *file ;
  file = fopen(path,"r") ;

  printf("%p\n",file) ;

  if(file==NULL) {
    printf("ERROR could not open file\n") ;
    exit(-1) ;
  }

  unsigned long N =  fsize(file) ; 

  printf("DATA SIZE : %lu \n", N) ;
  if(N==0) {
    printf("ERROR empty file\n") ;
    exit(-1) ;
  }

  rewind(file);

  float **data ;
  data = (float **) malloc( 2 * sizeof(float*) ) ;
  for(i=0;i<2;i++) {
    data[i] = (float*) malloc( (unsigned long) N * sizeof(float) ) ;
    for(j=0;j<N;j++) 
      data[i][j] = 0. ;
  }
  
  printf("Read data ") ;
  /* for(i=0;i<N;i++) */
  /* fread(&data[i], sizeof data[0][0], 2, file) ; */

  /* float x ; */
  /* fscanf(file,"%f ",&x) ; */
  /* printf("%f\n",x) ; */

  for(j=0;j<N;j++) {
    for(i=0;i<2;i++)
      fscanf(file,"%f ", &data[i][j] ) ;
    // printf("#%.3f Tspk %.3f\n", data[0][j], data[1][j]) ;
  }
  printf("... Done \n") ;
  fclose(file) ;

  for(i=0;i<N;i++)
    printf("#%.3f Tspk %.3f\n", data[0][i], data[1][i]) ; 
  printf("\n") ;


  printf("Sort Data ") ; 
  // for(int i=0;i<N;i++)
  //   qsort(data[i], N, 2*sizeof(float), dim_sort) ;

  /* gsl_sort_float(data[0], 1, N) ; */
  /* gsl_sort2_float(data[0], 1, data[1], 1, N) ; */

  size_t p[N] ;
  float tmp[N] ;
  gsl_sort_float_index(p, data[0], 1, N) ;

  /* for(i=0;i<N;i++)  */
    /* printf("%d ",p[i]) ; */

  printf("... Done \n") ;
  
  for(i=0;i<N;i++) 
    tmp[i] = data[0][p[i]] ;
  for(i=0;i<N;i++) 
    data[0][i] = tmp[i] ;

  for(i=0;i<N;i++) 
    tmp[i] = data[1][p[i]] ; 
  for(i=0;i<N;i++) 
    data[1][i] = tmp[i] ; 
  
  for(i=0;i<N;i++)
    printf("#%.3f Tspk %.3f\n", data[0][i], data[1][i]) ; 
  printf("\n") ;

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

  for(i=0;i<N_NEURONS;i++) 
    if(nbSpk[i]!=0) {
      printf("# %lu nbSpk %lu \r", i, nbSpk[i] ) ;
      printf("\n") ;
    }

  CumSumSpk[0] = 0 ;
  for(i=0;i<N_NEURONS;i++)
    CumSumSpk[i+1] = CumSumSpk[i] + nbSpk[i] ;

  printf("#%lu max nbSpk %lu\n", idxMax, nbSpkMax) ;
  
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

  free(SpkTimes) ;
  free(nbSpk) ;

  free(data) ;
  free(CumSumSpk) ;

}