#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>

#define nbpop 1
const char* dir = "Test" ;

#define N_NEURONS 10000UL
#define nbPref 10000UL
#define K 250.
#define g 1.

#define IF_RING 0
#define IF_GAUSS 0
const double Sigma[4] = {.25,.0625,.25,.125} ;


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

off_t fsize(const char *filename) {
  struct stat st; 
  
  if (stat(filename, &st) == 0)
    return st.st_size;
  
  return -1; 
}

///////////////////////////////////////////////////////////////////    

size_t getFileSize(const char* filename) {
  struct stat st;
  if(stat(filename, &st) != 0) {
    return 0;
  }
  return st.st_size;   
}

///////////////////////////////////////////////////////////////////    

void CreatePath(char *&path) {
  
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
  
  if(-1 == dir_err) {
    printf("error creating directory : ");
    exit(-1) ;
  }
  else {
    printf("Directory : ") ;
    printf("%s\n",path) ;
  }
  
}

///////////////////////////////////////////////////////////////////    

int dim_sort(const void *va, const void *vb)
{
  float a = *(const float *)va;
  float b = *(const float *)vb;

  if(a<b) return -1 ; 
  if(a>b) return 1 ;
  return 0 ;
}

///////////////////////////////////////////////////////////////////    

// __host__ void ImportSpikes() {

int main(int argc, char *argv[]) {

  unsigned long i=0;
  unsigned long j=0;

  char *path = '\0';
  CreatePath(path) ;

  FILE *file ;
  file = fopen(path,"r") ;
  unsigned long N = (unsigned long) sqrt(fsize(path)) ;

  printf("DATA SIZE : %lu\n",N) ;

  float **data ;
  data = (float **) malloc( (unsigned long) 2 * sizeof(float*) ) ;
  for(i=0;i<2;i++)
    data[i] = (float*) malloc( N * sizeof(float) ) ;
  
  printf("Read data \n") ;
  int dum ; 
    for(j=0;j<N;j++) {
      for(i=0;i<2;i++) 
	dum = fscanf(file, "%f", &data[i][j]) ; 
      /* printf("#%.3f Tspk %.3f\n", data[0][j], data[1][j]) ; */
    }
  
  fclose(file) ;
  printf("... Done \n") ;
  
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
    printf("#%.3f Tspk %.3f\r", data[0][i], data[1][i]) ; 
  
  float *SpkTimes ;
  unsigned long *neuronIdx ;
  unsigned long *nbSpk ;

  SpkTimes = (float *) malloc( (unsigned long) N * N_NEURONS * sizeof(float) ) ;
  
  neuronIdx = (unsigned long *) malloc( (unsigned long) N * sizeof(unsigned long) ) ;
  nbSpk = (unsigned long *) malloc( (unsigned long) N * N_NEURONS * sizeof(unsigned long) ) ;

  i=0;
  j=0;

  printf("\nSpike Times\n") ;
  while(i<N_NEURONS) { 
      nbSpk[i] = 0 ;
      for(j=0;j<N;j++) {
	if(i == data[0][j]) {
	  SpkTimes[j + i * N_NEURONS] = data[1][j] ;
	  nbSpk[i] += 1 ;
	  printf("# %lu Tspk ", i) ; 
	  printf("%.3f \r", SpkTimes[j + i * N_NEURONS]) ;
	  j++;
	}
      }
      i++ ;
  }
  
  printf("\n") ;

  for(i=0;i<100;i++) 
    printf("# %lu nbSpk %lu \n", i, nbSpk[i] ) ; 
  
  unsigned long nbSpkMax = nbSpk[0] ;
  unsigned long idxMax = 0 ;

  i=1 ;
  while(i<N_NEURONS) {
    if(nbSpk[i]>nbSpkMax) {
      nbSpkMax = nbSpk[i] ;
      idxMax = i ;
    }
    i++ ;
  }
    
  printf("\n#%lu max nbSpk %lu\n", idxMax, nbSpkMax) ;
  
  free(SpkTimes) ;
  free(nbSpk) ;
  free(neuronIdx) ;

}
