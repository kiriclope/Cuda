#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include "librairies.h"
#include "CudaFunc.cu"
#include "cuPrintf.cu"

#include "GlobalVars.h"
#include "ImportSpikes.cu"
#include "CrossCorr.cu"

#define SpkThresh 1
///////////////////////////////////////////////

///////////////////////////////////////////////////////////////////    

__host__ void nbNeurons(unsigned long * &nbN) {
  
  cudaCheck(cudaMallocHost((void **)&nbN, nbpop * sizeof( unsigned long )));
  printf("Number of neurons : ") ;
  unsigned long i = 0; 
  while(i<nbpop) {
    if(i==0) 
      nbN[i] = N_NEURONS*popSize ;
    else
      nbN[i] = (unsigned long) ( N_NEURONS - nbN[0] ) / max( (nbpop-1), 1 ) ;
       
    printf("%lu ", nbN[i]) ;
    ++i ;
  }
  printf("\n") ;
}

///////////////////////////////////////////////////////////////////    
 
__host__ void CptNeurons(unsigned long* nbN, unsigned long* &Cpt) {
  cudaCheck(cudaMallocHost((void **)&Cpt, nbpop * sizeof( unsigned long int)));
  printf("Counter : ") ;

  unsigned long i,j;
  for(i=0;i<nbpop+1;i++) {
    Cpt[i] = 0 ;
    for(j=0;j<i;j++) {
      Cpt[i] = Cpt[i] + nbN[j] ; 
    } 
    printf("%lu ", Cpt[i]) ;
  } 
  printf("\n") ; 
}

///////////////////////////////////////////////////////////////////    

int main(int argc, char *argv[]) {
  
  unsigned long i,j,k,l,ii;
  unsigned long int *nbN, *Cpt ;

  nbNeurons(nbN) ;
  CptNeurons(nbN, Cpt) ;

  cudaPrintfInit();

  ///////////////////////////
  // Import Data
  ///////////////////////////

  printf("Import Data : ") ;
  
  unsigned long Nfft ; 

  float **SpkTimes ;
  unsigned long *nbSpk = NULL ;
  
  ImportSpikeTrains(Nfft, SpkTimes, nbSpk) ;
  
  printf(", Nfft %lu\n", Nfft) ;

  printf("Sort SpikeTrains ") ; 
  for(i=0;i<N_NEURONS;i++)
    qsort(SpkTimes[i], nbSpk[i], sizeof(float), dim_sort) ;
  printf("... Done \n") ; 

  for(i=0;i<3;i++) {
    if(nbSpk[i]!=0) {
      printf("#%lu nbSpk %lu Tspk ",i , nbSpk[i] ) ; 
      for(j=0;j<10;j++) 
  	printf("%.3f ", SpkTimes[i][j]) ;
      printf("\n") ;
    }
  }
  
  // Nfft = (unsigned long) 512 ;

  // nbSpk = (unsigned long *) malloc( (unsigned long) 2 * sizeof(unsigned long) ) ;
  // SpkTimes = (float **) malloc( (unsigned long) 2 * sizeof(float*) ) ; 
  // for(i=0;i<2;i++) 
  //   SpkTimes[i] = (float*) malloc( (unsigned long) Nfft * sizeof(float) ) ; 

  // nbSpk[0] = Nfft/2 ;
  // nbSpk[1] = Nfft/2 ;

  // for(i=0;i<Nfft;i++) {

  //   if(i<Nfft/2)
  //     SpkTimes[0][i] = cos( (float) i );
  //   else
  //     SpkTimes[0][i] = 0. ;

  //   if(i<Nfft/2)
  //     SpkTimes[1][i] = sin( (float) i );
  //   else
  //     SpkTimes[1][i] = 0. ;    
  // }

  // unsigned long nbSpkMax = 0 ;
  // unsigned long idxMax = 0 ;

  // for(i=0;i<N_NEURONS;i++) {
  //   if(nbSpk[j]>nbSpkMax) {
  //     nbSpkMax = nbSpk[j] ;
  //     idxMax = j ;
  //   }
  // }

  ///////////////////////////
  // Compute 1D CrossCorr
  ///////////////////////////
  
  char cdum[500] ;
  float *xyCorr= NULL ;
  xyCorr = (float *) malloc( (unsigned long) Nfft * sizeof(float) );
  for(i=0;i<Nfft;i++) 
    xyCorr[i] = 0 ;

  float *AvgCorr= NULL ;
  AvgCorr = (float *) malloc( (unsigned long) Nfft * sizeof(float) );
  for(i=0;i<Nfft;i++) 
    AvgCorr[i] = 0 ;

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

  FILE *file, *idxFile ;
  if(IF_RING) {
    file = fopen("xyCorrRing.dat","wb") ;
    idxFile = fopen("idxRing.txt","w") ;
  }
  else {
    if(IF_AUTA) {
      sprintf(cdum,"xyCorr_%s_%s.dat", dir, strAuta) ;
      file = fopen(cdum,"wb") ;
    }
    else{
      sprintf(cdum,"xyCorr_%s.dat",dir) ;
      file = fopen(cdum,"wb") ;
    }
  }

  cufftHandle fftplan, ifftplan ;

  if (cufftPlan1d(&fftplan, Nfft, CUFFT_R2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed") ;
    exit(-1) ;
  }

  if (cufftPlan1d(&ifftplan, Nfft, CUFFT_C2R, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed") ;
    exit(-1) ;
  }

  cufftReal *dev_xdata, *dev_ydata, *dev_xyCorr ;
  cufftComplex *dev_iXdata, *dev_iYdata, *dev_iZdata ;

  cudaCheck( cudaMalloc((void**)&dev_xdata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_ydata, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_xyCorr, (unsigned long) Nfft * sizeof(cufftReal) * BATCH) ) ;
  
  cudaCheck( cudaMalloc((void**)&dev_iXdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iYdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;
  cudaCheck( cudaMalloc((void**)&dev_iZdata, (unsigned long) (Nfft/2+1) * sizeof(cufftComplex) * BATCH) ) ;

  // AllocateDevicePtr(Nfft, dev_xdata, dev_ydata, dev_xyCorr, dev_iXdata, dev_iYdata, dev_iZdata) ;

  cudaPrintfInit();

  unsigned long counter = 0 ;

  // for(i=0;i<N_NEURONS;i++) 
  //   for(j=AUTOCORR*i;j<=i;j++) {
      
  //     if(nbSpk[i]>=SpkThresh && nbSpk[j]>=SpkThresh ) {

  // 	  printf("Corr i %lu j %lu \r", i, j) ;
	  
  // 	  CrossCorr(Nfft, SpkTimes[i], SpkTimes[j], xyCorr, fftplan, ifftplan, dev_xdata, dev_ydata, dev_xyCorr, dev_iXdata, dev_iYdata, dev_iZdata) ; 
  // 	  fwrite(xyCorr, sizeof(float), Nfft, file) ;

  // 	  if(IF_RING)
  // 	    fprintf(idxFile,"%d\n" , abs( (int) (i-j) ) ) ;
	  
  // 	  counter ++ ;
	  
  // 	  for(k=0;k<Nfft;k++) {
  // 	    // printf("%f ", xyCorr[k]) ;	
  // 	    AvgCorr[k] +=  xyCorr[k] ;
  // 	    xyCorr[k] = 0 ;
  // 	  }
  // 	  // printf("\n") ;
  // 	}
  //   }

  for(i=0;i<nbpop;i++) 
    for(j=0;j<=i;j++) {
      counter = 0 ; 

      if(IF_AUTA) 
	sprintf(cdum,"%s_xyCorr%lu%lu_%s_%s.dat",model,i,j,dir,strAuta) ; 
      else
	sprintf(cdum,"xyCorr%lu%lu.dat",i,j) ; 

      printf("Writting %s\n", cdum ) ;

      file = fopen(cdum,"wb") ; 
      
      for(k=Cpt[i];k<Cpt[i+1];k++) 
	for(l=AUTOCORR*k;l<=k;l++) {
	  
	  if(nbSpk[k]>=SpkThresh && nbSpk[l]>=SpkThresh ) {

	    printf("Corr i %lu j %lu \r", k, l) ;	  
	    CrossCorr(Nfft, SpkTimes[k], SpkTimes[l], xyCorr, fftplan, ifftplan, dev_xdata, dev_ydata, dev_xyCorr, dev_iXdata, dev_iYdata, dev_iZdata) ; 
	    fwrite(xyCorr, sizeof(float), Nfft, file) ;
	    counter ++ ;

	    if(IF_RING) 
	      fprintf(idxFile,"%d\n" , abs( (int) (k-l) ) ) ; 
	    
	    for(ii=0;ii<Nfft;ii++) {
	      AvgCorr[ii] +=  xyCorr[ii] ; 
	      xyCorr[ii] = 0 ;
	    }
	  }
	}
      fclose(file) ;
      
      for(k=0;k<Nfft;k++) 
	AvgCorr[k] =  AvgCorr[k] / (float) counter ; 

      sprintf(cdum,"%s_AvgCorr%lu%lu.dat",model,i,j) ; 
      printf("Writting %s\n", cdum ) ;
      file = fopen(cdum,"wb") ; 
      fwrite(AvgCorr, sizeof(float), Nfft, file) ;
      fclose(file) ;

      for(k=0;k<Nfft;k++) 
	AvgCorr[k] = 0.0 ; 
      
    }

  if(IF_RING)
    fclose(idxFile) ;
  
  // for(k=0;k<nbpop*Nfft;k++) 
  //   AvgCorr[k] =  AvgCorr[k] / (float) counter ;
  
  printf("\n") ;

  printf("... Done\nFree Device Ptr \n") ;

  cufftDestroy(fftplan) ;
  cufftDestroy(ifftplan) ;
  
  cudaFree(dev_xdata) ; 
  cudaFree(dev_ydata) ;
  cudaFree(dev_xyCorr) ;

  cudaFree(dev_iXdata) ;
  cudaFree(dev_iYdata) ;
  cudaFree(dev_iZdata) ;

  // fclose(file) ;

  ///////////////////////////

  // if(IF_RING)
  //   file = fopen("AvgCorrRing.dat","wb") ;
  // else
  //   file = fopen("AvgCorr.dat","wb") ;

  // fwrite(AvgCorr, sizeof(float), Nfft, file) ;
  // fclose(file) ;

  ///////////////////////////
  
  printf("... Done\nFree CPU Memory \n") ;

  free(SpkTimes) ;
  free(nbSpk) ;
  free(xyCorr) ;  
  free(AvgCorr) ; 

  ///////////////////////////
  
  return 0 ;
}
