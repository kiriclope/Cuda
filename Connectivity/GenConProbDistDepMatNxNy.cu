#ifndef _CONNECTION_PROB_
#define _CONNECTION_PROB_
#include <stdio.h>
#include <cuda.h>
#include "devHostConstants.h"

///////////////////////////////////////////////////////////////////    

__global__ void initPreFactor(float *dev_preFactor) {
  unsigned long i;
  for(i = 0; i < nbpop * NX_NEURONS ; i++) 
    dev_preFactor[i] = 0 ; 
}

/* GENERATE CONNECTION MATRIX */

__device__ double XCordinate(unsigned long int neuronIdx, unsigned long int *nbN, unsigned long int *Cpt) { 
  double X = 0 ;
  int i = whichPop(neuronIdx) ;

  if(DIMENSION==1)
    X = fmod( (double) (neuronIdx-Cpt[i]), (double) nbN[i] ) * L / (double) nbN[i] ;
  else
    X = fmod( (double) (neuronIdx-Cpt[i]),  sqrt( (double) nbN[i] ) ) * L / sqrt( (double) nbN[i] )  ;
  
  return X ;
}

__device__ double YCordinate(unsigned long int neuronIdx, unsigned long int *nbN, unsigned long int *Cpt) { 
  double Y = 0 ;
  int i = whichPop(neuronIdx) ;
  Y = floor( (double) (neuronIdx-Cpt[i]) / sqrt( (double) nbN[i] ) ) * L / sqrt( (double) nbN[i] )  ;

  return Y ;
}

///////////////////////////////////////////////////////////////////    

__device__ double Gaussian1D(double mu, double sigma) {
  if(sigma!=0.)
    return exp(-mu*mu/2./sigma/sigma)/sqrt(2.*M_PI)/sigma ;
  else
    return 1. ;
}
///////////////////////////////////////////////////////////////////    

__device__ double ShortestDistOnCirc(double X, double Y) {
  double dist = 0.0;
  
  dist = fmod(abs(X-Y),L) ;

  if(dist > 0.5*L)
    dist = dist-L ;
  else
    dist = 0.5*L * dist ; 

  return dist;
}

///////////////////////////////////////////////////////////////////    

__device__ double ConProb(double xa, double xb, double varianceOfGaussian) {
  double distX = 0.0 ; 
  int IF_PERIODIC = 1 ;
  if(IF_PERIODIC)
    distX = ShortestDistOnCirc(xa, xb) ;
  else 
    distX = abs(xa - xb) ; 
  return Gaussian1D(distX, varianceOfGaussian);
}

///////////////////////////////////////////////////////////////////    

__device__ double ConProb2D(double xa, double xb, double ya, double yb, double varianceOfGaussian) {
  double distX = 0.0, distY = 0.0 ;
  int IF_PERIODIC = 1 ; 
  if(IF_PERIODIC) {
    distX = ShortestDistOnCirc(xa, xb) ;
    distY = ShortestDistOnCirc(ya, yb) ;
  }
  else {
    distX = abs(xa - xb) ; 
    distY = abs(ya - yb) ; 
  }
  return Gaussian1D(distX, varianceOfGaussian) * Gaussian1D(distY, varianceOfGaussian) ;
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelGenConProbMat(float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons ;
  unsigned long int i;
  double xa;

  if(id < maxNeurons & kNeuron < NX_NEURONS) { 
    xa = XCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    for(i=0; i < NY_NEURONS; i++)  // i-->id column to row, P[row][clm] = G(X[row],X[clm],Sigma[clm]) 
      dev_conVec[i + id * NY_NEURONS ] = (float) ConProb(xa, XCordinate(i,nbN,Cpt), Sigma[whichPop(i)]); 
  }
}

///////////////////////////////////////////////////////////////////    

__global__ void KernelGenConProbMat2D(float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma) {

  unsigned long id =  (unsigned long)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long kNeuron = id + lChunck * maxNeurons ;
  unsigned long i;
  double xa, ya ;

  if(id < maxNeurons & kNeuron < N_NEURONS) {    
    xa = XCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    ya = YCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    for(i=0; i < N_NEURONS; i++)  // i-->id column to row, P[row][clm] = G(X[row],X[clm],Sigma[clm]) 
      dev_conVec[i + id * N_NEURONS ] = (float) ConProb2D(xa, XCordinate(i,nbN,Cpt), ya, YCordinate(i,nbN,Cpt), Sigma[whichPop(i)] ) ; 
  }
}

///////////////////////////////////////////////////////////////////    

__global__ void KernelConProbPreFactor(float *dev_conVec,float *dev_preFactor, int lChunck, unsigned long int maxNeurons) {
  /*  COMPUTE PRE-FACTOR AND MULTIPLY zB[clm] = K / sum(conProd(row, :)) */

  unsigned long id =  (unsigned long)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long kNeuron = id + lChunck * maxNeurons ;
  unsigned long i;
  
  if(id < maxNeurons & kNeuron < NX_NEURONS) {// i-->id column to row 
    for(i=0;i<NY_NEURONS;i++) // sum over columns 
      dev_preFactor[ kNeuron + whichPop(i) * NX_NEURONS ] += (double) dev_conVec[ i + id * NY_NEURONS] ; 
  }
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelConProbNorm(float *dev_conVec, float *dev_preFactor, int lChunck, unsigned long int maxNeurons) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long int kNeuron = id + lChunck * maxNeurons;
  unsigned long int i;
  float preFactor = 0 ; 
  
  if(id < maxNeurons & kNeuron< NX_NEURONS) { 
    for(i=0;i<NY_NEURONS;i++) { // id-->i column to row, P[row][clm] = Zb[row] * C[row][clm]
      
      if(IF_SPEC) {
	if(dev_preFactor[kNeuron + whichPop(i) * NY_NEURONS] !=0)
	  preFactor =  sqrt(K) / dev_preFactor[kNeuron + whichPop(i) * NY_NEURONS] ; 
      }
      else
	if(dev_preFactor[kNeuron + whichPop(i) * NY_NEURONS] !=0)
	  preFactor = K / dev_preFactor[kNeuron + whichPop(i) * NY_NEURONS] ; 
      
      dev_conVec[i + id * NY_NEURONS] *= preFactor ; 
    }
  }
}

///////////////////////////////////////////////////////////////////    

__global__ void KernelGenConRing(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma, const double *Dij) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons ;
  unsigned long int i;
  double xa, xb;
  
  if(id < maxNeurons && kNeuron < N_NEURONS) { 
    xa = XCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    for(i=0; i < N_NEURONS; i++) { // i-->id column to row, P[row][clm] = G(X[row],X[clm],Sigma[clm])       
      xb = XCordinate(i,nbN,Cpt) ;
      // dev_conVec[id + i * maxNeurons] = (float) ( K / (float) nbN[whichPop(i)] ) * ( 1.0 + 2.0 * Sigma[ whichPop(kNeuron) + whichPop(i) * nbpop ] * cos( 2.0 * M_PI * (xa-xb) ) ) ;
      dev_conVec[i + id * N_NEURONS] = (float) ( K / (float) nbN[whichPop(i)] ) * ( 1.0 + 2.0 * Sigma[ whichPop(i) ] * Dij[ whichPop(i) + whichPop(kNeuron) * nbpop ] * cos( 2.0 * (xa-xb) ) ) ;
      // cuPrintf("id %d pop %d | i %d pop %d | idx %d Dij %.0f\n", kNeuron, whichPop(kNeuron), i, whichPop(i), whichPop(kNeuron) + whichPop(i) * nbpop, Dij[ whichPop(kNeuron) + whichPop(i) * nbpop ]) ;
           
    }
  }
}

///////////////////////////////////////////////////////////////////    

__global__ void KernelGenDistDepConMat(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons) {
  /* GENERATE CONNECTION MATRIX WITH ANOTOMIC CONNECTIVITY PROFILE */
  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long) threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long kNeuron = id + lChunck * maxNeurons, i;
  // float a = 1. ;
  
  if(id < maxNeurons && kNeuron < NX_NEURONS)
    for(i=0; i<NY_NEURONS; i++) {
      // a = 1. - (float) whichPop(i) ;
      // if(IF_SPEC) 
      // 	dev_conVec[id + i * maxNeurons] = ( K - a * sqrt(K) ) / ( (float) (N_NEURONS/nbpop) ) + a * dev_conVec[id + i * maxNeurons] ;

      if(dev_conVec[id + i * maxNeurons] >= randkernel(state, kNeuron)) 
	dev_conVec[id + i * maxNeurons] = 1. ;
      else
	dev_conVec[id + i * maxNeurons] = 0. ;       
    }  
}

#endif