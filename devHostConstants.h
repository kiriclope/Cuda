#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define N_THREADS 512

#define nbpop 4 // number of populations
#define N_NEURONS 80000ULL // total number of neurons
#define nbPref 10000
#define popSize .25 // proportion of neurons in excitatory pop
#define IF_Nk 0 // different number of neurons in each pop then fix popSize

#define K 5000.

#define L 1. // 2.0*M_PI // size of the ring

#define IF_CHUNKS 0
#define NCHUNKS 2
#define MAXNEURONS 5000ULL 

const double Sigma[4] = {10.,0.,0.,0.} ;
const double Dij[16] = {1.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,0., 0.,0.,0.,0.} ;

#define IF_RING 0 // standard ring with cosine interactions
#define IF_SPACE 0 // Gaussian spatial connections
#define IF_SPEC 0 // sqrt(K) specific connections 
#define IF_MATRIX 0 // save Cij matrix
#define IF_SPARSEVEC 1 // save sparse vectors

#endif
