#ifndef _NEURON_COUNTS
#define _NEURON_COUNTS

#define N_THREADS 512

#define nbpop 2 // number of populations
#define N_NEURONS 60000ULL // total number of neurons
#define popSize .5 // proportion of neurons in excitatory pop
#define IF_Nk 0 // different number of neurons in each pop then fix popSize


#define K 250.
#define L 1. // 2.0*M_PI // size of the ring

#define IF_CHUNKS 0
#define NCHUNKS 2
#define MAXNEURONS 5000ULL 

const double Sigma[4] = {5.,0.,.25,.25} ;


#define IF_RING 1 // standard ring with cosine interactions
#define IF_SPACE 0 // Gaussian spatial connections
#define IF_SPEC 1 // sqrt(K) specific connections 
#define IF_MATRIX 1 // save Cij matrix

#endif
