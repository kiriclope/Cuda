#ifndef _NEURON_COUNTS 
#define _NEURON_COUNTS 

#define N_THREADS 512

#define nbpop 1 // number of populations
#define N_NEURONS 10000ULL //43200ULL // total number of neurons

//#define N_NEURONS 76800ULL //43200ULL // total number of neurons
//#define NX_NEURONS 10000ULL // presynaptic Neurons 
//#define NY_NEURONS 10000ULL // postsynaptic Neurons, NX must be greater than NY

#define nbPref 10000.0
#define popSize 1.0 //.75 // proportion of neurons in excitatory pop
#define IF_Nk 0 // different number of neurons in each pop then fix popSize

#define IF_LARGE 0
const char* AtoB = "IE" ;

#define K 500. 

#define IF_CHUNKS 0 
#define NCHUNKS 4 
#define MAXNEURONS 10000ULL 

#define IF_AUTA 0
#define AUTA_Pop 0
#define AUTA_Pb 1.
const double AUTA_p[4] = {1., 1., 0., 0.} ; 

const double Sigma[4] = {.5, .25, .375, .25} ; 
const double Dij[16] = {1.,1.,1.,1., 0.,0.,0.,0., 0.,0.,0,0., 0.,0.,0.,0.} ; 

#define L M_PI // 2.0*M_PI // size of the ring
#define IF_RING 0 // standard ring with cosine interactions
#define IF_SPACE 0 // Gaussian spatial connections
#define DIMENSION 1 // Dimension of the ring
#define IF_SPEC 0 // sqrt(K) specific connections 
#define IF_MATRIX 0 // save Cij matrix
#define IF_SPARSEVEC 1 // save sparse vectors

#endif
