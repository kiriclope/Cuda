#ifndef _NEURON_COUNTS 
#define _NEURON_COUNTS 

#define N_THREADS 512 

#define nbpop 4 // number of populations 
#define N_NEURONS 80000ULL //43200ULL // total number of neurons

/* #define N_NEURONS 76800ULL //43200ULL // total number of neurons */
//#define NX_NEURONS 10000ULL // presynaptic Neurons 
//#define NY_NEURONS 10000ULL // postsynaptic Neurons, NX must be greater than NY 

#define nbPref 10000.0

#define popSize .25 // .75 // proportion of neurons in excitatory pop 
#define IF_Nk 0 // different number of neurons in each pop then fix popSize 

#define IF_LARGE 0
const char* AtoB = "IE" ; 

#define K 500. 

#define IF_CHUNKS 0
#define NCHUNKS 5
/* #define MAXNEURONS 10000ULL  */
#define MAXNEURONS 15360ULL

#define IF_SHARED 0
#define PROP_SHARED 0.0
#define CLUSTER_SIZE 0.0

#define IF_AUTA 0 
#define AUTA_Pop 0
#define AUTA_Pb 0.0
const double AUTA_p[4] = {1., 1., 0., 0.} ; 

/* const double Sigma[4] = {.25, .125, 10., .25} ; */
const double Sigma[4] = {.15, .12, .15, .12} ; 

const double Dij[16] = {0.,0.,1.,0., 0.,0.,0.,0., 0.,0.,0,0., 0.,0.,0.,0.} ; 

#define L 1. // 2.0*M_PI // size of the ring 
#define IF_RING 0 // standard ring with cosine interactions
#define IF_SPACE 0 // Gaussian spatial connections
#define DIMENSION 2 // Dimension of the ring
#define IF_SPEC 0 // sqrt(K) specific connections 
#define IF_MATRIX 0 // save Cij matrix 
#define IF_SPARSEVEC 1 // save sparse vectors

#endif
