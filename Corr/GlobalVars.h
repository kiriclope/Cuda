#define BATCH 1
#define N_THREADS 512

#define nbpop 2
const char* model = "Binary" ;
const char* dir = "Test" ;

#define N_NEURONS 20000UL
#define nbPref 10000UL
#define popSize 0.5
#define K 500.
#define g 1.

#define AUTOCORR 1
#define IF_RING 0
#define IF_GAUSS 0
const double Sigma[4] = {.5,.0625,.25,.125} ;

#define IF_AUTA 1 
#define AUTA_Pop 0
#define AUTA_Pb 1.
const double AUTA_p[4] = {0, 0, 0., 0.} ; 