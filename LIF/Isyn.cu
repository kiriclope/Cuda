#ifndef _ISYNAP_
#define _ISYNAP_
#include  <cuda.h>
#include "globalVars.h"
#include "devHostConstants.h"
#include "devFunctionProtos.h"
#define MAX_SPKS_PER_T_STEP 1000

__device__ double isynap(double vm, int *dev_conVec) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int i, spkNeuronId[MAX_SPKS_PER_T_STEP], localNSpks = 0;
  double totIsynap = 0, gE, gI, tempCurE = 0, tempCurI = 0;
  /* compute squares of entries in data array */
  /*!!!!! neurons ids start from ZERO  !!!!!! */
  if(mNeuron < N_NEURONS) {
    gE = dev_gE[mNeuron];
    gI = dev_gI[mNeuron];
    gE *= EXP_SUM_E;
    gI *= EXP_SUM_I;
    for(i = 0; i < N_NEURONS; ++i) {
      if(dev_IF_SPK[i]) { /* too many global reads */
        spkNeuronId[localNSpks] = i; 
        localNSpks += 1; /* nspks in prev step*/
      }
    }
    if(localNSpks > 0){
      for(i = 0; i < localNSpks; ++i) { 
        if(spkNeuronId[i] < NE) {
          gE += dev_conVec[spkNeuronId[i] + N_NEURONS * mNeuron];
        }
        else {
          gI += dev_conVec[spkNeuronId[i] + N_NEURONS * mNeuron]; /*optimize !!!! gEI_I*/
        }
      }
    }
    dev_gE[mNeuron] = gE;
    dev_gI[mNeuron] = gI;
    if(mNeuron < NE) {
      tempCurE = -1 *  gE * (1/sqrt(K)) * INV_TAU_SYNAP_E * G_EE
                          * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP_I * G_EI
                          * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    if(mNeuron >= NE){
      tempCurE = -1 * gE * (1/sqrt(K)) * INV_TAU_SYNAP_E * G_IE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * gI * (1/sqrt(K)) * INV_TAU_SYNAP_I * G_II * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    totIsynap = tempCurE + tempCurI; 
  }
  return totIsynap;
}

__global__ void expDecay() {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  if(mNeuron < N_NEURONS) {
    dev_gE[mNeuron] *= EXP_SUM_E;
    dev_gI[mNeuron] *= EXP_SUM_I;
  }
}

__global__ void expDecay(int *dev_histCountE, int *dev_histCountI) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  while(mNeuron < N_NEURONS) {
    // dev_gE[mNeuron] *= EXP_SUM_E;
    // dev_gI[mNeuron] *= EXP_SUM_I;
    if(mNeuron < NE) {
      dev_gE[mNeuron] *= EXP_SUM_EE;      
      dev_gI[mNeuron] *= EXP_SUM_I;
    }
    else {
      dev_gE[mNeuron] *= EXP_SUM_E;      
      dev_gI[mNeuron] *= EXP_SUM_II;
    }
    /*    if(mNeuron == 0) {
      for(int i = 0; i < N_NEURONS; ++i) {*/
    dev_histCountE[mNeuron] = 0;
    dev_histCountI[mNeuron] = 0;
        /*      }
    }*/
    mNeuron += stride;
  }
}


__global__ void computeConductanceHist(int *dev_histCountE, int *dev_histCountI) {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  while(mNeuron < N_NEURONS) {
      dev_gE[mNeuron] += (double)dev_histCountE[mNeuron];
      dev_gI[mNeuron] += (double)dev_histCountI[mNeuron];
      mNeuron += stride;
  }
}     

__global__ void computeConductance() {
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  int kNeuron;
  if(mNeuron < N_NEURONS) {
     if(dev_IF_SPK[mNeuron]) {  
      for(kNeuron = 0; kNeuron < dev_nPostNeurons[mNeuron]; ++kNeuron) { 
        if(mNeuron < NE) {       
          atomicAdd(&dev_gE[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron]], (double)1.0); /*atomic double add WORKS ONLY ON CC >= 2.0 */
       }
        else
          atomicAdd(&dev_gI[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron]], (double)1.0);
      }
     } 
  }
}

// __global__ void computeG_Optimal() {
//   unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
//   int kNeuron, localSpkId;
//   if(mNeuron < N_NEURONS) {
//     if(dev_IF_SPK[mNeuron]) {  
//       localSpkId = dev_prevStepSpkIdx[mNeuron];
//       for(kNeuron = 0; kNeuron < dev_nPostNeurons[mNeuron]; ++kNeuron) { 
//         if(mNeuron < NE) {       
// 	  dev_ESpkCountMat[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron] + N_NEURONS + localSpkId] += 1;
// 	}
//         else{
// 	  dev_ISpkCountMat[dev_sparseConVec[dev_sparseIdx[mNeuron] + kNeuron] + N_NEURONS + localSpkId] += 1;
// 	}
//       }
//     }
//   }
// }


// __global__ void spkSum(int nSpksInPrevStep) {
//   unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
//   int i, gE, gI; 
//   int stride = gridDim.x * blockDim.x;
//   while(mNeuron < N_NEURONS) {
//     gE = 0;
//     gI = 0;
//     for(i = 0; i < nSpksInPrevStep ; ++i){
//       gE += dev_ESpkCountMat[mNeuron + i * N_NEURONS];
//       gI += dev_ISpkCountMat[mNeuron + i * N_NEURONS];
//     }	
//     dev_gE[mNeuron] += (double)gE;
//     dev_gI[mNeuron] += (double)gI;
//     mNeuron += stride;
//   }
// }

__global__ void computeIsynap(double t, devPtr_t devPtrs) {
  // THIS IS THE FUNCTION BEING USED CURRENTLY
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;
  double vm, tempCurE = 0, tempCurI = 0;
  int localCurConter;
  if(mNeuron < N_NEURONS) {
    vm = dev_v[mNeuron];
    if(mNeuron < NE) {
      //tempCurE = -1 * dev_gE[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_E * G_EE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurE = -1 * dev_gE[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_EE * G_EE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      tempCurI = -1 * dev_gI[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_I * G_EI * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
    }
    if(mNeuron >= NE){
      tempCurE = -1 * dev_gE[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_E * G_IE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));
      //    tempCurE = -1 * dev_gE[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_IE * G_IE * (RHO * (vm - V_E) + (1 - RHO) * (E_L - V_E));      
      //      tempCurI = -1 * dev_gI[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_I * G_II * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));
      tempCurI = -1 * dev_gI[mNeuron] * (1/sqrt(K)) * INV_TAU_SYNAP_II * G_II * (RHO * (vm - V_I) + (1 - RHO) * (E_L - V_I));

    }
    dev_isynap[mNeuron] = tempCurE + tempCurI;

    if(t > N_I_AVGCUR_STORE_START_TIME & mNeuron >= NE & mNeuron < NE + N_I_2BLOCK_NA_CURRENT)    {
      dev_totalAvgEcurrent2I[mNeuron - NE] += tempCurE / ((TSTOP / DT) - (TSTOP - N_I_AVGCUR_STORE_START_TIME)/DT);
      dev_totalAvgIcurrent2I[mNeuron - NE] += tempCurI / ((TSTOP / DT) - (TSTOP - N_I_AVGCUR_STORE_START_TIME)/DT);
    }
    
    // if(mNeuron == SAVE_CURRENT_FOR_NEURON) {
      // localCurConter = curConter;
      // if(localCurConter < N_CURRENT_STEPS_TO_STORE) {
      // if(curConter < N_CURRENT_STEPS_TO_STORE) {	
	// glbCurE[localCurConter] = tempCurE;
	// glbCurI[localCurConter] = tempCurI;
	// STORE CONDUCTANCES
	// curConter += 1;
      // }
    // }

      if(t > DISCARDTIME) {
        devPtrs.dev_conductanceFF[mNeuron] += gFF[mNeuron];
	if(mNeuron < NE) {
	  devPtrs.dev_conductanceE[mNeuron] += dev_gE[mNeuron] * (1/sqrt(K)) * G_EE ;
	  devPtrs.dev_conductanceI[mNeuron] += dev_gI[mNeuron] * (1/sqrt(K)) * G_EI ;
	}
	else {
	  devPtrs.dev_conductanceE[mNeuron] += dev_gE[mNeuron] * (1/sqrt(K)) * G_IE ;
	  devPtrs.dev_conductanceI[mNeuron] += dev_gI[mNeuron] * (1/sqrt(K)) * G_II ;
	}
      }

    
    	/* bg current */
	/*	ibg = bgCur(vmOld); /* make sure AuxRffTotal<<<  >>> is run begore calling bgCur */
	/* FF input current*/
    RffTotal(t);
    Gff(t);
    dev_iffCurrent[mNeuron] = IFF(vm);
    
  }
}
#endif
