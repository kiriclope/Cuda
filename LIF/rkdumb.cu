#ifndef _RKDUMB_
#define _RKDUMB_ 
#include <cuda.h>
#include "globalVars.h"
#include "devFunctionProtos.h"

__global__ void rkdumb(kernelParams_t params, devPtr_t devPtrs) { 
  double x, xInit, IsynNew=0, *dev_spkTimes, *y,  *Isyn, *dev_time;
  int t, nstep, *totNSpks, *dev_spkNeuronIds;
  curandState *dev_state;
  
  double RK1=0,RK2=0,RK3=0,RK4=0 ;
  double Volt, VoltNew ;
  unsigned int localTotNspks = 0, localLastNSteps = 0;
  unsigned int mNeuron = threadIdx.x + blockDim.x * blockIdx.x;

  xInit = params.tStart ; 
  nstep = params.nSteps ; 
  totNSpks = devPtrs.dev_nSpks ; 
  y = devPtrs.dev_Volt ; 
  dev_time = devPtrs.dev_time ; 
  synapticCurrent = devPtrs.synapticCurrent ; 
  dev_state = devPtrs.devStates ; 
  dev_spkTimes = devPtrs.dev_spkTimes ; 
  dev_spkNeuronIds = devPtrs.dev_spkNeuronIds ; 
  t = devPtrs.t ; 

  if(mNeuron < N_NEURONS) { 
    if(t==0) { 
      dev_Volt[mNeuron] = (-1.0 * 70.0) +  (40.0 * randkernel(dev_state)) ; /* Vm(0) ~ U(-70, -30)*/ 
      dev_Isyn[mNeuron] = 0 ; 
      dev_sparseConVec = devPtrs.dev_sparseConVec ; 
    } 
    localLastNSteps = nstep - STORE_LAST_N_STEPS ; 

    /* TIMELOOP */ 
    x = xInit + (double)t * DT ; 
    dev_IF_SPK[mNeuron] = 0 ; 
    Volt = dev_Volt[mNeuron] ; 
    Isyn = dev_Isyn[mNeuron] ; 
    
    /* runge kutta 4 */
    RK1 = -(Volt-Vr)/Tm[whichPop[mNeuron]] + Isyn ; 
    RK2 = -(Volt-Vr + DT/2.*RK1)/Tm[whichPop[mNeuron]] + Isyn ; 
    RK3 = -(Volt-Vr + DT/2.*RK2)/Tm[whichPop[mNeuron]] + Isyn ; 
    RK4 = -(Volt-Vr + DT*RK3)/Tm[whichPop[mNeuron]] + Isyn ; 
    VoltNew = Volt + DT/6.*(RK1 + 2.*RK2 + 2.*RK3 + RK4) ; 
    
    x += DT ; 
    dev_Volt[mNeuron] = VoltNew ; 
    
    if(t >= localLastNSteps & (mNeuron >= N_NEURONS_TO_STORE_START &  mNeuron < N_NEURONS_TO_STORE_END)) { 
      y[(mNeuron - N_NEURONS_TO_STORE_START) + N_NEURONS_TO_STORE * (t - localLastNSteps)] = VoltNew ; 
      if(mNeuron == 0) 
        dev_time[t - localLastNSteps] = x ; 
    }
    
    if(t>2) 
      if(VoltNew > SPK_THRESH) 
	if(Volt <= SPK_THRESH) {
	  dev_IF_SPK[mNeuron] = 1 ; 
	  localTotNspks = atomicAdd(totNSpks, 1) ; /* atomic add on global introduces memory latency*/
	  if(localTotNspks + 1 < MAX_SPKS) { 
	    dev_spkNeuronIds[localTotNspks + 1] = mNeuron ; 
	    dev_spkTimes[localTotNspks + 1] = x ; 
	  }
	} 
  }
}

#endif