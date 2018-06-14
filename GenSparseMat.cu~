#ifndef _GENSPARSEMAT_
#define _GENSPARSEMAT_
void GenSparseMat(float *conVec,  int rows, int clms, int* sparseVec, int* idxVec, int* nPostNeurons ) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  
  unsigned long long int i, j, counter = 0, nPost;
  if(NE == 1) {
      nPostNeurons[0] = nPost; 
      for(i = 1; i < rows; ++i) {
      nPost = 0;
      for(j = 1; j < clms; ++j) {
	if(conVec[i + clms * j]) { /* i --> j  */
	  sparseVec[counter] = j;
	  counter += 1;
	  nPost += 1;
	}
      }
      nPostNeurons[i] = nPost; 
    }
  }

  else {
    for(i = 0; i < rows; ++i) {
      nPost = 0;
      for(j = 0; j < clms; ++j) {
	if(conVec[i + clms * j]) { /* i --> j  */
	  sparseVec[counter] = j;
	  counter += 1;
	  nPost += 1;
	}
      }
      nPostNeurons[i] = nPost; 
    }
  }
  
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }
}
#endif
