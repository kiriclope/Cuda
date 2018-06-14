#ifndef _GENSPARSEVEC_
#define _GENSPARSEVEC_
void GenSparseVec(float *conVec, vector<int> &IdPost, vector<int> &nbPost,vector<unsigned long int> &idxPost ) {

  // Sparse representation :
  // IdPost // Id of the post neurons
  // nbPost // number of post neurons
  // idxPost // idx of the post neurons
  
  for(int i=0;i<nbpop;i++) 
    for(unsigned long int k=Cpt[i];k<Cpt[i+1];k++) //Presynaptic neurons
      for(int j=0;j<nbpop;j++)
	for((unsigned long int) l=Cpt[j];l<Cpt[j+1];l++) //Postsynaptic neurons
	  if(conVec[k + N_NEURONS * l]) { // k-->l
	    IdPost.push_back(l); 
	    nbPost[k]++ ;
	    nbPreSab[j][i][l-Cpt[j]]++ ;
	  }
  
  idxPost[0] = 0;
  for(i = 1; i < N_NEURONS; ++i)
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }
}
#endif
