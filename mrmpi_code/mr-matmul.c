/* ----------------------------------------------------------------------
   MR-MPI = MapReduce-MPI library
   http://www.cs.sandia.gov/~sjplimp/mapreduce.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the modified Berkeley Software Distribution (BSD) License.

   See the README file in the top-level MapReduce directory.
------------------------------------------------------------------------- */

/*
MapReduce block matrix multiply example in C
Syntax: NRA NCA NCB
(1) generate random matrices A and B
(2) 
(3) 
*/
	//RESTRICT TO SQUARE MATRICES RIGHT NOW
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sys/stat.h"
#include "cmapreduce.h"

#define MASTER 0
	/*For now, hardcode size of matrices and submatrices */
#define NRA 100
#define NCA 100
#define NCB 100
#define  IB 10 //number of rows in A blocks
#define  KB 10 //number of columns in A blocks
#define  JB 10 //number of columns in C blocks
  /*Create submatrices for mapreduce */
  // Credit to John Norstad for Strategy 4 pseudocode
#define  NIB (NRA - 1)/IB + 1
#define  NKB (NCA -1)/KB +1
#define  NJB (NCB - 1)/JB +1 

void map(int itask, void *kv, void *ptr);
int partition(char *key, int keybytes);
void reduce(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr);

int initializeMatrix(double **M, int NR,int NC){
  time_t t;
  srand((unsigned) time(&t));
  for(int i = 0; i<NR; ++i){
    for(int j = 0; j<NC; ++j){
      M[i][j] = rand();
    }
  }
  return(0);
}

typedef struct {
  double **M1,**M2;
  int ntasks;
} Matrices; 

/* ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  int me,nprocs;
  double tstart,tstop;
  Matrices input;
  int nelements_in,nelements_out;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  /*  if(nprocs < 2 || argc != 4){
    if (me == MASTER) 
      printf("usage: main_matmul NRA NCA NCB \n NRA: number of rows in A \n NCA: number of columns in A \n NCB: number of rows in B \n (NRB == NCA) \n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
    } */
  /* Dynamically allocate matrices */
  double **A,**B,**C;
  double *dataA,*dataB,*dataC;
  //make contiguous multiarrays                                                                                
  A = (double **) malloc(sizeof(double *)*NRA);
  dataA = (double *) malloc(sizeof(double)*NRA*NCA);
  B = (double **) malloc(sizeof(double *)*NCA);
  dataB = (double *) malloc(sizeof(double)*NCA*NCB);
  C = (double **) malloc(sizeof(double *)*NRA);
  dataC = (double *) malloc(sizeof(double)*NRA*NCB);
  for(int i=0; i<NRA; i++){
    A[i] = &(dataA[NCA*i]);
  }
  for(int i=0; i<NCA; i++){
    B[i] = &(dataB[NCB*i]);
  }
  for(int i=0; i<NRA; i++){
    C[i] = &(dataC[NCB*i]);
  }
  initializeMatrix(A, NRA, NCA);
  initializeMatrix(B, NCA, NCB);
  //double check struct notation
  input.M1 = A; 
  input.M2 = B;
  input.ntasks = nprocs;

  /*Create MapReduce object */
  void *mr = MR_create(MPI_COMM_WORLD);
  // not sure these are documented in manual:
  MR_set_verbosity(mr,2);
  MR_set_timer(mr,1);

  MPI_Barrier(MPI_COMM_WORLD);
  tstart = MPI_Wtime();

  // each map task implicitly gets a map rank 
  // all tasks get the ptr to the key/value object in the MapReduce object
  // APPptr are the arguments for your map function
  /*  uint64_t MR_map(void *MRptr, int nmap,
		  void (*mymap)(int, void *KVptr, void *APPptr),
		  void *APPptr); */
  //use map task rank to partition 
  //map returns number of KV pairs
  nelements_in = MR_map(mr,nprocs,&map,(void *) &input);
  /* Redistribute all the block KVs to consistent processors */
  /* Turn the KV into a KVM since they all have the same key */
  MR_collate(mr, &partition); //both aggregates and converts
  nelements_out = MR_reduce(mr,&reduce,NULL);
  printf("nelements_in = %d \n", (int) nelements_in); 
  
  //unclear if we are getting the correct net total of KV pairs
  // for 100x100 matrices with 10 row, 10 column blocks
  // = 50K pairs
  MPI_Barrier(MPI_COMM_WORLD);
  tstop = MPI_Wtime();

  //debugging KVs
  //void MR_print(void *MRptr, int proc, int nstride, int kflag, int vflag ); nproc <0 implies all print
  MR_print(mr,0,1,1,4); //wont print full key_array, value_array

  MR_destroy(mr);

  if (me == 0) {
    printf("Time to mapreduce = %g (secs)\n",tstop-tstart);
  }
  MPI_Finalize();
}

/* ----------------------------------------------------------------------
   read a file
   for each word in file, emit key = word, value = NULL
------------------------------------------------------------------------- */
void map(int itask, void *kv, void *ptr)
{
  //have itask read from (NRA/ntasks)*itask to (NRA/ntasks)*(itask+1) -1 rows and cols
  //from both A and B
  //void MR_kv_add(void *KVptr, char *key, int keybytes,char *value, int valuebytes);
  int key_array[4]; 	//key consists of four ints, value consists of one double
  double value_array[3]; //2 ints and a double, but we will typecast for now
  Matrices *input = (Matrices *) ptr; 
  double **A,**B;
  A = input->M1; 
  B = input->M2; 
  int nmap = input->ntasks; 
  int i,j,k;
  int row_index,column_index;
  //  printf("itask = %d ntasks = %d\n",itask,nmap); //this prints correct answer
  //partition matrix by nmap, not by blocks. EMIT BY BLOCKS

  int num_rows = NRA/nmap; //assert that this has no remainder....

  for (i=0; i<num_rows; i++){ //read A block rows
    row_index = (num_rows*itask)+i; 
    for (k=0; k< num_rows; k++){
      column_index = (num_rows*itask)+k; 
      for (j=0; j<NJB; j++){ //create KV for all blocks that need it
	//PACK KEY ARRAY-- DO I NEED A SEPARATE ONE FOR EACH KV??
	key_array[0] = row_index/IB;
	key_array[1] = j;
	key_array[2] = column_index/KB; 
	key_array[3] = 0; //0 indicates A
	value_array[0] = (double) (row_index % IB);
	value_array[1] = (double) (column_index % KB);
	value_array[2] = A[row_index][column_index];
	MR_kv_add(kv,(char *)key_array,4*sizeof(int),(char *)value_array,3*sizeof(double));
      }
    }
  }
  for (k=0; k<num_rows; k++){ //read B block rows
    row_index = (num_rows*itask)+k; 
    for (j=0; j< num_rows; j++){
      column_index = (num_rows*itask)+j; 
      for (i=0; i<NIB; i++){ //create KV for all blocks that need it     
	key_array[0] = i;
	key_array[1] = column_index/JB; 
	key_array[2] = row_index/KB;
	key_array[3] = 1; //1 indicates B
	value_array[0] = (double) (row_index % KB);
	value_array[1] = (double) (column_index % JB);
	value_array[2] = B[row_index][column_index];
	MR_kv_add(kv,(char *)key_array,4*sizeof(int),(char *)value_array,3*sizeof(double));
      }
    }
  }
}
//must take the following arguments
void reduce(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr){
  int *key_array = (int *) key;
  double *value_array = (double *) multivalue; 
  int ib,jb,kb,matrix_indicator;
  int i,j;

  double **A,**C;
  double *dataA,*dataC;
  //Temporarily store blocks
  A = (double **) malloc(sizeof(double *)*IB);
  dataA = (double *) malloc(sizeof(double)*IB*KB);
  C = (double **) malloc(sizeof(double *)*IB);
  dataC = (double *) malloc(sizeof(double)*IB*JB);
  for(i=0; i<IB; i++){
    A[i] = &(dataA[KB*i]);
  }
  for(i=0; i<IB; i++){
    C[i] = &(dataC[JB*i]);
  }
  // exploit the fact that the block values are ordered
  int sib = -1;
  int sjb = -1;
  int skb = -1;
  
  int jbase,ibase;
  ib = key_array[0];
  jb = key_array[1];
  kb = key_array[2];
  matrix_indicator = key_array[3];

  int row,col;

  if ((ib != sib) || (jb != sjb)){
    if (sib != -1){ //Emit the last completed C block
      ibase = sib*IB;
      jbase =sjb*JB;
      for (i=0; i<IB; i++){
	for (j=0; j<JB; j++){
	  //	  v = C[i][j];
	  //emit C key
	}
      }
    }
    sib = ib;
    sjb =jb;
    skb = -1;
    /* Reset matrix C */
    for (i=0; i<IB; i++){
      for (j=0; j<JB; j++){
	C[i][j] = 0.0;
      }
    }     
  }			
  if (!matrix_indicator){ //A matrix
    skb = kb;
    /* Reset matrix A */
    for (i=0; i<IB; i++){
      for (j=0; j<KB; j++){
	A[i][j] = 0.0;
      }
    }
    //since this is a KMV structure, we can loop over the values
    for (i=0; i<nvalues; i++){
      row = (int) value_array[3*i];
      col = (int) value_array[3*i +1];
      A[row][col] = value_array[3*i+2];
    }
  }
  else {
    if( kb != skb) return;
    for (i=0; i<nvalues; i++){
      row = (int) value_array[3*i];
      col = (int) value_array[3*i +1];
      for (j=0; j<IB; j++)
	C[j][col] = A[j][row]*value_array[3*i+2];
    }
  }
} 

int partition(char *key, int keybytes){
  int *key_array = (int *) key;
  int ib,jb,kb,iproc;
  ib = key_array[0];
  jb = key_array[1];
  kb = key_array[2];

  iproc = ((ib*JB + jb)*KB + kb); //mr-mpi should automatically mod p this
  return(iproc);
}
