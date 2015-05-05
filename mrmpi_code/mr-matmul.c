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
#define NRA 1000
#define NCA 1000
#define NCB 1000
#define  IB 10 //number of rows in A blocks
#define  KB 10 //number of columns in A blocks
#define  JB 10 //number of columns in C blocks
  /*Create submatrices for mapreduce */
  // Credit to John Norstad for Strategy 4 pseudocode
#define  NIB (NRA - 1)/IB + 1
#define  NKB (NCA -1)/KB +1
#define  NJB (NCB - 1)/JB +1 

void map(int itask, void *kv, void *ptr);
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

  // how do you set the number of mapper procs (second argument)? 
  // for word frequency example, they spawn a map task for each file

  // each map task implicitly gets a map rank 
  // all tasks get the ptr to the key/value object in the MapReduce object
  // APPptr are the arguments for your map function
  /*  uint64_t MR_map(void *MRptr, int nmap,
		  void (*mymap)(int, void *KVptr, void *APPptr),
		  void *APPptr); */
  
  //do i need to build a struct to contain both partitions of A and B for each map task?
  //use map task rank to partition 
  //map returns number of KV pairs
  nelements_in = MR_map(mr,nprocs,&map,(void *) &input);
  // MR_collate(mr,NULL); //second argument is hashing function
  //nelements_out = MR_reduce(mr,&reduce,NULL);
  // nunique = MR_reduce(mr,&sum,NULL);

  MPI_Barrier(MPI_COMM_WORLD);
  tstop = MPI_Wtime();
  //kyle: no idea what is going on below
  /*  MR_sort_values(mr,&ncompare);

  count.n = 0;
  count.limit = 10;
  count.flag = 0;
  MR_map_mr(mr,mr,&output,&count);
  
  MR_gather(mr,1);
  MR_sort_values(mr,&ncompare);

  count.n = 0;
  count.limit = 10;
  count.flag = 1;
  MR_map_mr(mr,mr,&output,&count);
  */
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
  int key_array[4];
  Matrices *input = (Matrices *) ptr; 
  double **A,**B;
  A = input->M1; 
  B = input->M2; 
  int nmap = input->ntasks; 
  int i,j,k;
  int row_index,column_index;
  for (i=0; i<IB; i++){ //read A block rows
    row_index = itask*i; 
    for (k=0; k< KB; k++){
      column_index = itask*k; 
      for (j=0; j<NJB; j++){ //create KV for all blocks that need it
	//key consists of four ints, value consists of one double
	//PACK KEY ARRAY-- DO I NEED A SEPARATE ONE FOR EACH KV??
	key_array[0] = i/IB;
	key_array[1] = j;
	key_array[2] = k/KB; 
	key_array[3] = 0; //0 indicates A 
	MR_kv_add(kv,(char *)key_array,4*sizeof(int),(char *)&(A[i][k]),sizeof(double));
      }
    }
  }
}
