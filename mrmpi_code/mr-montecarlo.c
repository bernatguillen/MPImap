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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"
#include "string.h"
#include "sys/stat.h"
#include "cmapreduce.h"

#define MASTER 0
#define NSAMPLES 1E6

void map(int itask, void *kv, void *ptr);
void reduce(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr);

double rhs(double *x){
  return(x[0]*x[0]+x[1]*x[1]);
}

/* ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  int me,nprocs;
  double tstart,tstop;
  int nelements_in,nelements_out;
  int i,j,k;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); //Does not immediately abort

  /*Create MapReduce object */
  void *mr = MR_create(MPI_COMM_WORLD);
  MR_set_verbosity(mr,2);
  MR_set_timer(mr,1);

  tstart = MPI_Wtime();
  time_t t;
  srand((unsigned) time(&t));

  int ntotal, nlocal;
  nlocal = ((int)ceil(((double)NSAMPLES)/nprocs));
  ntotal = nlocal*nprocs;

  nelements_in = MR_map(mr,nprocs,&map,&nlocal); 
  /* Redistribute all the block KVs to consistent processors */
  /* Turn the KV into a KVM since they all have the same key */
  MR_collate(mr, NULL); //both aggregates and converts to KVM
  nelements_out = MR_reduce(mr,&reduce,NULL);
  printf("nelements_in = %d \n", (int) nelements_in); 
  printf("nelements_out = %d \n", (int) nelements_out); 
  MPI_Barrier(MPI_COMM_WORLD);
  tstop = MPI_Wtime();
  MR_destroy(mr);
  if (me == 0) {
    printf("Time = %g (secs)\n",tstop-tstart);
  }
  MPI_Finalize();
  return(0);
}

  
void map(int itask, void *kv, void *ptr)
{
  //void MR_kv_add(void *KVptr, char *key, int keybytes,char *value, int valuebytes);
  int key=1;
  int value;
  int i;
  int nlocal = *((int *) ptr); 
  double x[2];
  for(i = 0; i<nlocal; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      value = rhs(x) < 1 ? 1:0;
      MR_kv_add(kv,(char *)&key,sizeof(int),(char *) &value,sizeof(int));
  }
  return;
}

//must take the following arguments
void reduce(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr){
  int i;
  int new_key =1;
  int total_hits =0;
  int *hits = (int *)multivalue;
  for (i=0; i<nvalues; i++){
    total_hits += multivalue[i];
  }
  MR_kv_add(kv,(char *) &new_key,sizeof(int),(char *) &total_hits,sizeof(int));

} 
