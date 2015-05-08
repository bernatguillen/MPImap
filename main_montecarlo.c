#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <assert.h>
#include <math.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

double rhs(double *x){
  return(x[0]*x[0]+x[1]*x[1]);
}

int main(int argc, char **argv){
  int nprocs, myid, nworkers, source, dest, mtype, averow, extra, i,j, k, rc;
  int Nsamples, Nodesamp;
  MPI_Status status;
  MPI_Init(&argc,&argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); //Does not immediately abort
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  nworkers = nprocs - 1;
  MPI_Status stat;
  MPI_Request *ireq;
  ireq = (MPI_Request *) malloc(nprocs);
  if(nprocs < 2 || argc != 2){
    if (myid == MASTER)
      printf("usage: main_matmul Nsamples \n NRA: number of points to sample\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  /* Dynamically allocate vectors */
  Nsamples = atoi(argv[1]);
  Nodesamp = ((int)ceil(((double)Nsamples)/nprocs));
  Nsamples = Nodesamp*nprocs;
  double x[2];
  int resf;
  double time_s, time_f;
  int res = 0;
  srand(time(NULL));
  if(myid == MASTER){
    time_s = MPI_Wtime();
    for(i = 0; i<Nodesamp; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      res += rhs(x) < 1 ? 1:0;
    }
    MPI_Reduce(&res,&resf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
    time_f = MPI_Wtime();
    printf("Time elapsed: %lf \n", time_f-time_s);
    printf("Estimation of pi: %lf \n", 4*((double)resf)/Nsamples);
  }else{
    for(i = 0; i<Nodesamp; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      res += rhs(x) < 1 ? 1:0;
    }
    MPI_Reduce(&res,&resf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
  }
   MPI_Finalize();
  return(0);
}
