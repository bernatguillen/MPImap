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
  ireq = (MPI_Request *) malloc(3*nprocs);
  if(nprocs < 2 || argc != 2){
    if (myid == MASTER)
      printf("usage: montefault Nsamples \n NRA: number of points to sample\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  /* Dynamically allocate vectors */
  Nsamples = atoi(argv[1]);
  Nodesamp = ((int)ceil(((double)Nsamples)/nprocs));
  Nsamples = Nodesamp*nprocs;
  double x[2];
  int resf=0;
  double time_s, time_f;
  int res = 0;
  srand(time(NULL));
  if(myid == MASTER){
    time_s = MPI_Wtime();
    for(int i = 0; i<Nodesamp; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      resf += rhs(x) < 1 ? 1:0;
    }
    int count = 0;
    int loops = 0;
    int dest = 1;
    int rdy = 1;
    int stp = 0;
    int *done;
    done = (int*) malloc(nprocs);
    int *waiting;
    waiting = (int*) malloc(nprocs);
    double *t_w;
    t_w = (double*) malloc(nprocs);
    for(int i = 0; i<nprocs; ++i){
      done[i]  = 0;
      waiting[i] = 0;
      t_w[i] = 0.0;
    }
    while(count<nworkers){
      if(!loops){
	MPI_Isend(&rdy, 1, MPI_INT, dest, FROM_MASTER,MPI_COMM_WORLD,&ireq[dest]);
	t_w[dest] = MPI_Wtime();
      }
      if(!done[dest] && MPI_Wtime() - t_w[dest] >= waiting[i]){
	MPI_Test(ireq+dest,&done[dest],&stat);
	if(!done[dest]){
	  ++ waiting[dest];
	  t_w[dest] = MPI_Wtime();
	  if(waiting[dest] == 9){
	    MPI_Isend(&stp, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, &ireq[dest]);
	    for(int i = 0; i<Nodesamp;++i){
	      x[0] = ((double) rand())/RAND_MAX*2 - 1;
	      x[1] = ((double) rand())/RAND_MAX*2 - 1;
	      resf += rhs(x) < 1 ? 1:0;
	      ++count;
	      done[dest] = 1;
	    }
	  }
	}else{
	  MPI_Send(&rdy, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
	  MPI_Recv(&res, 1, MPI_INT, dest, FROM_WORKER, MPI_COMM_WORLD, &stat);
	  resf+=res;
	  ++count;
	  done[dest] = 1;
	}
      }
      dest = (dest == nworkers ? 1:dest+1);
      if(dest == 1 & !loops) loops = 1;
    }
    for(dest = 1; dest<nprocs; ++dest){
      MPI_Isend(&stp, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, ireq+dest);
    }
    time_f = MPI_Wtime();
    printf("Time elapsed: %lf \n", time_f-time_s);
    printf("Estimation of pi: %lf \n", 4*((double)resf)/Nsamples);
    MPI_Finalize();
  }else{
    for(int i = 0; i<Nodesamp; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      res += rhs(x) < 1 ? 1:0;
    }
    int msg;
    MPI_Recv(&msg, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &stat);
    MPI_Recv(&msg, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &stat);
    if(msg == 1){
      MPI_Send(&res, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }else{
      MPI_Finalize();
    }
    while(msg == 1){
      MPI_Recv(&msg, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &stat);
      if(msg==1){
	res = 0;
	for(int i = 0; i<Nodesamp; ++i){
	  x[0] = ((double) rand())/RAND_MAX*2 - 1;
	  x[1] = ((double) rand())/RAND_MAX*2 - 1;
	  res += rhs(x) < 1 ? 1:0;
	}
	MPI_Send(&res, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
      }
    }
    MPI_Finalize();
      
  }
  return(0);
}
