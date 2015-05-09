#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_MASTER_RDY 3
#define FROM_WORKER 2
#define FROM_WORKER_RDY 4
double rhs(double *x){
  return(x[0]*x[0]+x[1]*x[1]);
}

int main(int argc, char **argv){
  int nprocs, myid, nworkers, rc;
  int Nsamples, Nodesamp;
  MPI_Status stat;
  rc =1 ;
  MPI_Init(&argc,&argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); //Does not immediately abort
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  nworkers = nprocs - 1;
  if(nprocs < 2 || argc != 3){
    if (myid == MASTER)
      printf("usage: montefault Nsamples pfailure \n NRA: number of points to sample\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  /* Dynamically allocate vectors */
  Nsamples = atoi(argv[1]);
  double p_fail = atof(argv[2]);
  Nodesamp = ((int)ceil(((double)Nsamples)/nprocs));
  Nsamples = Nodesamp*nprocs;
  double x[2];
  int resf=0;
  double time_s, time_f;
  double time_e=1.0;
  int res = 0;
  srand(time(NULL)+getpid()+myid*nprocs);
  if(myid == MASTER){
    time_s = MPI_Wtime();
    int i=0;
    /*    printf("PID %d waiting\n", getpid());
    fflush(stdout);
    while(i==0){
      sleep(10);
      }*/
    for(i = 0; i<Nodesamp; ++i){
      x[0] = ((double) rand())/RAND_MAX*2 - 1;
      x[1] = ((double) rand())/RAND_MAX*2 - 1;
      resf += rhs(x) < 1 ? 1:0;
    }
    int dest;
    int count = 0;
    int loops = 0;
    MPI_Request *ireq;
    ireq = (MPI_Request *) malloc(nprocs);
    for(i = 0; i<nprocs; ++i){
      ireq[i] = MPI_REQUEST_NULL;
    }

    dest = 1;
    int rdy = 1;
    int rcv = 0;
    int stp = 0;
    int *failed;
    failed = (int *) malloc(nprocs*sizeof(int));
    int *done;
    done = (int *) malloc(nprocs*sizeof(int));
    if(!done) MPI_Abort(MPI_COMM_WORLD, rc);
    int *waiting;
    waiting = (int *) malloc(nprocs*sizeof(int));
    if(!waiting) MPI_Abort(MPI_COMM_WORLD, rc);
    double *t_w;
    double taux;
    t_w = (double *) malloc(nprocs*sizeof(double));
    if(!t_w) MPI_Abort(MPI_COMM_WORLD, rc);
    for(i = 0; i<nprocs; ++i){
      done[i]  = 0;
      waiting[i] = 0;
      t_w[i] = 0.0;
      failed[i] = 0;
    }
    while(count<nworkers){
      if(loops==0 && done[dest]==0){
	MPI_Isend(&rdy, 1, MPI_INT, dest, FROM_MASTER_RDY,MPI_COMM_WORLD,&ireq[dest]);
	MPI_Irecv(&rcv, 1, MPI_INT, dest, FROM_WORKER_RDY, MPI_COMM_WORLD, &ireq[dest]);
	t_w[dest] = MPI_Wtime();
      }
      taux = MPI_Wtime();
      if(!done[dest] && (taux - t_w[dest]) >= ((double) waiting[dest])*time_e/5){
	MPI_Test(&ireq[dest],&done[dest],&stat);
	if(!done[dest]){
	  ++waiting[dest];
	  t_w[dest] = MPI_Wtime();
	  if(waiting[dest] == 9){
	    MPI_Isend(&stp, 1, MPI_INT, dest, FROM_MASTER_RDY, MPI_COMM_WORLD, ireq + dest);
	    for(int i = 0; i<Nodesamp;++i){
	      x[0] = ((double) rand())/RAND_MAX*2 - 1;
	      x[1] = ((double) rand())/RAND_MAX*2 - 1;
	      resf += rhs(x) < 1 ? 1:0;
	    }
	    ++count;
	    done[dest] = 1;
	  }
	}else{
	  MPI_Sendrecv(&rdy, 1, MPI_INT, dest, FROM_MASTER_RDY, &res, 1, MPI_INT, dest, FROM_WORKER, MPI_COMM_WORLD, &stat);
	  resf+=res;
	  ++count;
	  time_e = (taux-t_w[dest]<time_e ? taux-t_w[dest]:time_e);
	  done[dest] = 1;
	}
      }
      dest = (dest == nworkers ? 1:dest+1);
      if(dest == 1 && !loops) loops = 1;
    }
    for(dest = 1; dest<nprocs; ++dest){
      MPI_Send(&stp, 1, MPI_INT, dest, FROM_MASTER_RDY, MPI_COMM_WORLD);
    }
    time_f = MPI_Wtime();
    printf("Time elapsed: %lf \n", time_f-time_s);
    printf("Estimation of pi: %lf \n", 4*((double)resf)/Nsamples);
    for(i = 0; i<nprocs; ++i){
      ireq[i] = MPI_REQUEST_NULL;
    }

    free(done);
    free(waiting);
    free(t_w);
  }else{
    srand(time(NULL) + getpid() + myid*nprocs);
    if(rand()/((double)RAND_MAX)>=p_fail){
      int i;
      for(i = 0; i<Nodesamp; ++i){
	x[0] = ((double) rand())/RAND_MAX*2 - 1;
	x[1] = ((double) rand())/RAND_MAX*2 - 1;
	res += rhs(x) < 1 ? 1:0;
      }
      int msg;
      MPI_Recv(&msg, 1, MPI_INT, MASTER, FROM_MASTER_RDY, MPI_COMM_WORLD, &stat);
      MPI_Send(&msg, 1, MPI_INT, MASTER, FROM_WORKER_RDY, MPI_COMM_WORLD);
      MPI_Recv(&msg, 1, MPI_INT, MASTER, FROM_MASTER_RDY, MPI_COMM_WORLD, &stat);
      if(msg == 1){
	MPI_Sendrecv(&res, 1, MPI_INT, MASTER, FROM_WORKER,&msg,1,MPI_INT,MASTER,FROM_MASTER_RDY, MPI_COMM_WORLD, &stat);
      }
      while(msg == 1){
	if(msg==1){
	  res = 0;
	  for(int i = 0; i<Nodesamp; ++i){
	    x[0] = ((double) rand())/RAND_MAX*2 - 1;
	    x[1] = ((double) rand())/RAND_MAX*2 - 1;
	    res += rhs(x) < 1 ? 1:0;
	  }
	  MPI_Sendrecv(&res, 1, MPI_INT, MASTER, FROM_WORKER, &msg,1,MPI_INT,MASTER,FROM_MASTER_RDY,MPI_COMM_WORLD, &stat);
	}
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return(0);
}
