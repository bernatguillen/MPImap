#include <stdio.h>
#include "mpi.h"
#include <time.h>

#define NRA 10
#define NCA 10
#define NCB 7
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

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

void main(int argc, char **argv){
  int nprocs, myid, nworkers, source, dest, mtype, rows, averow, extra, offset, i,j, k, rc;

  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  nworkers = nprocs - 1;
  MPI_Status stat;
  MPI_Request *ireq;
  ireq = (MPI_Request *) malloc(3*nworkers);
  if(nprocs < 2){
    printf("Need at least two tasks\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  /*Master task*/
  if (myid == MASTER){
    double A[NRA][NCA], B[NCA][NCB], C[NRA][NCB];
    initializeMatrix(A, NRA, NCA);
    initializeMatrix(B, NCA, NCB);
    MPI_Bcast(B, NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Broadcast

    averow = NRA/nworkers;
    extra = NRA%nworkers;
    offset = 0;
    mtype = FROM_MASTER;
    for(dest = 1; dest<nprocs; ++dest){
      rows = (dest <= extra) ? averow+1 : averow;
      MPI_Isend(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD,ireq + dest-1);
      MPI_Isend(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, ireq + nworkers + dest-1);
      MPI_Isend(&A[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, ireq + 2*nworkers + dest - 1);
      offset += rows;
    }
    MPI_Waitall(3*nworkers, ireq, &stat);
    mtype = FROM_WORKER
    for(source = 1; source < nprocs; ++dest){
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
      MPI_Recv(&C[offset][0], rows*NCA, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &stat);
    }

    free(ireq);
  }else{ //Worker process
    double B[NCA][NCB];
    MPI_Bcast(B, NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //Synchronous receive (?)
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    double **A = malloc(rows*sizeof(double *));
    for(i = 0; i<rows; ++i){
      A[i] = malloc(NCA*sizeof(double));
    }
    MPI_Recv(&A, rows*NCA, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    double **C = malloc(rows*sizeof(double *));
    for(i = 0; i<rows; ++i){
      C[i] = malloc(NCB*sizeof(double));
    }
    for(k = 0; k<NCB; ++k){
      for(i = 0; i<rows; ++i){
	C[i][k] = 0.0;
	for(j = 0; j<NCA; ++j){
	  C[i][k] += A[i][j]*B[j][k];
	}
      }
    }
    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&C, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    for(i = 0; i<rows; ++i){
      free(C[i]);
      free(A[i]);
    }
    free(C);
    free(A);
  }  
  printf("hello world\n"); 
  MPI_Finalize();
  return;

}
