#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>

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

int main(int argc, char **argv){
  int nprocs, myid, nworkers, source, dest, mtype, averow, extra, offset, i,j, k, rc;
  int NRA,NCA,NCB;
  int rows; 
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
  /* Dynamically allocate matrices */
    NRA = atoi(argv[1]);
    NCA = atoi(argv[2]);
    NCB = atoi(argv[3]);

  /*Master task*/
  if (myid == MASTER){
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
    MPI_Bcast(B, NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Broadcast
    averow = NRA/nworkers;
    extra = NRA%nworkers;
    offset = 0;
    mtype = FROM_MASTER;
    printf("master before sends \n");
    printf("%d %d\n",NCB,rows);
    for(dest = 1; dest<nprocs; ++dest){
      rows = (dest <= extra) ? averow+1 : averow;      
      MPI_Isend(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD,ireq + dest-1);
      MPI_Isend(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD, ireq + nworkers + dest-1);
      MPI_Isend(&A[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD, ireq + 2*nworkers + dest - 1);
      offset += rows;
    }
    printf("master after sends \n");
    MPI_Waitall(3*nworkers, ireq, &stat); //kyle: do we need to wait for the messages to complete before posting the recieves?
    mtype = FROM_WORKER;
    for(source = 1; source < nprocs; ++dest){
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
      MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
      //kyle edit: shouldnt this be rows*NCB?
      //      MPI_Recv(&C[offset][0], rows*NCA, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &stat);
      MPI_Recv(&C[offset][0], rows*NCB, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &stat);
    }

    //free(ireq);
  }else{ //Worker process
    double **B; 
    double *dataB;
    //make contiguous multiarrays
    B = (double **) malloc(sizeof(double *)*NCA); 
    dataB = (double *) malloc(sizeof(double)*NCA*NCB); 
    for(int i=0; i<NCA; i++){
      B[i] = &(dataB[NCB*i]);
    }  

    MPI_Bcast(B, NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //Synchronous receive (?)
    mtype = FROM_MASTER;
    printf("worker before recvs \n");
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    printf("worker NCB: %d rows: %d\n",NCB,rows);
    double **A = (double **) malloc(rows*sizeof(double *));
    double *dataA = (double *) malloc(sizeof(double)*NCA*rows); 
    for(i = 0; i<rows; ++i){
      A[i] = &(dataA[NCA*i]);
    }
    printf("worker before recv A \n");
    MPI_Recv(&A, rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &stat);
    printf("worker after recv A\n");
    double **C = (double **) malloc(rows*sizeof(double *));
    printf("worker after allocate C **\n");
    printf("worker NCB: %d rows: %d\n",NCB,rows);
    //KYLE: FOR SOME REASON, NCB AND ROWS GETS RESET FROM 10 LINES UP TO NOW
    double *dataC = (double *) malloc(sizeof(double)*NCB*rows); 
    printf("worker after allocate C \n");
    for(i = 0; i<rows; ++i){
      C[i] = &(dataC[NCB*i]);
    }
    for(k = 0; k<NCB; ++k){
      for(i = 0; i<rows; ++i){
	C[i][k] = 0.0;
	for(j = 0; j<NCA; ++j){
	  C[i][k] += A[i][j]*B[j][k];
	}
      }
    }
    printf("worker after recvs \n");
    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&C, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    /*for(i = 0; i<rows; ++i){
      free(C[i]);
      free(A[i]);
    }
    free(C);
    free(A); */
  }  
  printf("hello world\n"); 
  MPI_Finalize();
  return(0);

}
