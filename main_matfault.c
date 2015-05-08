#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <assert.h>
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2
#define RDY 1

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

int reassigndata(int *working, int *waiting, int dest0, int nprocs, double **A, int *offset_m, int *rows_m, int NRA, int NCA){
  int mtype, averow, extra, nworkers, dest, di;
  int go = 1;
  nworkers = nprocs - 1;
  mtype = FROM_MASTER;
  averow = NRA/nworkers;
  extra = NRA%nworkers;
  dest = 0;
  di = 1;
  while(dest == 0){
    if(di!=dest0 && working[di]==MPI_SUCCESS && waiting[di] == 0){
      dest = di;
    }
    di = (di+1 < nworkers ? di+1 : 1);
    if(di == 1) return -1; //error, no node available
  }
  MPI_Send(&go, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
  MPI_Send(&offset_m[dest0], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
  MPI_Send(&rows_m[dest0], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
  MPI_Send(&A[offset_m[dest0]][0], rows_m[dest]*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
  return 1;
}


int senddata(int *working, int dest, int nprocs, double **A, int *offset_m, int *rows_m, int NRA, int NCA){
  int mtype, averow, extra,nworkers;
  nworkers = nprocs - 1;
  mtype = FROM_MASTER;
  averow = NRA/nworkers;
  extra = NRA%nworkers;
  if(dest < nprocs-1){
    offset_m[dest] = offset_m[dest-1] + rows_m[dest-1];
    rows_m[dest] += (dest <= extra) ? averow+1 : averow;      
    working[dest]=MPI_Send(&offset_m[dest], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    if(working[dest] != MPI_SUCCESS){
      rows_m[dest+1] = rows_m[dest];
      rows_m[dest] = 0;
      return senddata(working, dest+1, nprocs, A, offset_m, rows_m, NCA, NRA);
    }else{
      working[dest]|=MPI_Send(&rows_m[dest], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      if(working[dest] != MPI_SUCCESS){
	offset_m[dest+1] = offset_m[dest];
	offset_m[dest] = -1;
	rows_m[dest+1] = rows_m[dest];
	rows_m[dest] = 0;
	return senddata(working, dest+1, nprocs, A, offset_m, rows_m, NCA, NRA);
      }else{
      	working[dest]|=MPI_Send(&A[offset_m[dest]][0], rows_m[dest]*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
	if(working[dest] != MPI_SUCCESS){
	  offset_m[dest+1] = offset_m[dest];
	  offset_m[dest] = -1;
	  rows_m[dest+1] = rows_m[dest];
	  rows_m[dest] = 0;
	  return senddata(working, dest+1, nprocs, A, offset_m, rows_m, NCA, NRA);
	}else{
	  return senddata(working, dest+1, nprocs, A, offset_m, rows_m, NCA, NRA);
	}
      }
    }
  }else{
    offset_m[dest] = offset_m[dest-1] + rows_m[dest-1];
    rows_m[dest] += (dest <= extra) ? averow+1 : averow;      
    working[dest]=MPI_Send(&offset_m[dest], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    if(working[dest]!=MPI_SUCCESS){
      offset_m[0] = offset_m[dest];
      rows_m[0] = rows_m[dest];
      return -1;
    }else{
      working[dest]|=MPI_Send(&rows_m[dest], 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      if(working[dest]!=MPI_SUCCESS){
	offset_m[0] = offset_m[dest];
	rows_m[0] = rows_m[dest];
	return -1;
      }else{
	working[dest]|=MPI_Send(&A[offset_m[dest]][0], rows_m[dest]*NCA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
	if(working[dest]!=MPI_SUCCESS){
	  offset_m[0] = offset_m[dest];
	  rows_m[0] = rows_m[dest];
	  return -1;
	}
      }
    }
  }
  return 0;
}

int main(int argc, char **argv){
  int nprocs, myid, nworkers, source, dest, mtype, i,j, k, rc;
  int *rows_m, *offset_m, *working;
  int NRA,NCA,NCB;
  int rows, offset; 
  MPI_Status status;
  MPI_Init(&argc,&argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); //Does not immediately abort
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  nworkers = nprocs - 1;
  MPI_Status stat;
  MPI_Request *ireq;
  ireq = (MPI_Request *) malloc(3*nworkers);
  if(nprocs < 2 || argc != 4){
    if (myid == MASTER)
      printf("usage: main_matmul NRA NCA NCB \n NRA: number of rows in A \n NCA: number of columns in A \n NCB: number of rows in B \n (NRB == NCA) \n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }
  /* Dynamically allocate matrices */
    NRA = atoi(argv[1]);
    NCA = atoi(argv[2]);
    NCB = atoi(argv[3]);

  /*Master task*/
  if (myid == MASTER){
    //Begin timing
    double time_s,time_f;
    time_s = MPI_Wtime();
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
    MPI_Bcast(&(B[0][0]), NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Broadcast

    offset_m = (int *) malloc(sizeof(int)* nprocs);
    rows_m = (int *) malloc(sizeof(int)* nprocs);
    working = (int *) malloc(sizeof(int)* nprocs);
    offset_m[0] = 0; 

    for(int i = 0; i<nprocs; ++i){
      working[i] = MPI_SUCCESS;
      rows_m[i] = 0;
    }
    if(senddata(working,1,nprocs,A,offset_m,rows_m,NRA,NCA) == -1){
      //compute
    }
    //senddata
    mtype = FROM_WORKER;
    int *flags;
    flags = (int *) malloc(sizeof(int)* nprocs);
    int *waiting; 
    waiting = (int *) malloc(sizeof(int) * nprocs);
    double *time_w;
    time_w = (double *) malloc(sizeof(double) * nprocs);
    for(int i = 0; i<nprocs; ++i){
      time_w[i] = 0.0;
      waiting[i] = 0;
    }
    int source = 1;
    int count = 0;
    int rdy = 1;
    int go = 1;
    int stop = 0;
    for(int i = 0; i<nworkers; ++i){
      if(working[i+1] != MPI_SUCCESS){
	++count;
      }
    }
    while(count < nworkers){
      if(working[source] == MPI_SUCCESS && MPI_Wtime() - time_w[source] >= waiting[source]){
	if(waiting[source] == 0){
	  MPI_Isend(&rdy,1,MPI_INT,source,FROM_MASTER,MPI_COMM_WORLD, ireq+source);
	}
	MPI_Test(ireq+source,&flags[source],&stat);
	if(flags[source]){
	  MPI_Send(&go,1,MPI_INT,source,FROM_MASTER,MPI_COMM_WORLD);
	  MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
	  MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &stat);
	  MPI_Recv(&C[offset][0], rows*NCB, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &stat);
	  /* Ensure that we receive the same amount of info and positions as MASTER sent */
	  assert(rows == rows_m[source]);
	  assert(offset == offset_m[source]);
	  ++count;
	  waiting[source] = 0;
	}else{
	  time_w[source] = MPI_Wtime();
	  ++waiting[source];
	  if(waiting[source] == 9){
	    reassigndata(working,waiting,source,nprocs,A,offset_m,rows_m,NRA,NCA);
	    MPI_Isend(&stop,1,MPI_INT,source,FROM_MASTER,MPI_COMM_WORLD,ireq);
	  }
	}
      }
      source = source+1 < nprocs ? source+1:1;
    }
    for(int i = 1; i<nprocs;++i){
      MPI_Isend(&stop,1,MPI_INT,i,FROM_MASTER,MPI_COMM_WORLD,ireq);
    }
    time_f = MPI_Wtime();
    printf("computation took %lf seconds \n",time_f-time_s);
    MPI_Finalize();
  }else{ //Worker process
    double **B; 
    double *dataB;
    //make contiguous multiarrays
    B = (double **) malloc(sizeof(double *)*NCA); 
    dataB = (double *) malloc(sizeof(double)*NCA*NCB); 
    for(int i=0; i<NCA; i++){
      B[i] = &(dataB[NCB*i]);
    }  

    MPI_Bcast(&(B[0][0]), NCA*NCB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
    double **A = (double **) malloc(rows*sizeof(double *));
    double *dataA = (double *) malloc(sizeof(double)*NCA*rows); 
    for(i = 0; i<rows; ++i){
      A[i] = &(dataA[NCA*i]);
    }
    MPI_Recv(&(A[0][0]), rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &stat); 
    double **C = (double **) malloc(rows*sizeof(double *));
    double *dataC = (double *) malloc(sizeof(double)*NCB*rows); 
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
    free(A);
    free(dataA);
    mtype = FROM_WORKER;
    int mess;
    MPI_Recv(&mess,1,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&stat);
    MPI_Recv(&mess,1,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&stat);
    if(mess == 1){ 
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&(C[0][0]), rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }else{
      MPI_Finalize();
    }
    free(C);
    free(dataC);
    MPI_Recv(&mess,1,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&stat);
    if(mess == 1){
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &stat);
      A = (double **) malloc(rows*sizeof(double *));
      dataA = (double *) malloc(sizeof(double)*NCA*rows); 
      for(i = 0; i<rows; ++i){
	A[i] = &(dataA[NCA*i]);
      }
      MPI_Recv(&(A[0][0]), rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &stat); 
      C = (double **) malloc(rows*sizeof(double *));
      dataC = (double *) malloc(sizeof(double)*NCB*rows); 
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
      MPI_Recv(&mess,1,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&stat);
      MPI_Recv(&mess,1,MPI_INT,MASTER,FROM_MASTER,MPI_COMM_WORLD,&stat);
      if(mess == 1){ 
	MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	MPI_Send(&(C[0][0]), rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      }else{
	MPI_Finalize();
      }
    }else{
      MPI_Finalize();
    }
  }  
  return(0);
}
