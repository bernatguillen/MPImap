#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define NPARTICLES 1000
#define NT 100
#define DT 0.1
#define NEWTON 6.67E-11
#define EPSILON 1E-2 //softening parameter for close encoutners

/* 2D Direct summation N-body simulation using data decomposition 
   (not domain decomposition) */

typedef struct{
  int pid;
  double pm; //mass in kg
  double px; //position in m
  double py;
  double pvx;
  double pvy;
} Particle; 

void main(int argc, char **argv){
  int nprocs, myid;
  int i,j,k,l;
  double tstart,tstop;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  /* Size of square domain */
  double lx = 1.0;
  double ly = lx; 

  tstart = MPI_Wtime();

  /* Initialize local particles */
  time_t t;
  srand((unsigned) time(&t));
  int nlocal = NPARTICLES/nprocs;
  Particle *particles; 
  particles = (Particle *) malloc(nlocal*sizeof(Particle)); 
  for (i=0; i<nlocal; i++){
    particles[i].pid = myid*nlocal+i; 
    particles[i].pm = 1.0;
    particles[i].px = rand();
    particles[i].py = rand();
    particles[i].pvx = rand()/50;
    particles[i].pvy = rand()/50;
  }
  double *forces; 
  forces = (double *) malloc(nlocal*sizeof(double));
  /* Message arrays: 1 identical outbound, nprocs-1 inbound */
  //manually pack the array of particle messages with stride 3
  double *particle_out = (double *) malloc(3*nlocal*sizeof(double)); 
  double **particle_in = (double **) malloc((nprocs)*sizeof(double *));
  for (i=0; i<nprocs; i++)
    particle_in[i] = (double *) malloc(3*nlocal*sizeof(double)); 
  MPI_Request sreq; 
  MPI_Request *rreq = (MPI_Request *) malloc(nprocs*sizeof(MPI_Request));
  MPI_Status stat; 
  /* Main timestepping loop */
  double time;
  for (i=0; i<NT; i++){
    time = i*DT; 
    //zero out force array
    for (j=0; j<nlocal; j++)
      forces[j]=0.0;
    /*Send particle positions, mass info to every other process */
    for (j=0; j<nlocal; j++){
      particle_out[3*j] = particles[j].pm;
      particle_out[3*j+1] = particles[j].px;
      particle_out[3*j+2] = particles[j].py;
    }
    for (j=0; j<nprocs; j++){
      if (j!=myid)
	MPI_Isend(particle_out,3*nlocal,MPI_DOUBLE,j,0,MPI_COMM_WORLD,&sreq);
    }
    /*Post receives fro other processor particle info*/
    for (j=0; j<nprocs; j++){
      if (j!=myid)
	MPI_Irecv(particle_in[j],3*nlocal,MPI_DOUBLE,j,0,MPI_COMM_WORLD,&rreq[j]);
    }
    /*Compute net forces among own particles */
    for (j=0; j<nlocal; j++){
      for (k=0; k<nlocal; k++){
	if (k!=j)
	  forces[k] += (NEWTON*particles[k].pm*particles[j].pm)/(pow(particles[k].px - particles[j].px,2) + pow(particles[k].py - particles[j].py,2) + pow(EPSILON,2));
      }
    }
    /* Wait for recvs, update force for everyone of my particles */
    for (j=0; j<nprocs; j++){
      if (j!=myid){ 
	MPI_Wait(&rreq[j],&stat); 
	for (k=0; k<nlocal; k++){
	  for (l=0; l<nlocal; l++){
	    //remember, particle_in is not Particle struct, but doubles
	    forces[k] += (NEWTON*particles[k].pm*particle_in[j][l*3])/(pow(particles[k].px - particle_in[j][l*3+1],2) + pow(particles[k].py - particle_in[j][l*3+2],2) + pow(EPSILON,2));
	  }
	}
      }
    }

    /*Use leapfrog scheme to update local velocities */

    /*Regularize pairwise distance */

    /*Eliminate particles that have left the domain. */
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  tstop = MPI_Wtime();
  if (myid == 0) {
    printf("Time = %g (secs)\n",tstop-tstart);
  }


  MPI_Finalize();
  return;

}
