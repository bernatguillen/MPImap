#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define NPARTICLES 5000
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
  double phalf_vx;
  double phalf_vy;
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
    particles[i].px = rand()/(double) RAND_MAX;
    particles[i].py = rand()/(double) RAND_MAX;
    /*sample direction (negative or positive boolean) */
    if( rand() % 2){
	particles[i].pvx = 0.01*rand()/(double) RAND_MAX;
    }
    else {
	particles[i].pvx = -0.01*rand()/(double) RAND_MAX;
    }
    if( rand() % 2){
	particles[i].pvy = 0.01*rand()/(double) RAND_MAX;
    }
    else {
	particles[i].pvy = -0.01*rand()/(double) RAND_MAX;
    }
    //    particles[i].phalf_vx = 0.01* rand()/(double) RAND_MAX;
    //particles[i].phalf_vy = 0.01*rand()/(double) RAND_MAX;
    //       printf("pid: %d x: %lf y: %lf vx: %lf vy: %lf vhalfx: %lf vhalfy: %lf\n",particles[i].pid,particles[i].px,particles[i].py,particles[i].pvx,particles[i].pvy,particles[i].phalf_vx,particles[i].phalf_vy);
  }
  double *forces; 
  forces = (double *) malloc(2*nlocal*sizeof(double));

  /* Need initial V at two separate time levels for leap frog */

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
  double norm_squared,unit_x,unit_y;
  for (i=0; i<NT; i++){
    time = i*DT; 
    //zero out force array
    for (j=0; j<2*nlocal; j++)
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
	if (k!=j){
	  /*Regularize pairwise distance */
	  norm_squared= pow(particles[k].px - particles[j].px,2) + pow(particles[k].py - particles[j].py,2)+ pow(EPSILON,2);
	  unit_x = (particles[k].px - particles[j].px)/sqrt(norm_squared); 
	  unit_y = (particles[k].py - particles[j].py)/sqrt(norm_squared); 
	  forces[2*k] += (NEWTON*particles[k].pm*particles[j].pm*unit_x)/norm_squared;
	  forces[2*k+1] += (NEWTON*particles[k].pm*particles[j].pm*unit_y)/norm_squared;
	}
      }
    }
    /* Wait for recvs, update force for everyone of my particles */
    for (j=0; j<nprocs; j++){
      if (j!=myid){ 
	MPI_Wait(&rreq[j],&stat); 
	for (k=0; k<nlocal; k++){
	  for (l=0; l<nlocal; l++){
	    //remember, particle_in is not Particle struct, but doubles
	    norm_squared= pow(particles[k].px - particle_in[j][l*3+1],2) + pow(particles[k].py - particle_in[j][l*3+2],2)+ pow(EPSILON,2);
	    unit_x = (particles[k].px - particle_in[j][l*3+1])/sqrt(norm_squared); 
	    unit_y = (particles[k].py - particle_in[j][l*3+2])/sqrt(norm_squared); 
	    forces[2*k] += (NEWTON*particles[k].pm*particle_in[j][l*3]*unit_x)/norm_squared;
	    forces[2*k+1] += (NEWTON*particles[k].pm*particle_in[j][l*3]*unit_y)/norm_squared;
	  }
	}
      }
    }

    /*Use leapfrog scheme to update local velocities */
    if (i==0){ /* Seed vhalf at first time level */
      //should use higher order, but i dont want to compute multiple force timesteps
      for (j=0; j<nlocal; j++){
	particles[j].phalf_vx = particles[j].pvx + forces[2*j]*DT/(particles[j].pm*2);
	particles[j].phalf_vy = particles[j].pvy + forces[2*j+1]*DT/(particles[j].pm*2);
      }
    }
    else {
      for (j=0; j<nlocal; j++){
	particles[j].pvx = particles[j].phalf_vx;
	particles[j].pvy = particles[j].phalf_vy;
	particles[j].phalf_vx = particles[j].pvx + forces[2*j]*DT/particles[j].pm;
	particles[j].phalf_vy = particles[j].pvy + forces[2*j+1]*DT/particles[j].pm;
      }
    }
    for (j=0; j<nlocal; j++){
      particles[j].px = particles[j].px + particles[j].phalf_vx*DT;
      particles[j].py = particles[j].py + particles[j].phalf_vy*DT;
      /*Eliminate particles that have left the domain. (for now, just reseed position and velocity)*/
      if (particles[j].px  <  0.0 || particles[j].px > lx || particles[j].py > ly || particles[j].px < 0.0){
	//        printf("reseeding...\n");
	//	printf("pid: %d x: %lf y: %lf vx: %lf vy: %lf vhalfx: %lf vhalfy: %lf\n",particles[j].pid,particles[j].px,particles[j].py,particles[j].pvx,particles[j].pvy,particles[j].phalf_vx,particles[j].phalf_vy);
	particles[j].px = rand()/(double) RAND_MAX;
	particles[j].py = rand()/(double) RAND_MAX;
	/*sample direction (negative or positive boolean) */
	if( rand() % 2){
	  particles[j].pvx = 0.01*rand()/(double) RAND_MAX;
	}
	else {
	  particles[j].pvx = -0.01*rand()/(double) RAND_MAX;
	}
	if( rand() % 2){
	  particles[j].pvy = 0.01*rand()/(double) RAND_MAX;
	}
	else {
	  particles[j].pvy = -0.01*rand()/(double) RAND_MAX;
	}
	particles[j].phalf_vx = particles[j].pvx;
	particles[j].phalf_vy = particles[j].pvy;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  tstop = MPI_Wtime();
  if (myid == 0) {
    printf("Time = %g (secs)\n",tstop-tstart);
  }

  MPI_Finalize();
  return;

}
