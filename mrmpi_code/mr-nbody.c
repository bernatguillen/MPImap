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

#define NPARTICLES 5000
#define NT 100
#define DT 0.1
#define NEWTON 6.67E-11
#define EPSILON 1E-2 //softening parameter for close encounter

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

typedef struct{
  int id;
  int ntask;
  double x; 
  double y;
  double m;
} Particle_message;

typedef struct{
  double *f;
  Particle *p;
  int n;
}Force_struct; 

void map_positions(int itask, void *kv, void *ptr);
void reduce_forces(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr);
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

  /*Create MapReduce object */
  void *mr = MR_create(MPI_COMM_WORLD);
  MR_set_verbosity(mr,0);
  MR_set_timer(mr,0);
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
    particles[i].pid = me*nlocal+i; 
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
    //       printf("pid: %d x: %lf y: %lf vx: %lf vy: %lf vhalfx: %lf vhalfy: %lf\n",particles[i].pid,particles[i].px,particles[i].py,particles[i].pvx,particles[i].pvy,particles[i].phalf_vx,particles[i].phalf_vy);
  }
  Particle_message *particle_m; 
  particle_m = (Particle_message *) malloc(nlocal*sizeof(Particle_message)); 

  double time;
  /* Create structure for passing owned particles and forces to reducer */
  double *forces;
  double unit_x,unit_y,norm_squared;
  forces = (double *) malloc(2*nlocal*sizeof(double));
  Force_struct force_particles;
  force_particles.f = forces;
  force_particles.p = particles;
  force_particles.n = nlocal;

  /* Main timestepping loop */
  for (i=0; i<NT; i++){
    time = i*DT; 
    for (j=0; j<nlocal; j++){
      particle_m[j].id = particles[j].pid;
      particle_m[j].x = particles[j].px;
      particle_m[j].y = particles[j].py;
      particle_m[j].m = particles[j].pm;
      particle_m[j].ntask = nprocs; 
    }
    nelements_in = MR_map(mr,nprocs,&map_positions,(void *) particle_m); //all processes must participate
    /* Redistribute all the block KVs to consistent processors */
    /* Turn the KV into a KVM since they all have the same key */
    MR_collate(mr, NULL); //both aggregates and converts to KVM

    /*Zero out forces from last timestep */
    for (j=0; j<2*nlocal; j++)
      force_particles.f[j]=0.0;
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
    /* NEED TO ENSURE THAT THE REDUCER IS THE SAME AS THE MAPPER TO PRESERVE LOCALLY OWNED PARTICLES*/
    nelements_out = MR_reduce(mr,&reduce_forces,(void *)&force_particles);
    //    printf("nelements_in = %d \n", (int) nelements_in); 
    //printf("nelements_out = %d \n", (int) nelements_out); 
    
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
  } //end of timestepper
  tstop = MPI_Wtime();
  MR_destroy(mr);
  if (me == 0) {
    printf("Time = %g (secs)\n",tstop-tstart);
  }

  MPI_Finalize();
  return(0);
}

  
void map_positions(int itask, void *kv, void *ptr)
{
  //void MR_kv_add(void *KVptr, char *key, int keybytes,char *value, int valuebytes);
  int key_array[4]; 	
  double value_array[3];
  Particle_message *particle_m = (Particle_message *) ptr; 
  int i,j;
  int ntask = particle_m[0].ntask;
  int nlocal = NPARTICLES/ntask; 
  for (i=0; i<ntask; i++){ //send all of the local particle info to every other process
    if (i != itask){
      for (j=0;j<nlocal; j++){
	MR_kv_add(kv,(char *) &i,sizeof(int),(char *) &particle_m[j],sizeof(Particle_message)); 
      }
    }
  }
  return;
}

//must take the following arguments
void reduce_forces(char *key, int keybytes,char *multivalue, int nvalues, int *valuebytes, void *kv, void *ptr){
  //key is unnecessary and hopefully just equal to the proc rank
  Particle_message *v_array = (Particle_message *) multivalue; 
  int k,j;
  Force_struct *force_particles = (Force_struct *) ptr; 
  int nlocal = force_particles->n;
  Particle *particles = force_particles->p;
  double *forces = force_particles->f;
  double unit_x,unit_y,norm_squared;

  for (j=0; j<nvalues; j++){
    for (k=0; k<nlocal; k++){      
      norm_squared= pow(particles[k].px - v_array[j].x,2) + pow(particles[k].py - v_array[j].y,2)+ pow(EPSILON,2);
      unit_x = (particles[k].px - v_array[j].x)/sqrt(norm_squared); 
      unit_y = (particles[k].py - v_array[j].y)/sqrt(norm_squared); 
      forces[2*k] += (NEWTON*particles[k].pm*v_array[j].m*unit_x)/norm_squared;
      forces[2*k+1] += (NEWTON*particles[k].pm*v_array[j].m*unit_y)/norm_squared;
    }
  }
} 
