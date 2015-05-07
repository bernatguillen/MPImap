# MPI-based makefile using mpic++ and mpicc

CC =            mpicc -m64
CPP =           mpic++ -m64
CCFLAGS =       -g -O -I/Users/kfelker/Documents/Courses/COS598/final_project/mrmpi-7Apr14/src
LINK =          mpic++
LINKFLAGS =     -g -O
USRLIB =        /Users/kfelker/Documents/Courses/COS598/final_project/mrmpi-7Apr14/src/libmrmpi_mpicc.a
SYSLIB =

# Targets

all:    mr-matmul mr-nbody mr-montecarlo

mr-matmul:      mr-matmul.o $(USRLIB)
	$(LINK) $(LINKFLAGS) mr-matmul.o $(USRLIB) $(SYSLIB) -o mr-matmul

mr-nbody:	mr-nbody.o $(USRLIB)
	$(LINK) $(LINKFLAGS) mr-nbody.o $(USRLIB) $(SYSLIB) -o mr-nbody

mr-montecarlo:	mr-montecarlo.o $(USRLIB)
	$(LINK) $(LINKFLAGS) mr-montecarlo.o $(USRLIB) $(SYSLIB) -o mr-montecarlo

clean:
	rm *.o mr-matmul mr-nbody mr-montecarlo

%.o:%.c
	$(CC) $(CCFLAGS) -c $<