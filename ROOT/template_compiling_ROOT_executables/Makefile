#
# %Id: Makefile,v 1.1 
#
#
#
CC = gcc
CPP = g++

CCFLAGS = -O -g -Wall -fPIC $(shell root-config --cflags)  
LIBS = $(shell root-config --libs)  
LDFLAGS = -lMinuit 

%: %.cc 
	$(CPP) $(CCFLAGS) -c -o ./$*.o $*.cc
	$(CPP) ./$*.o $(LIBS) $(LDFLAGS) -o ./$*



