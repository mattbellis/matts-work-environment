#
# %Id: Makefile,v 1.1 
#
#
#
CC = gcc
CPP = g++

CCFLAGS = $(shell root-config --cflags)
LIBS = $(shell root-config --libs) 
LDFLAGS = -lMinuit 

%: %.c
	$(CC) $(CCFLAGS) -c -o objects/$*.o $*.c
	$(CC) objects/$*.o $(LIBS) $(LDFLAGS) -o $(HOME)/bin/$*

%: %.cc
	$(CPP) $(CCFLAGS) -c -o objects/$*.o $*.cc
	$(CPP) objects/$*.o $(LIBS) $(LDFLAGS) -o $(HOME)/bin/$*

%: %.C
	$(CPP) $(CCFLAGS) -c -o objects/$*.o $*.C
	$(CPP) objects/$*.o $(LIBS) $(LDFLAGS) -o $(HOME)/bin/$*

%: %.cpp
	$(CPP) $(CCFLAGS) -c -o objects/$*.o $*.cpp
	$(CPP) objects/$*.o $(LIBS) $(LDFLAGS) -o $(HOME)/bin/$*

%: %.cxx
	$(CPP) $(CCFLAGS) -c -o objects/$*.o $*.cxx
	$(CPP) objects/$*.o $(LIBS) $(LDFLAGS) -o $(HOME)/bin/$*
