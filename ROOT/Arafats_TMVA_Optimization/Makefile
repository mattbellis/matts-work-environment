#
# Matt's super simple ROOT Makefile
#

################################################################################
# Standard boilerplate compiler options
################################################################################
CC = gcc
CPP = g++

################################################################################
# Standard ROOT flags
################################################################################
CCFLAGS = -O -g -Wall -fPIC $(shell root-config --cflags)
LIBS = $(shell root-config --libs)

################################################################################
# More particular dynamic libraries you may or may not need.
################################################################################
#LDFLAGS = -lTMVA.1 -lMLP -lXMLIO -lTreePlayer -lMinuit
LDFLAGS = -lTMVA -lMLP -lXMLIO -lTreePlayer -lMinuit

################################################################################
# Compile and link!!!!! 
################################################################################
%: %.cc
	$(CPP) $(CCFLAGS) -c -o ./$*.o $*.cc
	$(CPP) ./$*.o $(LIBS) $(LDFLAGS) -o ./$*
