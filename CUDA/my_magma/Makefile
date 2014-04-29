#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.4.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      December 2013
#//////////////////////////////////////////////////////////////////////////////

CUDADIR=/usr/local/cuda-5.5
DIR = testing
MAGMA_DIR = /home/bellis/src/magma-1.4.1
include /home/bellis/src/magma-1.4.1/Makefile.internal

# ----------
ifeq ($(FORT), pgfortran)
	FOBJ  = fortran_thunking.o
	ZSRCF = testing_zgetrf_gpu_f.cuf
else
	FOBJ  = fortran.o
	ZSRCF = testing_zgetrf_gpu_f.F90	\
		testing_zgetrf_f.f90
endif


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)
ALLOBJ := $(ALLOBJ:.cuf=.o)
ALLOBJ := $(ALLOBJ:.cpp=.o)

EXE    := $(ALLOBJ:.o=)

# --------------------
#LIBTEST := libtest.a
#LIBSRC  := testing_zutil.o testing_cutil.o testing_dutil.o testing_sutil.o testing_util.o
#LIBOBJ  := $(LIBSRC:.cpp=.o)
#LIBEXE  := $(LIBOBJ:.o=)
#
## testing_zutil.cpp isn't in ZSRC (since they don't generate EXEs),
## so add generation rules here instead of .Makefile.gen
#testing_cutil.cpp: testing_zutil.cpp
	#$(PYTHON) ../tools/codegen.py  -p c --file testing_zutil.cpp
#
#testing_dutil.cpp: testing_zutil.cpp
	#$(PYTHON) ../tools/codegen.py  -p d --file testing_zutil.cpp
#
#testing_sutil.cpp: testing_zutil.cpp
	#$(PYTHON) ../tools/codegen.py  -p s --file testing_zutil.cpp
#
#generate: testing_cutil.cpp testing_dutil.cpp testing_sutil.cpp


# --------------------
.PHONY: all lib clean cleanall cleanexe

.DEFAULT_GOAL := all
all: $(EXE)

lib:
	cd .. && $(MAKE) lib

# depend on header
$(ALLOBJ): $(MAGMA_DIR)/testing/testings.h

clean:
	rm -f *.o *~ $(LIBTEST)

cleanexe:
	rm -f $(EXE)

cleanall: clean cleanexe

ifeq ($(FORT), pgfortran)
testing_%.o : testing_%.cuf
	$(FORT) $(FOPTS) $(INC_MAGMA) -c $< -o $@ 
else
testing_%.o : testing_%.f90
	$(FORT) $(FOPTS) $(INC_MAGMA) -c $< -o $@ 
endif

testing_%_f : testing_%_f.o $(FOBJ)
	$(FORT) $(LDOPTS) $< $(FOBJ) -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) \
	$(LIB) 

testing_% : testing_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) \
	$(LIB) 

magma_ssyevd_gpu: magma_ssyevd_gpu.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	$(LIBTEST) $(LIBLAPACKTEST) -L$(MAGMA_DIR)/lib -lmagma \
	$(LIBDIR) \
	$(LIB) 

#
# These two files are provided by cuda
# CUDA 5 needs -DCUBLAS_GFORTRAN or appropriate setting to compile cuda/src/fortran.c bindings
#
fortran.o : OPTS += -DCUBLAS_GFORTRAN
fortran.o : $(CUDADIR)/src/fortran.c
	$(CC) $(OPTS) $(INC_MAGMA) -c $< -o $@

fortran_thunking.o : $(CUDADIR)/src/fortran_thunking.c
	$(CC) $(OPTS) $(INC_MAGMA) -c $< -o $@

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
