# Makefile
#
# author      : sam adams, modified by Mario Juric <mjuric@ias.edu>
#             : modified by S.Bickerton to build on OSX with standard library install
# version     : 0.2
# date        : 20080212
# discription : generic Makefile for making cuda programs
#
BIN               := parallel_reduction.x

# flags
#COMMONFLAGS       := -D_DEBUG -O0 -g
COMMONFLAGS       := -O2
CUDA_INSTALL_PATH := /usr/local/cuda
INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include -I/Developer/CUDA/common/inc -I/home/bellis/cuda/selfgz600327900/sdk/C/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L/Developer/CUDA/lib -L/home/bellis/cuda/selfgz600327900/sdk/C/common/lib/linux/ -L/home/bellis/cuda/selfgz600327900/sdk/C/lib/
#INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include -I/Developer/CUDA/C/common/inc
#LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L/Developer/CUDA/C/lib
CXXFLAGS          := $(COMMONFLAGS)
LDFLAGS           := -lrt -lm -lcudart -lcutil
#LDFLAGS           := -lSystem.B -lm -lcudart -lcutil

# compilers
#NVCC              := nvcc -arch sm_11 $(COMMONFLAGS) --device-emulation
NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_11 $(COMMONFLAGS)

# files
CPP_SOURCES       := # mc.cpp mc_cpu.cpp deviceQuery.cpp util.cpp
CU_SOURCES        := parallel_reduction.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu 
	$(NVCC) -c $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -fPIC -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

#util.o: util.cpp util.h
#mc.o: mc.cpp util.h
#mc_gpu.cu_o: mc_gpu.cu
#mc_cpu.o: mc_cpu.cpp

clean:
	rm -f $(BIN) *.o *.cu_o *.cubin
