# Makefile
#
# author      : sam adams, modified by Mario Juric <mjuric@ias.edu>
#             : modified by S.Bickerton to build on OSX with standard library install
# version     : 0.2
# date        : 20080212
# discription : generic Makefile for making cuda programs
#
BIN               := mytestroot.x

# flags
#COMMONFLAGS       := -D_DEBUG -O0 -g
#COMMONFLAGS       := -m64 
##$(shell root-config --cflags)

CUDA_INSTALL_PATH := /usr/local/cuda
INCLUDES          += -I. -I/home/bellis/NVIDIA_GPU_Computing_SDK/C/common/inc/ $(shell root-config --cflags)
LIBS              := -L/usr/local/cuda/lib/ $(shell root-config --libs) 
CXXFLAGS          := $(COMMONFLAGS)
#LDFLAGS           := -lrt -lm -lcudart -lcutil
#LDFLAGS := -lSystem.B -lm -lcudart -lcutil_x86_64 
LDFLAGS := -lm -lcudart 

# compilers
#NVCC              := nvcc -arch sm_11 $(COMMONFLAGS) --device-emulation
NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_11 $(COMMONFLAGS)

# files
CPP_SOURCES       := mytestroot.cpp  
CU_SOURCES        := mytestroot_gpu.cu
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
mytestroot.o: mytestroot.cpp 
mytestroot_gpu.cu_o: mytestroot_gpu.cu


clean:
	rm -f $(BIN) *.o *.cu_o *.cubin
