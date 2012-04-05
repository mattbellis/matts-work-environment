# Makefile
#
# author      : Matt Bellis
# date        : Jan., 2011
# discription : generic Makefile for making cuda programs
#
BIN               := example_1.x 

# flags
#COMMONFLAGS       := -D_DEBUG -O0 -g
#COMMONFLAGS       := -m64 
##$(shell root-config --cflags)

#CUDA_INSTALL_PATH := /opt/cuda/
#CUDA_INSTALL_PATH := /usr/local/cuda/
CUDA_INSTALL_PATH := /usr/

SDK_INSTALL_PATH := /home/bellis/NVIDIA_GPU_Computing_SDK
#INCLUDES          += -I. -I/home/bellis/NVIDIA_GPU_Computing_SDK/C/common/inc/ $(shell root-config --cflags)
#LIBS              := -L/usr/local/cuda/lib/ $(shell root-config --libs) 
INCLUDES          += -I. -I/u/br/bellis/CUDA/SDK/C/common/inc/ 
LIBS              := -L$(CUDA_INSTALL_PATH)/lib64/ 
CXXFLAGS          := $(COMMONFLAGS)
#LDFLAGS           := -lrt -lm -lcudart -lcutil
#LDFLAGS := -lSystem.B -lm -lcudart -lcutil_x86_64 
#LDFLAGS := -lm -lcudart 
LDFLAGS := -lcudart

# compilers
#NVCC              := nvcc -arch sm_11 $(COMMONFLAGS) --device-emulation
NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_11 $(COMMONFLAGS)

# files
#CPP_SOURCES       := mytestroot.cpp  
CPP_SOURCES       := 
CU_SOURCES        := example_1.cu 
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
#mytestroot.o: mytestroot.cpp 
#mytestroot_gpu.cu_o: mytestroot_gpu.cu


clean:
	rm -f $(BIN) *.o *.cu_o *.cubin
