# Makefile
#

BIN               := minuit_simple

CUDA_INSTALL_PATH := /usr/local/cuda
#INCLUDES          += -I. -I/home/bellis/NVIDIA_GPU_Computing_SDK/C/common/inc/ $(shell root-config --cflags)
INCLUDES          += -I. -I/home/bellis/NVIDIA_GPU_Computing_SDK/C/common/inc/ -m32 -I$(shell root-config --incdir)
LIBS              := -L/usr/local/cuda/lib/ $(shell root-config --libs) -lMinuit
CXXFLAGS          := $(COMMONFLAGS)
LDFLAGS := -lm -lcudart 

# compilers
NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc -arch sm_11 $(COMMONFLAGS)

# Need this to get around gcc4.4 on my machine.
NVCCFLAGS         := --compiler-options -fno-inline -Xcompiler -D__builtin_stdarg_start=__builtin_va_start

# files
#CPP_SOURCES       := mytestroot.cpp  
CU_SOURCES        := $(BIN).cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) -c $(INCLUDES) -o $@ $< $(NVCCFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -fPIC -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(BIN).cu_o: $(BIN).cu

clean:
	rm -f $(BIN) *.o *.cu_o *.cubin
