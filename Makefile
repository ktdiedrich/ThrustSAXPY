# Compiler and flags
NVCC = nvcc
GPP = g++
CXXFLAGS = -std=c++11 -ggdb3
NVCCFLAGS = --expt-extended-lambda -std=c++11 -G -g

# Check if nvcc is available
NVCC_CHECK := $(shell command -v $(NVCC) 2> /dev/null)
ifeq ($(NVCC_CHECK),)
$(error "nvcc not found. Please ensure CUDA is installed and nvcc is in your PATH.")
endif

# Check if g++ is available
GPP_CHECK := $(shell command -v $(GPP) 2> /dev/null)
ifeq ($(GPP_CHECK),)
$(error "g++ not found. Please ensure g++ is installed and in your PATH.")
endif
# Check if CUDA is installed
CUDA_PATH := $(shell nvcc --version | grep -oP '(?<=V)[0-9]+\.[0-9]+')
ifeq ($(CUDA_PATH),)
$(error "CUDA not found. Please ensure CUDA is installed and nvcc is in your PATH.")
endif
# Check if Thrust is available
THRUST_PATH := $(shell find /usr/include -name thrust -type d 2> /dev/null)
ifeq ($(THRUST_PATH),)
$(error "Thrust not found. Please ensure Thrust is installed with CUDA.")
endif
# Set CUDA library and include paths	
# Adjust these paths according to your CUDA installation
CUDA_LIB_PATH = -L/lib/x86_64-linux-gnu
CUDA_INCLUDE_PATH = -I/usr/include

LINK_FLAGS = -lcudart $(CUDA_LIB_PATH) $(CUDA_INCLUDE_PATH)


# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files
SAXPY_SRC = $(SRC_DIR)/thrustSAXPY.cu
DETECT_SRC = $(SRC_DIR)/detect_compute_capability.cpp
NN_SRC = $(SRC_DIR)/thrustNN.cu

# Output binaries
SAXPY_BIN = $(BIN_DIR)/ThrustSAXPY
DETECT_BIN = $(BIN_DIR)/detect_compute_capability
NN_BIN = $(BIN_DIR)/thrustNN

# Default compute capability (fallback)
DEFAULT_COMPUTE_CAPABILITY = 30

# Rule to detect compute capability
COMPUTE_CAPABILITY = $(shell $(DETECT_BIN) 2>/dev/null || echo $(DEFAULT_COMPUTE_CAPABILITY))

# Targets
all: $(DETECT_BIN) $(SAXPY_BIN) $(NN_BIN)

# Compile detect_compute_capability
$(DETECT_BIN): $(DETECT_SRC)
	@mkdir -p $(BIN_DIR)
	$(GPP) $(CXXFLAGS) -o $@ $< $(LINK_FLAGS)

# Compile ThrustSAXPY
$(SAXPY_BIN): $(SAXPY_SRC) $(DETECT_BIN)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(GPP) -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=\"sm_$(COMPUTE_CAPABILITY),compute_$(COMPUTE_CAPABILITY)\" $< -o $@


# Compile thrustNN
$(NN_BIN): $(NN_SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(GPP) -gencode=arch=compute_$(COMPUTE_CAPABILITY),code=\"sm_$(COMPUTE_CAPABILITY),compute_$(COMPUTE_CAPABILITY)\" $< -o $@


# Clean up binaries
clean:
	rm -rf $(BIN_DIR)/*