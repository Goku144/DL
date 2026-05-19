##############
# PUBLIC VARS
##############

INC := -Ipublic/inc -Ilib/src -I/usr/local/cuda/include -I/usr/include

###############
# NVCC OPTIONS 
###############

NVCC ?= $(shell command -v nvcc 2>&1)
CUDA_AVAILABLE := $(if $(NVCC),1,0)
FLAG_CUDA := -O3 -Wno-deprecated-gpu-targets -arch=sm_75 -Xcompiler -fno-exceptions -diag-suppress 550
LIBS_CUDA := -lcudnn -lcusparse -lcusolver -lcurand -lcublasLt -lcublas -lcudart
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))

#############
# CONDITIONS
#############

ifeq ($(CUDA_AVAILABLE),0)
$(info visit the website to install: https://developer.nvidia.com/cuda/toolkit)
$(error because CUDA is not available compiling .cu sources)
endif

############
# Build App
############

app: app/build/app
	@mkdir -p public/checkpoints
	@./$< 

app/build/app: app/bin/app.o $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ $(LIBS_CUDA) -o $@

app/bin/app.o: app/src/app.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Build Lib
############

lib: $(OBJS_CUDA)

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Utility
############

dataset:
	@python3 -m venv .venv
	@.venv/bin/pip install GitPython
	@.venv/bin/python public/target/src/DataSet.py

clean:
	rm -rf lib/bin app/bin app/build

.PHONY: app run lib clean