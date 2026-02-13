# Makefile for cryo-EM denoising engine (CUDA Fortran + cuDNN)

FC = nvfortran
FFLAGS = -cuda -O2 -Minfo=accel
LIBS = -lcudnn

# Shared modules
MODULES = conv2d_cudnn.cuf mrc_reader.cuf

# Targets
TARGET_V1 = cryo_denoise_engine
TARGET_V2 = cryo_denoise_engine_v2

all: $(TARGET_V1) $(TARGET_V2)

$(TARGET_V1): $(MODULES) cryo_denoise_engine.cuf
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

$(TARGET_V2): conv2d_cudnn_v2.cuf mrc_reader.cuf cryo_denoise_engine_v2.cuf
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET_V1) $(TARGET_V2) *.mod *.o

.PHONY: all clean
