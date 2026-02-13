# Makefile for cryo-EM denoising engine (CUDA Fortran + cuDNN)

FC = nvfortran
FFLAGS = -cuda -O2 -Minfo=accel
LIBS = -lcudnn

TARGET = cryo_denoise_engine
SRCS = conv2d_cudnn.cuf mrc_reader.cuf cryo_denoise_engine.cuf

$(TARGET): $(SRCS)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET) *.mod *.o

.PHONY: clean
