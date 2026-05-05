CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall

FFTW_PREFIX := $(shell brew --prefix fftw 2>/dev/null || echo /usr/local)
FFTW_INC    := $(FFTW_PREFIX)/include
FFTW_LIB    := $(FFTW_PREFIX)/lib

SRC_DIR  := ../../src
KISS_DIR := kissfft

# ── KissFFT fetch ────────────────────────────────────────────────────────────
KISS_SRCS := $(KISS_DIR)/kiss_fft.c
KISS_HDRS := $(KISS_DIR)/kiss_fft.h $(KISS_DIR)/_kiss_fft_guts.h $(KISS_DIR)/kiss_fft_log.h

$(KISS_DIR):
	mkdir -p $(KISS_DIR)

$(KISS_HDRS) $(KISS_SRCS): | $(KISS_DIR)
	curl -sSfL -o $(KISS_DIR)/kiss_fft.h \
	    https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft.h
	curl -sSfL -o $(KISS_DIR)/_kiss_fft_guts.h \
	    https://raw.githubusercontent.com/mborgerding/kissfft/master/_kiss_fft_guts.h
	curl -sSfL -o $(KISS_DIR)/kiss_fft.c \
	    https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft.c
	curl -sSfL -o $(KISS_DIR)/kiss_fft_log.h \
	    https://raw.githubusercontent.com/mborgerding/kissfft/master/kiss_fft_log.h

# ── Object files ─────────────────────────────────────────────────────────────

# Original FFTW engine compiled under renamed symbols
orig_engine.o: $(SRC_DIR)/frft_engine.cpp $(SRC_DIR)/frft_engine.h
	$(CXX) $(CXXFLAGS) \
	    -DFRFTEngine=FRFTEngineOrig \
	    -DFRFTEngineManager=FRFTEngineManagerOrig \
	    -I$(FFTW_INC) \
	    -c $< -o $@

# KissFFT engine (double precision)
wasm_engine.o: frft_engine_wasm.cpp frft_engine_wasm.h $(KISS_SRCS) $(KISS_HDRS)
	$(CXX) $(CXXFLAGS) \
	    -Dkiss_fft_scalar=double \
	    -I. \
	    -c frft_engine_wasm.cpp -o $@

kiss_fft.o: $(KISS_DIR)/kiss_fft.c $(KISS_HDRS)
	$(CXX) $(CXXFLAGS) \
	    -Dkiss_fft_scalar=double \
	    -I$(KISS_DIR) \
	    -c $(KISS_DIR)/kiss_fft.c -o $@

test_engine.o: test_engine.cpp frft_engine_wasm.h $(SRC_DIR)/frft_engine.h $(KISS_HDRS)
	$(CXX) $(CXXFLAGS) \
	    -Dkiss_fft_scalar=double \
	    -I. -I$(SRC_DIR) -I$(FFTW_INC) \
	    -c test_engine.cpp -o $@

# ── Executable ───────────────────────────────────────────────────────────────
test_engine: test_engine.o orig_engine.o wasm_engine.o kiss_fft.o
	$(CXX) $(CXXFLAGS) $^ \
	    -L$(FFTW_LIB) -lfftw3 \
	    -o $@

# ── Targets ──────────────────────────────────────────────────────────────────
.PHONY: all test clean

all: test_engine

test: test_engine
	./test_engine

clean:
	rm -f *.o test_engine
	rm -rf $(KISS_DIR)
