NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -std=c++14 -Wno-deprecated-gpu-targets

SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

TARGET = $(BIN_DIR)/cuda_edge_filter.exe
SRC = $(SRC_DIR)/main.cu

all: directories $(TARGET)

directories:
	mkdir -p $(BIN_DIR) $(DATA_DIR)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)/* $(DATA_DIR)/*.csv

help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."

.PHONY: all directories clean help
