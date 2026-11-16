CXX          := g++
CXXFLAGS     := -std=c++17 -Wall -Wextra -Wpedantic -O2 -Iinclude

BIN_DIR      := bin
SRC_DIR      := src
EXAMPLES_DIR := examples

LIB_SRCS     := $(wildcard $(SRC_DIR)/*.cpp)

TARGETS      := $(BIN_DIR)/miniann \
                $(BIN_DIR)/predict_miniann \
                $(BIN_DIR)/xor_miniann

.PHONY: all clean run-train run-predict run-xor

all: $(TARGETS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/miniann: $(EXAMPLES_DIR)/main.cpp $(LIB_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/predict_miniann: $(EXAMPLES_DIR)/predict_main.cpp $(LIB_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/xor_miniann: $(EXAMPLES_DIR)/xor_main.cpp $(LIB_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

run-train: $(BIN_DIR)/miniann
	./$(BIN_DIR)/miniann

run-predict: $(BIN_DIR)/predict_miniann
	./$(BIN_DIR)/predict_miniann

run-xor: $(BIN_DIR)/xor_miniann
	./$(BIN_DIR)/xor_miniann

clean:
	rm -rf $(BIN_DIR)
