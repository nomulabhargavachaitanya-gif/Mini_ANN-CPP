CXX          := g++
CXXFLAGS     := -std=c++17 -Wall -Wextra -Wpedantic -O2 -Iinclude -MMD -MP

BIN_DIR      := bin
BUILD_DIR    := build
SRC_DIR      := src
EXAMPLES_DIR := examples
TESTS_DIR    := tests

LIB_SRCS     := $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS     := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/src/%.o,$(LIB_SRCS))

EXAMPLE_SRCS := $(wildcard $(EXAMPLES_DIR)/*.cpp)
EXAMPLE_OBJS := $(patsubst $(EXAMPLES_DIR)/%.cpp,$(BUILD_DIR)/examples/%.o,$(EXAMPLE_SRCS))

TEST_SRCS    := $(wildcard $(TESTS_DIR)/*.cpp)
TEST_OBJS    := $(patsubst $(TESTS_DIR)/%.cpp,$(BUILD_DIR)/tests/%.o,$(TEST_SRCS))

ALL_OBJS     := $(LIB_OBJS) $(EXAMPLE_OBJS) $(TEST_OBJS)
DEPS         := $(ALL_OBJS:.o=.d)

TARGETS      := $(BIN_DIR)/miniann \
                $(BIN_DIR)/predict_miniann \
                $(BIN_DIR)/xor_miniann

TEST_BIN     := $(BIN_DIR)/test_miniann

.PHONY: all test clean run-train run-predict run-xor

all: $(TARGETS)

$(BIN_DIR) $(BUILD_DIR)/src $(BUILD_DIR)/examples $(BUILD_DIR)/tests:
	@mkdir -p $@

$(BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)/src
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/examples/%.o: $(EXAMPLES_DIR)/%.cpp | $(BUILD_DIR)/examples
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TESTS_DIR)/%.cpp | $(BUILD_DIR)/tests
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/miniann: $(BUILD_DIR)/examples/main.o $(LIB_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/predict_miniann: $(BUILD_DIR)/examples/predict_main.o $(LIB_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/xor_miniann: $(BUILD_DIR)/examples/xor_main.o $(LIB_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_BIN): $(BUILD_DIR)/tests/test_miniann.o $(LIB_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

run-train: $(BIN_DIR)/miniann
	./$(BIN_DIR)/miniann

run-predict: $(BIN_DIR)/predict_miniann
	./$(BIN_DIR)/predict_miniann

run-xor: $(BIN_DIR)/xor_miniann
	./$(BIN_DIR)/xor_miniann

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)

-include $(DEPS)
