# Quasai Build System - Makefile wrapper around CMake

BUILD_DIR     := build
CMAKE         := cmake
CTEST         := ctest
BUILD_TYPE    ?= Release
JOBS          ?= $(shell nproc)

.DEFAULT_GOAL := help

.PHONY: help all clean debug release rebuild \
        lib tests examples benchmarks \
        run_tests run_benchmarks run_example \
        sanitize format docs open_docs \
        configure check bench

help:
	@echo "Quasai Build System"
	@echo "==================="
	@echo ""
	@echo "Build targets:"
	@echo "  make all           - Build library + tests + examples (default)"
	@echo "  make lib           - Build static library only"
	@echo "  make tests         - Build and run all tests"
	@echo "  make examples      - Build all examples"
	@echo "  make benchmarks    - Build benchmarks (requires BLAS)"
	@echo "  make debug         - Build with debug symbols + QUASAI_DEBUG=ON"
	@echo "  make release       - Build optimized Release (default)"
	@echo "  make sanitize      - Build with ASan+UBSan enabled"
	@echo ""
	@echo "Run targets:"
	@echo "  make run_tests     - Run all unit tests (ctest)"
	@echo "  make run_benchmarks- Run benchmarks (builds if needed)"
	@echo "  make run_example   - Run simple_example"
	@echo "  make run_linreg    - Run linear_regression example"
	@echo ""
	@echo "Utility targets:"
	@echo "  make configure     - Run cmake configure only"
	@echo "  make clean         - Remove build directory"
	@echo "  make rebuild       - Clean + build"
	@echo "  make format        - Format source with clang-format"
	@echo "  make docs          - Generate Doxygen documentation"
	@echo "  make open_docs     - Open docs in browser"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE=Debug|Release|RelWithDebInfo|MinSizeRel (default: Release)"
	@echo "  JOBS=N             - Parallel jobs (default: nproc)"
	@echo "  EXAMPLE=name       - Specific example to run (default: simple_example)"

all: examples tests

configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

lib: configure
	@$(CMAKE) --build $(BUILD_DIR) --target quasai -j $(JOBS)

example_targets := simple_example linear_regression

examples: lib $(example_targets)

$(example_targets):
	@$(CMAKE) --build $(BUILD_DIR) --target $@ -j $(JOBS)

test_targets := core_tests autograd_tests nn_tests ops_tests storage_tests transform_tests

tests: lib $(test_targets)

$(test_targets):
	@$(CMAKE) --build $(BUILD_DIR) --target $@ -j $(JOBS)

benchmarks: configure
	@cd $(BUILD_DIR) && $(CMAKE) .. -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(BUILD_DIR) --target quasai_benchmarks -j $(JOBS)

debug: BUILD_TYPE := Debug
debug: QUASAI_DEBUG := ON
debug: configure
	@cd $(BUILD_DIR) && $(CMAKE) .. -DQUASAI_DEBUG=ON -DCMAKE_BUILD_TYPE=Debug
	@$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

release: BUILD_TYPE := Release
release: all

sanitize: BUILD_TYPE := Debug
sanitize: configure
	@cd $(BUILD_DIR) && $(CMAKE) .. -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
	@$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

rebuild: clean all

clean:
	@rm -rf $(BUILD_DIR)

run_tests: tests
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure -j $(JOBS)

run_benchmarks: benchmarks
	@echo "Running benchmarks..."
	@./$(BUILD_DIR)/benchmarks/quasai_benchmarks $(BENCH_ARGS)

run_example: examples
	@./$(BUILD_DIR)/examples/$(or $(EXAMPLE),simple_example)

run_linreg: examples
	@./$(BUILD_DIR)/examples/linear_regression

format:
	@find include src examples tests -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i

docs:
	@doxygen Doxyfile

open_docs: docs
	@xdg-open docs/doxygen/html/index.html 2>/dev/null || open docs/doxygen/html/index.html 2>/dev/null || echo "Open docs/doxygen/html/index.html manually"

check: run_tests

bench: run_benchmarks