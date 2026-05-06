# ===== Configuration =====
BUILD_DIR := build
CMAKE := cmake
CTEST := ctest

BUILD_TYPE ?= Release

# ===== Default =====
.PHONY: all
all: build

# ===== Clean =====
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)


# ===== Build (normal) =====
.PHONY: build
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(CMAKE) --build $(BUILD_DIR) -j

# ===== Rebuild (clean + build) =====
.PHONY: rebuild
rebuild: clean build

# ===== Build with benchmarks enabled =====
.PHONY: build_with_benchmark
build_with_benchmark:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_BENCHMARKS=ON \
		-DBUILD_TORCH_BENCHMARKS=ON
	@$(CMAKE) --build $(BUILD_DIR) -j

# ===== Run tests =====
.PHONY: run_tests
run_tests: build
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure

# ===== Run benchmarks (build + execute benchmark binary) =====
.PHONY: run_benchmark
run_benchmark: build_with_benchmark
	@echo "Running benchmarks..."
	@./$(BUILD_DIR)/benchmarks/quasai_benchmarks

# ==== Generate documentation (using Doxygen) ====
.PHONY: docs
docs:
	@doxygen Doxyfile

# ==== Open documentation in browser ====
.PHONY: open_docs
open_docs: docs
	@xdg-open docs/doxygen/html/index.html
