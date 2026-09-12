CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
SAMPLING_TEST := tests/test_sampling
GLM53_KDA_TEST := tests/test_glm53_kda
QWEN4EXP_HC_TEST := tests/test_qwen4exp_hc_norm
GLM53_KDA_ROCM_TEST := tests/test_glm53_kda_rocm
QWEN4EXP_GDN_TEST := tests/test_qwen4exp_gdn
QWEN4EXP_MOE_TEST := tests/test_qwen4exp_moe
QWEN4EXP_QSA_TEST := tests/test_qwen4exp_qsa
QWEN4EXP_GDN_REF_TEST := tests/test_qwen4exp_gdn_ref
QWEN4EXP_QSA_REF_TEST := tests/test_qwen4exp_qsa_ref
QWEN4EXP_GRAPH_TEST := tests/test_qwen4exp_graph
QWEN4EXP_PLE_KERNEL_TEST := tests/test_qwen4exp_ple_kernels
QWEN4EXP_MTP_TEST := tests/test_qwen4exp_mtp
QWEN4EXP_MODEL_REGISTRATION_TEST := tests/test_qwen4exp_model_registration
QWEN4EXP_PARITY_TEST := tests/test_qwen4exp_parity

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

# SAN=1 builds the C and Objective-C objects and every link with
# AddressSanitizer.  For host-heap faults that one allocator happens to absorb
# and another traps on -- a graph-test overflow that ran green on Metal and
# aborted on CUDA with "malloc(): corrupted top size" is what this is for.
# Use it on a whole tree: `make clean && make SAN=1 test-qwen4exp-graph`, since
# an ASan object cannot be linked against a plain one.
#
# On Darwin the graph test's support-model case forks a child that is MEANT to
# die, and Apple's Metal driver segfaults in its own teardown there under ASan
# (AGX::Heap::deallocateImpl).  That is an ASan/driver artifact in a child that
# was going to terminate anyway, not a finding: the parent reads the refusal and
# the check passes.  Grep the log for heap-buffer-overflow, use-after-free and
# friends rather than for the word AddressSanitizer.
ifeq ($(SAN),1)
SAN_FLAGS := -fsanitize=address -fno-omit-frame-pointer
CFLAGS += $(SAN_FLAGS)
OBJCFLAGS += $(SAN_FLAGS)
LDFLAGS += $(SAN_FLAGS)
LDLIBS ?= -lm -pthread $(SAN_FLAGS)
else
LDLIBS ?= -lm -pthread
endif
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
# Every GPU kernel test links one backend object plus ds4_image.o, which carries
# ds4_deepseek4_attention_bounds -- the backend objects reference it from the
# vision mixed-batch mask, so neither links on its own.  One definition here
# instead of a per-test ifeq; GPU_TEST_LINK_SAFE is the same link without
# -ffast-math, for the tests that compare against an IEEE f32 reference.
GPU_TEST_OBJS := ds4_metal.o ds4_image.o
GPU_TEST_LINK = $(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
GPU_TEST_LINK_SAFE = $(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(METAL_LDLIBS)
# `make test` runs the qwen4exp device kernel tests on Metal, where a device is
# always present and both backends have been exercised.  CUDA and ROCm hosts run
# them one at a time through `make test-qwen4exp-gdn` / `make test-qwen4exp-moe`;
# the CUDA twins have not yet run on a GPU, so they stay out of `make test`
# there.
GPU_KERNEL_TESTS := $(QWEN4EXP_GDN_TEST) $(QWEN4EXP_MOE_TEST) $(QWEN4EXP_HC_TEST) $(QWEN4EXP_QSA_TEST) $(QWEN4EXP_PLE_KERNEL_TEST) $(QWEN4EXP_GRAPH_TEST)
GPU_KERNEL_TEST_RUN := ./$(QWEN4EXP_GDN_TEST) && ./$(QWEN4EXP_MOE_TEST) && sh tests/qwen4exp_moe_mutants.sh && DS4_METAL_MATH_SAFE=1 ./$(QWEN4EXP_HC_TEST) && ./$(QWEN4EXP_QSA_TEST) && DS4_METAL_MATH_SAFE=1 ./$(QWEN4EXP_PLE_KERNEL_TEST) && sh tests/qwen4exp_hc_mutants.sh && sh tests/qwen4exp_ple_mutants.sh && ./tests/test_qwen4exp_graph.sh
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp_cpu.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# The Qwen4-Exp kernels are their own translation unit so they can drop fast
# math: --use_fast_math would substitute __cosf/__sinf in the QSA rope, whose
# angle reaches the position index itself, and flush the GDN decays and the
# QSA softmax terms to zero.  See the header of ds4_cuda_qwen4exp.cu.
QWEN4EXP_NVCCFLAGS := $(filter-out --use_fast_math,$(NVCCFLAGS)) -ftz=false -prec-div=true -prec-sqrt=true
# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_cuda_qwen4exp.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp_cpu.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_HOST_CFLAGS ?= -fPIC
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt -lrocblas
ROCM_MMQ_Y ?= 64
ROCM_MMQ_FLAGS := $(ROCM_CFLAGS) -std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=$(ROCM_MMQ_Y) $(MMQ_INCLUDES)
ROCM_MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o cuda/mmq/quantize.rocm.o cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o cuda/mmq/d2r_stubs.rocm.o
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
GPU_TEST_OBJS := ds4_cuda.o ds4_cuda_qwen4exp.o ds4_image.o $(MMQ_OBJS)
GPU_TEST_LINK = $(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
GPU_TEST_LINK_SAFE = $(GPU_TEST_LINK)
GPU_KERNEL_TESTS :=
GPU_KERNEL_TEST_RUN := @echo "qwen4exp device kernel tests: skipped, need a verified GPU host (make test-qwen4exp-gdn, -moe, -hc, -qsa, -ple-kernels)"
endif

# The MTP cycle test appends rather than assigns: it needs no GPU, because its
# QSA/MoE hooks are documented pass-throughs and its ds4_gpu_tensor is host
# memory, so it runs on every host alongside the tests that do need a device.
GPU_KERNEL_TESTS += $(QWEN4EXP_MTP_TEST)

.PHONY: all help clean test test-qwen4exp test-qwen4exp-ple test-qwen4exp-ple-kernels test-qwen4exp-ple-kernel-mutants test-qwen4exp-gdn test-qwen4exp-gdn-ref test-qwen4exp-qsa-ref test-qwen4exp-moe test-qwen4exp-moe-mutants test-qwen4exp-cuda-moe test-qwen4exp-hc test-qwen4exp-hc-mutants test-qwen4exp-hc-fuse-mutants test-qwen4exp-qsa test-qwen4exp-mtp test-qwen4exp-graph test-qwen4exp-parity test-rocm test-glm53-kda-rocm test-metal-session-batch test-mxfp4-cuda test-mxfp4-rocm test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm

ifeq ($(UNAME_S),Darwin)
.PHONY: metal-decode-schedule-bench metal-prefill-variant-bench check-mxfp4-half-lut

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make test-qwen4exp-ple  Run the Qwen4-Exp PLE n-gram table and hash-reference tests"
	@echo "  make test-qwen4exp-ple-kernels  Build and run the Qwen4-Exp PLE gate and convolution kernel tests"
	@echo "  make test-qwen4exp-gdn  Build and run the Qwen4exp gated delta net kernel test"
	@echo "  make test-qwen4exp-gdn-ref  Check the GDN kernel against a double-precision reference"
	@echo "  make test-qwen4exp-qsa-ref  Check the dense QSA path against a double-precision reference"
	@echo "  make test-qwen4exp-moe  Build and run the Qwen4exp routed MoE kernel test"
	@echo "  make test-qwen4exp-cuda-moe  Check the CUDA expert dequantisers on the host"
	@echo "  make test-qwen4exp-hc   Build and run the Qwen4exp HC/norm/rope/head kernel test"
	@echo "  make test-qwen4exp-qsa  Run the Qwen4-Exp QSA kernel tests"
	@echo "  make test-qwen4exp-graph  Run the Qwen4-Exp end-to-end graph test"
	@echo "  make test-qwen4exp-mtp  Build and run the Qwen4exp MTP cycle test"
	@echo "  make test-qwen4exp-parity  Compare our op references against the ds4-metal parity reference"
	@echo "  make metal-decode-schedule-bench  Build the balanced Metal decode schedule benchmark"
	@echo "  make metal-prefill-variant-bench  Build the balanced Metal prefill variant benchmark"
	@echo "  make check-mxfp4-half-lut  Verify the checked-in MXFP4 half LUT matches the generator"
	@echo "  make test-mxfp4-metal  Check the MXFP4 half LUT, then run Metal MXFP4 exactness tests"
	@echo "  make test-qwen4exp-hc  Run the qwen4exp HC/norm/rope/embedding/head kernel tests"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.c ds4.h ds4_distributed.h ds4_tp.h $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_official.c $(CORE_OBJS) rax.o ds4_gpu_args.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

speed-bench/metal_decode_schedule_bench.o: speed-bench/metal_decode_schedule_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_decode_schedule_bench: speed-bench/metal_decode_schedule_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-decode-schedule-bench: speed-bench/metal_decode_schedule_bench

speed-bench/metal_prefill_variant_bench.o: speed-bench/metal_prefill_variant_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_prefill_variant_bench: speed-bench/metal_prefill_variant_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-prefill-variant-bench: speed-bench/metal_prefill_variant_bench

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

check-mxfp4-half-lut:
	python3 metal/generate_mxfp4_half_lut.py --check

test-mxfp4-metal: check-mxfp4-half-lut tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make test-mxfp4-rocm     Build and run the synthetic ROCm MXFP4 MoE test"
	@echo "  make test-rocm           Core regression suite on ROCm-only hosts"
	@echo "  make test-qwen4exp-hc    Run the qwen4exp HC/norm/rope/embedding/head kernel tests"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make test-qwen4exp-ple   Run the Qwen4-Exp PLE n-gram table tests"
	@echo "  make test-qwen4exp-ple-kernels  Build and run the Qwen4-Exp PLE gate and convolution kernel tests"
	@echo "  make test-qwen4exp-gdn   Build and run the Qwen4exp gated delta net kernel test"
	@echo "  make test-qwen4exp-moe   Build and run the Qwen4exp routed MoE kernel test"
	@echo "  make test-qwen4exp-cuda-moe  Check the CUDA expert dequantisers on the host"
	@echo "  make test-qwen4exp-hc    Build and run the Qwen4exp HC/norm/rope/head kernel test"
	@echo "  make test-qwen4exp-qsa   Run the Qwen4-Exp QSA kernel tests"
	@echo "  make test-qwen4exp-graph Run the Qwen4-Exp end-to-end graph test"
	@echo "  make test-qwen4exp-mtp   Build and run the Qwen4exp MTP cycle test"
	@echo "  make test-qwen4exp-parity  Compare our op references against the ds4-metal parity reference"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CUDA_ARCH="$(if $(strip $(CUDA_ARCH)),$(strip $(CUDA_ARCH)),sm_121)"

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

# Core regression suite for ROCm-only hosts: the CUDA-specific binaries
# (tests/test_sampling, the CUDA session/mixed-batch oracles) are not part
# of this target; run them through `make test` / `make cuda-regression` on
# CUDA hosts.  Everything else mirrors `make test`.
test-rocm:
	$(MAKE) -B ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test \
		tests/test_layer_pack tests/test_qwen4exp_ple \
		tests/test_qwen4exp_ple_hash_ref \
		$(QWEN4EXP_PARITY_TEST) \
		tests/test_engine_mgpu_placement tests/test_gpu_args \
		ds4 ds4-server ds4-bench ds4-agent \
		CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_qwen4exp_ple
	python3 tests/test_qwen4exp_ple.py
	./$(QWEN4EXP_PARITY_TEST)
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	sh tests/qwen4exp_cuda_moe_parity.sh

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) $(ROCM_HOST_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen4exp.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp.inc ds4_qwen4exp_ple.h ds4_qwen4exp_graph.h ds4_qwen4exp_graph.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h ds4_qwen4exp_model_registration.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_qwen4exp_ple.o: ds4_qwen4exp_ple.c ds4_qwen4exp_ple.h
	$(CC) $(CFLAGS) -c -o $@ ds4_qwen4exp_ple.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen4exp.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp.inc ds4_qwen4exp_ple.h ds4_qwen4exp_graph.h ds4_qwen4exp_graph.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp_hc_ref.h ds4_qwen4exp_hc_host.inc ds4_qwen4exp_ple_host.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

tests/test_glm53_kda.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_kda.c

tests/test_qwen4exp_qsa.o: tests/test_qwen4exp_qsa.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4exp_qsa.c

tests/test_glm53_vision_engine.o: tests/test_glm53_vision_engine.c ds4.h ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_vision_engine.c

tests/test_glm53_vision_engine: tests/test_glm53_vision_engine.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_glm53_vision_prompt.o: tests/test_glm53_vision_prompt.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_vision_prompt.c

tests/test_glm53_vision_prompt: tests/test_glm53_vision_prompt.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(GLM53_KDA_TEST): tests/test_glm53_kda.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK)

.PHONY: test-glm53-kda
test-glm53-kda: $(GLM53_KDA_TEST)
	./$(GLM53_KDA_TEST)

tests/test_qwen4exp_gdn.o: tests/test_qwen4exp_gdn.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4exp_gdn.c

$(QWEN4EXP_GDN_TEST): tests/test_qwen4exp_gdn.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK)

.PHONY: test-qwen4exp-gdn
test-qwen4exp-gdn: $(QWEN4EXP_GDN_TEST)
	./$(QWEN4EXP_GDN_TEST)

# The double-precision reference pair.  Built WITHOUT -ffast-math: the whole
# point is an IEEE reference the kernel is measured against, and fast math is
# free to reassociate the reference's own sums.
tests/test_qwen4exp_gdn_ref.o: tests/test_qwen4exp_gdn_ref.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_qwen4exp_gdn_ref.c

$(QWEN4EXP_GDN_REF_TEST): tests/test_qwen4exp_gdn_ref.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK_SAFE)

.PHONY: test-qwen4exp-gdn-ref
test-qwen4exp-gdn-ref: $(QWEN4EXP_GDN_REF_TEST)
	./$(QWEN4EXP_GDN_REF_TEST)

tests/test_qwen4exp_qsa_ref.o: tests/test_qwen4exp_qsa_ref.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_qwen4exp_qsa_ref.c

$(QWEN4EXP_QSA_REF_TEST): tests/test_qwen4exp_qsa_ref.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK_SAFE)

.PHONY: test-qwen4exp-qsa-ref
test-qwen4exp-qsa-ref: $(QWEN4EXP_QSA_REF_TEST)
	./$(QWEN4EXP_QSA_REF_TEST)

# Built without -ffast-math: a kernel that decodes a superblock as the wrong
# type overflows float32, and the error bands must see the resulting non-finite
# values rather than have the comparison folded away under finite-math-only.
tests/test_qwen4exp_moe.o: tests/test_qwen4exp_moe.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_qwen4exp_moe.c

# ds4_image.o supplies ds4_deepseek4_attention_bounds, which the Metal and CUDA
# backends reference from the mixed-batch mask helper.
$(QWEN4EXP_MOE_TEST): tests/test_qwen4exp_moe.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK)

.PHONY: test-qwen4exp-moe
test-qwen4exp-moe: $(QWEN4EXP_MOE_TEST)
	./$(QWEN4EXP_MOE_TEST)

# Proves the F32 router projection's exactness sweep bites: each mutant
# rewrites one line of matmul_f32_warp_tile_kernel and must make
# ./tests/test_qwen4exp_moe --router-f32 fail.  Two commutative controls are
# expected to pass; see the script header.  Rebuilds ds4_cuda.o per mutant.
.PHONY: test-qwen4exp-router-f32-mutants
test-qwen4exp-router-f32-mutants: $(QWEN4EXP_MOE_TEST)
	CUDA_ARCH=$(CUDA_ARCH) bash tests/qwen4exp_router_f32_mutants.sh

# Proves the test above bites: each mutant rewrites one kernel line through
# DS4_METAL_QWEN4EXP_MOE_SOURCE and must make the test fail.
.PHONY: test-qwen4exp-moe-mutants
test-qwen4exp-moe-mutants: $(QWEN4EXP_MOE_TEST)
	sh tests/qwen4exp_moe_mutants.sh

# The CUDA half of the same decode, checked on the host: the script lifts the
# K-quant accessors out of ds4_cuda_qwen4exp.cu and runs them here.  No CUDA
# toolchain and no GPU, so it runs on every host.
.PHONY: test-qwen4exp-cuda-moe
test-qwen4exp-cuda-moe: tests/test_qwen4exp_cuda_moe.c tests/qwen4exp_cuda_moe_parity.sh ds4_cuda_qwen4exp.cu
	sh tests/qwen4exp_cuda_moe_parity.sh

# The qwen4exp op tests assert exactness wherever the arithmetic is exact in
# f32, so the reference side is built without -ffast-math (which would let the
# host turn 1/sqrt into an approximation) and the GPU side runs against the
# safe-math shader library.  The bands hold either way; only the zero-tolerance
# comparisons need IEEE semantics on both sides.
tests/test_qwen4exp_hc_norm.o: tests/test_qwen4exp_hc_norm.c ds4_gpu.h ds4_qwen4exp_hc_ref.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_qwen4exp_hc_norm.c

# ds4_image.o carries ds4_deepseek4_attention_bounds, which the backend
# objects reference from the vision mixed-batch mask.
$(QWEN4EXP_HC_TEST): tests/test_qwen4exp_hc_norm.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK_SAFE)

.PHONY: test-qwen4exp-hc
test-qwen4exp-hc: $(QWEN4EXP_HC_TEST)
	DS4_METAL_MATH_SAFE=1 ./$(QWEN4EXP_HC_TEST)
	./$(QWEN4EXP_HC_TEST)
	DS4_QWEN4EXP_NO_HC_STAGED=1 ./$(QWEN4EXP_HC_TEST)
	DS4_QWEN4EXP_NO_HC_NQI_STAGED=1 ./$(QWEN4EXP_HC_TEST)

# Proves the inject checks bite: each mutant rewrites one line of the inject
# kernel through DS4_METAL_QWEN4EXP_HC_SOURCE and must make the test fail.
.PHONY: test-qwen4exp-hc-mutants
test-qwen4exp-hc-mutants: $(QWEN4EXP_HC_TEST)
	sh tests/qwen4exp_hc_mutants.sh

# Proves the FUSED mixer's zero-tolerance equivalence check bites.  Each mutant
# rewrites one line of the fused kernels in ds4_cuda_qwen4exp.cu and rebuilds,
# so this is minutes rather than seconds and is not part of test-qwen4exp-hc.
# The last few mutants are deliberate no-ops and must SURVIVE.
.PHONY: test-qwen4exp-hc-fuse-mutants
test-qwen4exp-hc-fuse-mutants: $(QWEN4EXP_HC_TEST)
	sh tests/qwen4exp_hc_fuse_mutants.sh

# The PLE kernel test asserts bit exactness on the rolling convolution window
# and on the in-place gate, so the reference side is built without -ffast-math
# and the GPU side runs against the safe-math shader library, the way the HC
# test above does.
tests/test_qwen4exp_ple_kernels.o: tests/test_qwen4exp_ple_kernels.c ds4_gpu.h ds4_qwen4exp_hc_ref.h ds4_qwen4exp_ple_ref.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_qwen4exp_ple_kernels.c

$(QWEN4EXP_PLE_KERNEL_TEST): tests/test_qwen4exp_ple_kernels.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK_SAFE)

.PHONY: test-qwen4exp-ple-kernels
test-qwen4exp-ple-kernels: $(QWEN4EXP_PLE_KERNEL_TEST)
	DS4_METAL_MATH_SAFE=1 ./$(QWEN4EXP_PLE_KERNEL_TEST)
	sh tests/qwen4exp_ple_mutants.sh

# Proves the test above bites from the other side: each mutant rewrites one
# kernel line through DS4_METAL_QWEN4EXP_PLE_SOURCE and must make it fail.
.PHONY: test-qwen4exp-ple-kernel-mutants
test-qwen4exp-ple-kernel-mutants: $(QWEN4EXP_PLE_KERNEL_TEST)
	sh tests/qwen4exp_ple_mutants.sh
# The QSA block runs against a CPU f32 reference at the production shape.  It
# needs a GPU backend, so it links the Metal or CUDA object the way the KDA
# test does and reports a skip when no device is present.
$(QWEN4EXP_QSA_TEST): tests/test_qwen4exp_qsa.o $(GPU_TEST_OBJS)
	$(GPU_TEST_LINK)

# The graph test drives ds4.c itself, so it needs a build of ds4.c with the test
# hooks AND a live GPU backend: ds4_cpu_test_hooks.o is -DDS4_NO_GPU and would
# stub out every kernel the graph calls.
ds4_metal_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h ds4_qwen4exp.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp.inc ds4_qwen4exp_ple.h ds4_qwen4exp_graph.h ds4_qwen4exp_graph.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h ds4_qwen4exp_model_registration.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_qwen4exp_graph.o: tests/test_qwen4exp_graph.c ds4.h ds4_qwen4exp_mtp.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

# The graph test drives ds4.c against a live backend, so its object list and
# its linker are per platform: ds4_metal_test_hooks.o and ds4_metal.o under
# Objective-C ARC on Darwin, the CUDA test-hooks object and the two CUDA units
# through nvcc elsewhere.  The old single rule named ds4_metal.o unconditionally
# and could not build on Linux at all.
ifeq ($(UNAME_S),Darwin)
$(QWEN4EXP_GRAPH_TEST): tests/test_qwen4exp_graph.o ds4_metal_test_hooks.o ds4_metal.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
$(QWEN4EXP_GRAPH_TEST): tests/test_qwen4exp_graph.o ds4_cuda_test_hooks.o ds4_cuda.o ds4_cuda_qwen4exp.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-qwen4exp-graph
test-qwen4exp-graph: $(QWEN4EXP_GRAPH_TEST) ds4
	./tests/test_qwen4exp_graph.sh

test-qwen4exp-qsa: $(QWEN4EXP_QSA_TEST)
	./$(QWEN4EXP_QSA_TEST)

tests/test_glm53_kda_rocm.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

$(GLM53_KDA_ROCM_TEST): tests/test_glm53_kda_rocm.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-glm53-kda-rocm: $(GLM53_KDA_ROCM_TEST)
	./$(GLM53_KDA_ROCM_TEST)

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_cuda_qwen4exp.cuh ds4_gpu_mgpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Every Qwen4-Exp CUDA kernel, built without fast math (QWEN4EXP_NVCCFLAGS).
ds4_cuda_qwen4exp.o: ds4_cuda_qwen4exp.cu ds4_cuda_qwen4exp.cuh ds4_gpu.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_gpu_mgpu.h ds4_qwen4exp_hc_ref.h ds4_qwen4exp_hc_host.inc ds4_qwen4exp_ple_host.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h
	$(NVCC) $(QWEN4EXP_NVCCFLAGS) -c -o $@ ds4_cuda_qwen4exp.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

cuda/mmq/ds4_ggml_stubs.rocm.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/ds4_mmq.rocm.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/quantize.rocm.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmid.rocm.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmvq.rocm.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/d2r_stubs.rocm.o: cuda/mmq/test/d2r_stubs.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

tests/test_mxfp4_rocm.o: tests/test_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_mxfp4_rocm: tests/test_mxfp4_rocm.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/bench_mxfp4_rocm.o: tests/bench_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/bench_mxfp4_rocm: tests/bench_mxfp4_rocm.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-mxfp4-rocm: tests/test_mxfp4_rocm
	./tests/test_mxfp4_rocm

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_args.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qwen4exp_ple.o: tests/test_qwen4exp_ple.c ds4_qwen4exp_ple.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen4exp_ple: tests/test_qwen4exp_ple.o ds4_qwen4exp_ple.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# The C half checks the hash, the end-of-sequence rule and the IQ4_NL block
# layout without a file.  The Python half writes a synthetic table of a few
# million rows and compares ids, dequantized rows, gathered bytes and the hot
# set against an independent implementation.
test-qwen4exp-ple: tests/test_qwen4exp_ple tests/test_qwen4exp_ple_hash_ref
	./tests/test_qwen4exp_ple
	./tests/test_qwen4exp_ple_hash_ref
	python3 tests/test_qwen4exp_ple.py

# The row ids the hash names, against the reference, on real prompt ids.  No
# GGUF and no GPU: the constants are the production ones and the golden is the
# reference's answer.  tests/qwen4exp_ple_row_addressing.py is the other half,
# row index to byte offset, and needs the artifact and gguf-py, so it runs on a
# host that has them rather than here.
tests/test_qwen4exp_ple_hash_ref.o: tests/test_qwen4exp_ple_hash_ref.c ds4_qwen4exp_ple.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qwen4exp_ple_hash_ref: tests/test_qwen4exp_ple_hash_ref.o ds4_qwen4exp_ple.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Cross-implementation fixtures from the ds4-metal parity reference (MIT).
# Reference against reference: no GPU and no model, so it runs on every host.
# Built without -ffast-math because several cases assert bit exactness, which
# needs IEEE semantics on the host side.
tests/test_qwen4exp_parity.o: tests/test_qwen4exp_parity.c ds4_qwen4exp_hc_ref.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

$(QWEN4EXP_PARITY_TEST): tests/test_qwen4exp_parity.o
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4exp-parity
test-qwen4exp-parity: $(QWEN4EXP_PARITY_TEST)
	./$(QWEN4EXP_PARITY_TEST)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h ds4_qwen4exp.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp.inc ds4_qwen4exp_ple.h ds4_qwen4exp_graph.h ds4_qwen4exp_graph.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h ds4_qwen4exp_model_registration.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qwen4exp_loader.o: tests/test_qwen4exp_loader.c
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_qwen4exp_loader: tests/test_qwen4exp_loader.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_qwen4exp_ple.o ds4_qwen4exp_mtp_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Writes a reduced-layer synthetic qwen4exp GGUF set (sparse, a few hundred MiB
# on disk) and runs the loader against it plus the fault-injected variants.
test-qwen4exp: tests/test_qwen4exp_loader
	./tests/test_qwen4exp_loader.sh

ds4_qwen4exp_mtp.o: ds4_qwen4exp_mtp.c ds4_qwen4exp_mtp.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_qwen4exp_mtp.c

# The CPU build gets the cycle, the depth envelope and the rollback contract;
# the head is graph code and is compiled out, exactly as ds4.c compiles out its
# own graph half.
ds4_qwen4exp_mtp_hooks.o: ds4_qwen4exp_mtp_hooks.c ds4_qwen4exp_mtp.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ ds4_qwen4exp_mtp_hooks.c

ds4_qwen4exp_mtp_cpu.o: ds4_qwen4exp_mtp.c ds4_qwen4exp_mtp.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -I. -c -o $@ ds4_qwen4exp_mtp.c

# The test supplies its own ds4_gpu_tensor and GPU entry points, so it links
# the MTP object alone: no backend, no device, every host.
tests/test_qwen4exp_mtp.o: tests/test_qwen4exp_mtp.c ds4_qwen4exp_mtp.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4exp_mtp.c

$(QWEN4EXP_MTP_TEST): tests/test_qwen4exp_mtp.o ds4_qwen4exp_mtp.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4exp-mtp
test-qwen4exp-mtp: $(QWEN4EXP_MTP_TEST)
	./$(QWEN4EXP_MTP_TEST)

tests/test_qwen4exp_model_registration.o: tests/test_qwen4exp_model_registration.c ds4_qwen4exp_model_registration.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

$(QWEN4EXP_MODEL_REGISTRATION_TEST): tests/test_qwen4exp_model_registration.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4exp-model-registration test-cuda-spark-arch
test-qwen4exp-model-registration: $(QWEN4EXP_MODEL_REGISTRATION_TEST)
	./$(QWEN4EXP_MODEL_REGISTRATION_TEST)

test-cuda-spark-arch:
	sh tests/test_cuda_spark_arch.sh

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_shard_fd.o: tests/test_gpu_model_shard_fd.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_shard_fd: tests/test_gpu_model_shard_fd.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-gpu-model-shard-fd
test-gpu-model-shard-fd: tests/test_gpu_model_shard_fd
	./tests/test_gpu_model_shard_fd.sh

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h ds4_qwen4exp.h ds4_qwen4exp_moe_types.h ds4_qwen4exp_hc_types.h ds4_qwen4exp.inc ds4_qwen4exp_ple.h ds4_qwen4exp_graph.h ds4_qwen4exp_graph.inc ds4_qwen4exp_matmul.h ds4_qwen4exp_mtp.h ds4_qwen4exp_model_registration.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_deepseek4_vision_image $(SAMPLING_TEST) $(GPU_KERNEL_TESTS) \
	tests/test_qwen4exp_loader tests/test_qwen4exp_ple \
	$(QWEN4EXP_MODEL_REGISTRATION_TEST) $(QWEN4EXP_PARITY_TEST) \
	ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_qwen4exp_ple
	python3 tests/test_qwen4exp_ple.py
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_sampling
	./tests/test_deepseek4_vision_image
	./$(QWEN4EXP_MTP_TEST)
	./tests/test_qwen4exp_loader.sh
	./$(QWEN4EXP_MODEL_REGISTRATION_TEST)
	sh tests/test_cuda_spark_arch.sh
	sh tests/qwen4exp_cuda_moe_parity.sh
	$(GPU_KERNEL_TEST_RUN)

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f ds4_qwen4exp_mtp.o ds4_qwen4exp_mtp_hooks.o ds4_qwen4exp_mtp_cpu.o ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o speed-bench/metal_decode_schedule_bench speed-bench/metal_prefill_variant_bench speed-bench/*.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_metal tests/test_mxfp4_rocm tests/test_mxfp4_cuda tests/test_metal_session_batch tests/test_glm53_kda tests/test_glm53_kda_rocm tests/test_qwen4exp_gdn tests/test_qwen4exp_moe tests/test_qwen4exp_hc_norm tests/test_qwen4exp_qsa tests/test_qwen4exp_graph tests/test_glm53_vision_engine tests/test_glm53_vision_prompt tests/test_deepseek4_vision_image tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_model_shard_fd tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_qwen4exp_loader tests/test_qwen4exp_model_registration tests/test_qwen4exp_ple tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
