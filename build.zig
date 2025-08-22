const std = @import("std");

/// Build configuration for ultra-fast AI model Zig kernels
/// Provides FFI support for Rust integration and optimized neural network operations
pub fn build(b: *std.Build) void {
    // Target configuration - optimize for CPU performance
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Kernel library - shared object for FFI
    const kernels_lib = b.addLibrary(.{
        .name = "ultra_fast_kernels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
    });
    
    // Optimization flags for neural network performance
    // kernels_lib.bundle_compiler_rt = true; // Not available in 0.15.1
    
    // Export symbols for FFI
    // kernels_lib.export_symbol_names = &.{  // Not available in 0.15.1
    //     "snn_activate",
    //     "snn_spike_train",
    //     "matrix_multiply_sparse", 
    //     "matrix_quantize_4bit",
    //     "sparse_mask_apply",
    //     "energy_monitor_start",
    //     "energy_monitor_end",
    // };

    // Static library for linking with Rust
    const kernels_static = b.addLibrary(.{
        .name = "ultra_fast_kernels_static",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });

    // Install artifacts
    b.installArtifact(kernels_lib);
    b.installArtifact(kernels_static);

    // Generate C header for FFI
    const header_step = b.addWriteFiles();
    _ = header_step.add("ultra_fast_kernels.h", generateCHeader());
    
    // Test suite for all kernels
    const kernel_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Benchmark suite
    const kernel_benchmarks = b.addExecutable(.{
        .name = "kernel_benchmarks", 
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/kernels/bench.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Build steps
    const lib_step = b.step("lib", "Build kernel libraries");
    lib_step.dependOn(&kernels_lib.step);
    lib_step.dependOn(&kernels_static.step);

    const test_step = b.step("test", "Run kernel tests");
    const test_run = b.addRunArtifact(kernel_tests);
    test_step.dependOn(&test_run.step);

    const bench_step = b.step("bench", "Run kernel benchmarks");
    const bench_run = b.addRunArtifact(kernel_benchmarks);
    bench_step.dependOn(&bench_run.step);

    // Default build includes everything
    b.default_step.dependOn(lib_step);
    b.default_step.dependOn(test_step);
}

/// Generate C header file for FFI integration
fn generateCHeader() []const u8 {
    return 
        \\#ifndef ULTRA_FAST_KERNELS_H
        \\#define ULTRA_FAST_KERNELS_H
        \\
        \\#include <stddef.h>
        \\#include <stdint.h>
        \\
        \\#ifdef __cplusplus
        \\extern "C" {
        \\#endif
        \\
        \\// SNN Operations
        \\typedef struct {
        \\    float threshold;
        \\    float decay;
        \\    uint32_t refractory_period;
        \\} SNNParams;
        \\
        \\// Activate SNN layer with sparse output
        \\// Returns number of spikes generated
        \\uint32_t snn_activate(const float* input, size_t input_len, 
        \\                     float* output, const SNNParams* params);
        \\
        \\// Generate spike train from input
        \\void snn_spike_train(const float* input, size_t input_len, 
        \\                    uint8_t* spikes, size_t time_steps);
        \\
        \\// Matrix Operations  
        \\void matrix_multiply_sparse(const float* a, const float* b, float* c,
        \\                           size_t m, size_t n, size_t k, 
        \\                           const uint8_t* mask);
        \\
        \\// 4-bit quantization for memory efficiency
        \\void matrix_quantize_4bit(const float* input, uint8_t* output, 
        \\                         float* scale, size_t len);
        \\
        \\// Sparse Operations
        \\void sparse_mask_apply(const float* input, float* output, 
        \\                      const uint8_t* mask, size_t len);
        \\
        \\// Energy Monitoring
        \\uint64_t energy_monitor_start(void);
        \\double energy_monitor_end(uint64_t start_time);
        \\
        \\#ifdef __cplusplus
        \\}
        \\#endif
        \\
        \\#endif // ULTRA_FAST_KERNELS_H
    ;
}