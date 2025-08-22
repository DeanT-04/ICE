//! Build configuration for Zig kernel tests
//!
//! Usage: zig build test

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add kernel test executable
    const kernel_tests = b.addTest(.{
        .root_source_file = .{ .path = "kernel_tests.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add individual kernel module tests
    const snn_tests = b.addTest(.{
        .root_source_file = .{ .path = "snn.zig" },
        .target = target,
        .optimize = optimize,
    });

    const matrix_tests = b.addTest(.{
        .root_source_file = .{ .path = "matrix.zig" },
        .target = target,
        .optimize = optimize,
    });

    const sparse_tests = b.addTest(.{
        .root_source_file = .{ .path = "sparse.zig" },
        .target = target,
        .optimize = optimize,
    });

    const quantize_tests = b.addTest(.{
        .root_source_file = .{ .path = "quantize.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Create test steps
    const run_kernel_tests = b.addRunArtifact(kernel_tests);
    const run_snn_tests = b.addRunArtifact(snn_tests);
    const run_matrix_tests = b.addRunArtifact(matrix_tests);
    const run_sparse_tests = b.addRunArtifact(sparse_tests);
    const run_quantize_tests = b.addRunArtifact(quantize_tests);

    // Main test command
    const test_step = b.step("test", "Run all kernel tests");
    test_step.dependOn(&run_kernel_tests.step);
    test_step.dependOn(&run_snn_tests.step);
    test_step.dependOn(&run_matrix_tests.step);
    test_step.dependOn(&run_sparse_tests.step);
    test_step.dependOn(&run_quantize_tests.step);

    // Individual test commands
    const test_kernel_step = b.step("test-kernel", "Run comprehensive kernel tests");
    test_kernel_step.dependOn(&run_kernel_tests.step);

    const test_snn_step = b.step("test-snn", "Run SNN kernel tests");
    test_snn_step.dependOn(&run_snn_tests.step);

    const test_matrix_step = b.step("test-matrix", "Run matrix operation tests");
    test_matrix_step.dependOn(&run_matrix_tests.step);

    const test_sparse_step = b.step("test-sparse", "Run sparse computation tests");
    test_sparse_step.dependOn(&run_sparse_tests.step);

    const test_quantize_step = b.step("test-quantize", "Run quantization tests");
    test_quantize_step.dependOn(&run_quantize_tests.step);

    // Performance test command (runs tests with release optimization)
    const perf_tests = b.addTest(.{
        .root_source_file = .{ .path = "kernel_tests.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });

    const run_perf_tests = b.addRunArtifact(perf_tests);
    const perf_step = b.step("test-perf", "Run performance benchmarks");
    perf_step.dependOn(&run_perf_tests.step);
}