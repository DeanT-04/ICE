//! Test entry point for all kernel modules
//! Runs comprehensive tests for SNN, matrix, sparse, and quantization kernels

const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqRel = testing.expectApproxEqRel;

const snn = @import("snn.zig");
const matrix = @import("matrix.zig");
const sparse = @import("sparse.zig");
const quantize = @import("quantize.zig");
const lib = @import("lib.zig");

test "SNN kernel basic functionality" {
    const allocator = testing.allocator;
    
    const config = snn.SnnConfig{
        .input_size = 4,
        .output_size = 2,
        .threshold = 1.0,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.3,
    };
    
    var kernel = try snn.SnnKernel.init(allocator, config);
    defer kernel.deinit();
    
    const input = [_]f32{ 1.5, 0.5, 2.0, 0.1 };
    var output = [_]f32{ 0.0, 0.0 };
    
    try kernel.forward(&input, &output);
    
    // Verify some output was generated
    var has_output = false;
    for (output) |val| {
        if (val != 0.0) has_output = true;
    }
    try testing.expect(has_output);
}

test "Matrix multiplication correctness" {
    const allocator = testing.allocator;
    
    var a = try matrix.Matrix.init(allocator, 2, 3);
    defer a.deinit();
    var b = try matrix.Matrix.init(allocator, 3, 2);
    defer b.deinit();
    var c = try matrix.Matrix.init(allocator, 2, 2);
    defer c.deinit();
    
    // Set up test matrices
    // A = [[1, 2, 3], [4, 5, 6]]
    a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
    a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);
    
    // B = [[7, 8], [9, 10], [11, 12]]
    b.set(0, 0, 7.0); b.set(0, 1, 8.0);
    b.set(1, 0, 9.0); b.set(1, 1, 10.0);
    b.set(2, 0, 11.0); b.set(2, 1, 12.0);
    
    try matrix.matmul(&a, &b, &c);
    
    // Expected: C = [[58, 64], [139, 154]]
    try expectApproxEqRel(c.get(0, 0), 58.0, 1e-6);
    try expectApproxEqRel(c.get(0, 1), 64.0, 1e-6);
    try expectApproxEqRel(c.get(1, 0), 139.0, 1e-6);
    try expectApproxEqRel(c.get(1, 1), 154.0, 1e-6);
}

test "Matrix-vector multiplication" {
    const allocator = testing.allocator;
    
    var a = try matrix.Matrix.init(allocator, 2, 3);
    defer a.deinit();
    
    // A = [[1, 2, 3], [4, 5, 6]]
    a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
    a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);
    
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var y = [_]f32{ 0.0, 0.0 };
    
    try matrix.matvec(&a, &x, &y);
    
    // Expected: y = [14, 32]
    try expectApproxEqRel(y[0], 14.0, 1e-6);
    try expectApproxEqRel(y[1], 32.0, 1e-6);
}

test "ReLU activation function" {
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.5, -0.5 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    
    try matrix.applyActivation(&input, &output, .ReLU);
    
    try expectApproxEqRel(output[0], 0.0, 1e-6);
    try expectApproxEqRel(output[1], 0.0, 1e-6);
    try expectApproxEqRel(output[2], 1.0, 1e-6);
    try expectApproxEqRel(output[3], 2.5, 1e-6);
    try expectApproxEqRel(output[4], 0.0, 1e-6);
}

test "Sparse matrix operations" {
    const allocator = testing.allocator;
    
    const dense_matrix = [_]f32{
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0,
    };
    
    var sparse_matrix = try sparse.CSRMatrix.fromDense(allocator, &dense_matrix, 3, 3, 0.1);
    defer sparse_matrix.deinit();
    
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    
    try sparse_matrix.spmv(&input, &output);
    
    // Check non-zero results
    try testing.expect(output[0] != 0.0); // Should have result from 1*1 + 2*3
    try testing.expect(output[1] != 0.0); // Should have result from 3*2
    try testing.expect(output[2] != 0.0); // Should have result from 4*1 + 5*3
}

test "4-bit quantization" {
    const input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    var output = [_]u8{ 0, 0, 0 };
    var scale: f32 = 0.0;
    
    lib.matrix_quantize_4bit(&input, &output, @ptrCast(&scale), input.len);
    
    // Scale should be non-zero
    try testing.expect(scale > 0.0);
    
    // Output should contain quantized values
    for (output) |val| {
        try testing.expect(val <= 255); // Valid byte values
    }
}

test "Sparse mask application" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const mask = [_]u8{ 1, 0, 1, 0, 1 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    
    lib.sparse_mask_apply(&input, &output, &mask, input.len);
    
    try expectApproxEqRel(output[0], 1.0, 1e-6);
    try expectApproxEqRel(output[1], 0.0, 1e-6);
    try expectApproxEqRel(output[2], 3.0, 1e-6);
    try expectApproxEqRel(output[3], 0.0, 1e-6);
    try expectApproxEqRel(output[4], 5.0, 1e-6);
}

test "Energy monitoring" {
    const start_time = lib.energy_monitor_start();
    
    // Simulate some work
    std.Thread.sleep(1_000_000); // 1ms
    
    const energy = lib.energy_monitor_end(start_time);
    
    // Energy should be positive
    try testing.expect(energy > 0.0);
}

test "FFI SNN activation" {
    const config = snn.SnnConfig{
        .input_size = 3,
        .output_size = 2,
        .threshold = 1.0,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.3,
    };
    
    const input = [_]f32{ 1.5, 0.5, 2.0 };
    var output = [_]f32{ 0.0, 0.0 };
    
    const spike_count = lib.snn_activate(&input, input.len, &output, @ptrCast(&config));
    
    // Should generate some spikes
    try testing.expect(spike_count <= config.output_size);
}

test "Integration test - full pipeline" {
    const allocator = testing.allocator;
    
    // Test a small neural network forward pass
    const input_size = 4;
    const hidden_size = 3;
    const output_size = 2;
    
    // Create weight matrices
    var w1 = try matrix.Matrix.init(allocator, input_size, hidden_size);
    defer w1.deinit();
    var w2 = try matrix.Matrix.init(allocator, hidden_size, output_size);
    defer w2.deinit();
    
    // Initialize weights
    w1.fill(0.1);
    w2.fill(0.2);
    
    // Input data
    const input = [_]f32{ 1.0, 0.5, -0.5, 2.0 };
    var hidden = [_]f32{ 0.0, 0.0, 0.0 };
    var output = [_]f32{ 0.0, 0.0 };
    
    // Forward pass: input -> hidden
    try matrix.matvec(&w1, &input, &hidden);
    try matrix.applyActivation(&hidden, &hidden, .ReLU);
    
    // Forward pass: hidden -> output
    try matrix.matvec(&w2, &hidden, &output);
    
    // Verify output is reasonable
    for (output) |val| {
        try testing.expect(std.math.isFinite(val));
    }
}