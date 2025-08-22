//! Comprehensive test suite for all Zig kernels
//!
//! Tests SNN, matrix operations, sparse computations, and quantization kernels.

const std = @import("std");
const testing = std.testing;
const math = std.math;

const snn = @import("snn.zig");
const matrix = @import("matrix.zig");
const sparse = @import("sparse.zig");
const quantize = @import("quantize.zig");

// =============================================================================
// SNN Kernel Tests
// =============================================================================

test "SNN kernel creation and basic operations" {
    const config = snn.SnnConfig{
        .input_size = 10,
        .output_size = 5,
        .threshold = 0.5,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.2,
    };
    
    var kernel = try snn.SnnKernel.init(testing.allocator, config);
    defer kernel.deinit();
    
    // Test configuration
    try testing.expect(kernel.config.input_size == 10);
    try testing.expect(kernel.config.output_size == 5);
    try testing.expect(kernel.current_time == 0);
    
    // Test forward pass
    const input = [_]f32{ 0.6, 0.3, 0.8, 0.1, 0.7, 0.4, 0.9, 0.2, 0.5, 0.0 };
    var output = [_]f32{0.0} ** 5;
    
    try kernel.forward(&input, &output);
    
    // Verify sparsity constraint
    var spike_count: u32 = 0;
    for (output) |val| {
        if (val > 0.0) spike_count += 1;
    }
    
    const max_spikes = @as(u32, @intFromFloat(@as(f32, @floatFromInt(config.output_size)) * config.sparse_rate));
    try testing.expect(spike_count <= max_spikes + 1);
}

test "SNN reset functionality" {
    const config = snn.SnnConfig{
        .input_size = 5,
        .output_size = 3,
        .threshold = 0.3,
        .decay_rate = 0.8,
        .refractory_period = 1,
        .sparse_rate = 0.5,
    };
    
    var kernel = try snn.SnnKernel.init(testing.allocator, config);
    defer kernel.deinit();
    
    // Process input
    const input = [_]f32{ 0.5, 0.4, 0.6, 0.2, 0.7 };
    var output = [_]f32{0.0} ** 3;
    try kernel.forward(&input, &output);
    
    // Reset and verify
    kernel.reset();
    const stats = kernel.getStats();
    try testing.expect(stats.current_time == 0);
    try testing.expect(stats.active_neurons == 0);
}

// =============================================================================
// Matrix Operation Tests
// =============================================================================

test "Matrix creation and operations" {
    var mat = try matrix.Matrix.init(testing.allocator, 3, 2);
    defer mat.deinit();
    
    try testing.expect(mat.rows == 3);
    try testing.expect(mat.cols == 2);
    
    mat.set(1, 1, 5.5);
    try testing.expectApproxEqAbs(mat.get(1, 1), 5.5, 1e-6);
    
    mat.fill(2.0);
    for (mat.data) |val| {
        try testing.expectApproxEqAbs(val, 2.0, 1e-6);
    }
}

test "Matrix multiplication" {
    var a = try matrix.Matrix.init(testing.allocator, 2, 3);
    defer a.deinit();
    var b = try matrix.Matrix.init(testing.allocator, 3, 2);
    defer b.deinit();
    var c = try matrix.Matrix.init(testing.allocator, 2, 2);
    defer c.deinit();
    
    // A = [[1,2,3], [4,5,6]]
    a.set(0, 0, 1.0); a.set(0, 1, 2.0); a.set(0, 2, 3.0);
    a.set(1, 0, 4.0); a.set(1, 1, 5.0); a.set(1, 2, 6.0);
    
    // B = [[7,8], [9,10], [11,12]]
    b.set(0, 0, 7.0); b.set(0, 1, 8.0);
    b.set(1, 0, 9.0); b.set(1, 1, 10.0);
    b.set(2, 0, 11.0); b.set(2, 1, 12.0);
    
    try matrix.matmul(&a, &b, &c);
    
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    try testing.expectApproxEqAbs(c.get(0, 0), 58.0, 1e-6);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 154
    try testing.expectApproxEqAbs(c.get(1, 1), 154.0, 1e-6);
}

test "Activation functions" {
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var output = [_]f32{0.0} ** 4;
    
    // Test ReLU
    try matrix.applyActivation(&input, &output, .ReLU);
    try testing.expectApproxEqAbs(output[0], 0.0, 1e-6);
    try testing.expectApproxEqAbs(output[1], 0.0, 1e-6);
    try testing.expectApproxEqAbs(output[2], 1.0, 1e-6);
    try testing.expectApproxEqAbs(output[3], 2.0, 1e-6);
    
    // Test Sigmoid
    try matrix.applyActivation(&input, &output, .Sigmoid);
    try testing.expectApproxEqAbs(output[1], 0.5, 1e-6); // sigmoid(0) = 0.5
    try testing.expect(output[0] < output[1]);
    try testing.expect(output[1] < output[2]);
}

// =============================================================================
// Sparse Computation Tests
// =============================================================================

test "Sparse matrix operations" {
    var sparse_mat = try sparse.SparseMatrix.init(testing.allocator, 3, 3, 4);
    defer sparse_mat.deinit();
    
    try testing.expect(sparse_mat.rows == 3);
    try testing.expect(sparse_mat.cols == 3);
    try testing.expect(sparse_mat.nnz == 4);
    
    try sparse_mat.setValue(0, 1, 2.5);
    try sparse_mat.setValue(1, 2, 3.7);
    
    try testing.expectApproxEqAbs(sparse_mat.getValue(0, 1), 2.5, 1e-6);
    try testing.expectApproxEqAbs(sparse_mat.getValue(1, 2), 3.7, 1e-6);
    try testing.expectApproxEqAbs(sparse_mat.getValue(2, 2), 0.0, 1e-6);
}

test "Sparse activation enforcement" {
    var input = [_]f32{ 0.9, 0.2, 0.8, 0.1, 0.7 };
    var output = [_]f32{0.0} ** 5;
    const target_sparsity: f32 = 0.4; // Keep 40%
    
    try sparse.enforceSparseActivation(&input, &output, target_sparsity);
    
    var active_count: u32 = 0;
    for (output) |val| {
        if (val > 0.0) active_count += 1;
    }
    
    const expected_active = @as(u32, @intFromFloat(5.0 * target_sparsity));
    try testing.expect(active_count <= expected_active + 1);
}

// =============================================================================
// Quantization Tests
// =============================================================================

test "4-bit quantization" {
    const input = [_]f32{ -2.0, 0.0, 1.0, 2.0 };
    var quantized = [_]u8{0} ** 4;
    var output = [_]f32{0.0} ** 4;
    
    var scale: f32 = undefined;
    var zero_point: u8 = undefined;
    
    try quantize.quantize4bit(&input, &quantized, &scale, &zero_point);
    try quantize.dequantize4bit(&quantized, &output, scale, zero_point);
    
    // Check ordering preservation
    try testing.expect(output[0] <= output[1]);
    try testing.expect(output[1] <= output[2]);
    try testing.expect(output[2] <= output[3]);
}

test "8-bit quantization accuracy" {
    const input = [_]f32{ -5.0, 0.0, 5.0, 2.5 };
    var quantized = [_]u8{0} ** 4;
    var output = [_]f32{0.0} ** 4;
    
    var scale: f32 = undefined;
    var zero_point: u8 = undefined;
    
    try quantize.quantize8bit(&input, &quantized, &scale, &zero_point);
    try quantize.dequantize8bit(&quantized, &output, scale, zero_point);
    
    // 8-bit should be reasonably accurate
    for (input, output) |orig, reconstructed| {
        const error = @fabs(orig - reconstructed);
        const relative_error = if (orig != 0.0) error / @fabs(orig) else error;
        try testing.expect(relative_error < 0.1); // Less than 10% error
    }
}

test "Block-wise quantization" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0 };
    var quantized = [_]u8{0} ** 8;
    var output = [_]f32{0.0} ** 8;
    
    var scales = [_]f32{0.0} ** 2;
    var zero_points = [_]u8{0} ** 2;
    const block_size: u32 = 4;
    
    try quantize.quantizeBlocks(&input, &quantized, &scales, &zero_points, block_size);
    try quantize.dequantizeBlocks(&quantized, &output, &scales, &zero_points, block_size);
    
    // Verify each block maintains reasonable accuracy
    for (0..2) |block_idx| {
        const start = block_idx * 4;
        for (start..start + 4) |i| {
            const error = @fabs(input[i] - output[i]);
            const scale = scales[block_idx];
            const max_error = scale * 255.0 / 2.0;
            try testing.expect(error <= max_error);
        }
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

test "SNN performance" {
    const config = snn.SnnConfig{
        .input_size = 256,
        .output_size = 128,
        .threshold = 0.5,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.15,
    };
    
    var kernel = try snn.SnnKernel.init(testing.allocator, config);
    defer kernel.deinit();
    
    var input: [256]f32 = undefined;
    var output = [_]f32{0.0} ** 128;
    
    var prng = std.rand.DefaultPrng.init(12345);
    const random = prng.random();
    for (&input) |*val| {
        val.* = random.float(f32);
    }
    
    const start_time = std.time.nanoTimestamp();
    for (0..50) |_| {
        try kernel.forward(&input, &output);
    }
    const end_time = std.time.nanoTimestamp();
    
    const avg_time = @as(u64, @intCast(end_time - start_time)) / 50;
    try testing.expect(avg_time < 1_000_000); // Less than 1ms
}

// =============================================================================
// Integration Tests
// =============================================================================

test "Full pipeline integration" {
    // Test quantization -> SNN -> activation pipeline
    const size: u32 = 32;
    
    // Step 1: Quantize input
    var input_f32: [size]f32 = undefined;
    var prng = std.rand.DefaultPrng.init(99999);
    const random = prng.random();
    for (&input_f32) |*val| {
        val.* = (random.float(f32) - 0.5) * 2.0;
    }
    
    var quantized = [_]u8{0} ** size;
    var dequantized = [_]f32{0.0} ** size;
    var scale: f32 = undefined;
    var zero_point: u8 = undefined;
    
    try quantize.quantize8bit(&input_f32, &quantized, &scale, &zero_point);
    try quantize.dequantize8bit(&quantized, &dequantized, scale, zero_point);
    
    // Step 2: Process through SNN
    const snn_config = snn.SnnConfig{
        .input_size = size,
        .output_size = size / 2,
        .threshold = 0.3,
        .decay_rate = 0.9,
        .refractory_period = 1,
        .sparse_rate = 0.2,
    };
    
    var snn_kernel = try snn.SnnKernel.init(testing.allocator, snn_config);
    defer snn_kernel.deinit();
    
    var snn_output = [_]f32{0.0} ** (size / 2);
    try snn_kernel.forward(&dequantized, &snn_output);
    
    // Step 3: Apply activation
    var final_output = [_]f32{0.0} ** (size / 2);
    try matrix.applyActivation(&snn_output, &final_output, .ReLU);
    
    // Verify pipeline integrity
    var non_zero_count: u32 = 0;
    for (final_output) |val| {
        try testing.expect(val >= 0.0); // ReLU ensures non-negative
        if (val > 0.0) non_zero_count += 1;
    }
    
    // Should maintain sparsity
    const sparsity_ratio = @as(f32, @floatFromInt(non_zero_count)) / @as(f32, @floatFromInt(final_output.len));
    try testing.expect(sparsity_ratio <= 0.5); // Should be reasonably sparse
}