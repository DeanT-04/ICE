//! Main library entry point for ultra-fast AI kernels
//! Exports all kernel functions for FFI integration with Rust

const std = @import("std");
const snn = @import("snn.zig");
const matrix = @import("matrix.zig");
const sparse = @import("sparse.zig");
const quantize = @import("quantize.zig");

/// Global allocator for kernel operations
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Energy monitoring state
var energy_start_time: u64 = 0;

/// SNN activation function for FFI
pub export fn snn_activate(input: [*]const f32, input_len: usize, 
                      output: [*]f32, params: [*]const snn.SnnConfig) callconv(.c) u32 {
    const input_slice = input[0..input_len];
    const config = params[0];
    
    // Create temporary SNN kernel
    var kernel = snn.SnnKernel.init(allocator, config) catch return 0;
    defer kernel.deinit();
    
    const output_slice = output[0..config.output_size];
    
    // Process through SNN
    kernel.forward(input_slice, output_slice) catch return 0;
    
    // Count spikes generated
    var spike_count: u32 = 0;
    for (output_slice) |val| {
        if (val > 0.0) spike_count += 1;
    }
    
    return spike_count;
}

/// Generate spike train from input
export fn snn_spike_train(input: [*]const f32, input_len: usize, 
                         spikes: [*]u8, time_steps: usize) callconv(.c) void {
    const input_slice = input[0..input_len];
    const spike_slice = spikes[0..input_len * time_steps];
    
    for (0..time_steps) |t| {
        for (0..input_len) |i| {
            const spike_prob = 1.0 / (1.0 + std.math.exp(-input_slice[i])); // Sigmoid
            
            // Stochastic spike generation
            var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp() + @as(i64, @intCast(t)) + @as(i64, @intCast(i))));
            const random_val = prng.random().float(f32);
            
            spike_slice[t * input_len + i] = if (random_val < spike_prob) 1 else 0;
        }
    }
}

/// Sparse matrix multiplication
export fn matrix_multiply_sparse(a: [*]const f32, b: [*]const f32, c: [*]f32,
                                m: usize, n: usize, k: usize, 
                                mask: [*]const u8) callconv(.c) void {
    // Zero output matrix
    @memset(c[0..m * n], 0.0);
    
    // Sparse multiplication
    for (0..m) |i| {
        for (0..k) |j| {
            const a_val = a[i * k + j];
            if (a_val == 0.0) continue;
            
            for (0..n) |l| {
                const mask_idx = i * n + l;
                if (mask[mask_idx] == 0) continue; // Skip if masked out
                
                c[i * n + l] += a_val * b[j * n + l];
            }
        }
    }
}

/// 4-bit quantization
pub export fn matrix_quantize_4bit(input: [*]const f32, output: [*]u8, 
                              scale: [*]f32, len: usize) callconv(.c) void {
    const input_slice = input[0..len];
    const output_slice = output[0..len / 2]; // 4-bit packing
    
    // Find min/max for scale calculation
    var min_val: f32 = std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);
    
    for (input_slice) |val| {
        min_val = @min(min_val, val);
        max_val = @max(max_val, val);
    }
    
    const range = max_val - min_val;
    const quantize_scale = range / 15.0; // 4-bit = 16 levels (0-15)
    scale[0] = quantize_scale;
    
    // Quantize values
    for (0..len / 2) |i| {
        const val1 = input_slice[i * 2];
        const val2 = input_slice[i * 2 + 1];
        
        const q1 = @as(u8, @intFromFloat(@max(0, @min(15, (val1 - min_val) / quantize_scale))));
        const q2 = @as(u8, @intFromFloat(@max(0, @min(15, (val2 - min_val) / quantize_scale))));
        
        // Pack two 4-bit values into one byte
        output_slice[i] = (q1 << 4) | (q2 & 0x0F);
    }
}

/// Apply sparse mask
pub export fn sparse_mask_apply(input: [*]const f32, output: [*]f32, 
                           mask: [*]const u8, len: usize) callconv(.c) void {
    const VECTOR_SIZE = 8;
    var i: usize = 0;
    
    // SIMD vectorized masking
    while (i + VECTOR_SIZE <= len) : (i += VECTOR_SIZE) {
        const input_vec: @Vector(VECTOR_SIZE, f32) = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        var mask_vec: @Vector(VECTOR_SIZE, f32) = undefined;
        
        // Convert mask to float vector
        for (0..VECTOR_SIZE) |j| {
            mask_vec[j] = if (mask[i + j] != 0) 1.0 else 0.0;
        }
        
        const result_vec = input_vec * mask_vec;
        output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].* = result_vec;
    }
    
    // Handle remaining elements
    while (i < len) : (i += 1) {
        output[i] = if (mask[i] != 0) input[i] else 0.0;
    }
}

/// Start energy monitoring
pub export fn energy_monitor_start() callconv(.c) u64 {
    energy_start_time = @intCast(std.time.nanoTimestamp());
    return energy_start_time;
}

/// End energy monitoring and return energy estimate
pub export fn energy_monitor_end(start_time: u64) callconv(.c) f64 {
    const end_time = std.time.nanoTimestamp();
    const duration_ns = end_time - start_time;
    const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
    
    // Simple energy estimation based on execution time
    // Assumes baseline power consumption of 50W for CPU
    const baseline_power_w = 50.0;
    const energy_joules = (baseline_power_w * duration_ms) / 1000.0;
    
    return energy_joules;
}

/// Library initialization
export fn kernel_lib_init() callconv(.c) bool {
    return true;
}

/// Library cleanup
export fn kernel_lib_deinit() callconv(.c) void {
    _ = gpa.deinit();
}