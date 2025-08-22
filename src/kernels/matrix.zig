//! High-performance matrix operations for neural network computations
//!
//! SIMD-optimized implementations for matrix multiplication, activation functions,
//! and other essential neural network operations.

const std = @import("std");
const math = std.math;
const simd = std.simd;

/// SIMD vector configuration
const VECTOR_SIZE = 8;
const VectorF32 = @Vector(VECTOR_SIZE, f32);

/// Matrix structure for efficient operations
pub const Matrix = struct {
    data: []f32,
    rows: u32,
    cols: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create new matrix
    pub fn init(allocator: std.mem.Allocator, rows: u32, cols: u32) !Self {
        const data = try allocator.alloc(f32, rows * cols);
        return Self{
            .data = data,
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
    }

    /// Create matrix from existing data
    pub fn fromSlice(allocator: std.mem.Allocator, data: []const f32, rows: u32, cols: u32) !Self {
        if (data.len != rows * cols) return error.InvalidDimensions;
        
        const matrix_data = try allocator.dupe(f32, data);
        return Self{
            .data = matrix_data,
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
    }

    /// Cleanup matrix
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }

    /// Get element at (row, col)
    pub fn get(self: *const Self, row: u32, col: u32) f32 {
        return self.data[row * self.cols + col];
    }

    /// Set element at (row, col)
    pub fn set(self: *Self, row: u32, col: u32, value: f32) void {
        self.data[row * self.cols + col] = value;
    }

    /// Fill matrix with value
    pub fn fill(self: *Self, value: f32) void {
        @memset(self.data, value);
    }

    /// Zero matrix
    pub fn zero(self: *Self) void {
        self.fill(0.0);
    }
};

/// Matrix multiplication: C = A * B
pub fn matmul(a: *const Matrix, b: *const Matrix, c: *Matrix) !void {
    if (a.cols != b.rows or c.rows != a.rows or c.cols != b.cols) {
        return error.IncompatibleDimensions;
    }

    c.zero();

    // Optimized matrix multiplication with SIMD
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            var sum: f32 = 0.0;
            var k: u32 = 0;

            // SIMD vectorized inner loop
            while (k + VECTOR_SIZE <= a.cols) : (k += VECTOR_SIZE) {
                const a_vec: VectorF32 = a.data[(i * a.cols + k)..(i * a.cols + k + VECTOR_SIZE)][0..VECTOR_SIZE].*;
                var b_vec: VectorF32 = undefined;
                
                // Gather B values
                for (0..VECTOR_SIZE) |idx| {
                    b_vec[idx] = b.data[(k + idx) * b.cols + j];
                }
                
                const prod_vec = a_vec * b_vec;
                sum += @reduce(.Add, prod_vec);
            }

            // Handle remaining elements
            while (k < a.cols) : (k += 1) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }

            c.data[i * c.cols + j] = sum;
        }
    }
}

/// Matrix-vector multiplication: y = A * x
pub fn matvec(a: *const Matrix, x: []const f32, y: []f32) !void {
    if (x.len != a.cols or y.len != a.rows) {
        return error.IncompatibleDimensions;
    }

    for (0..a.rows) |i| {
        var sum: f32 = 0.0;
        var j: u32 = 0;

        // SIMD vectorized computation
        while (j + VECTOR_SIZE <= a.cols) : (j += VECTOR_SIZE) {
            const a_vec: VectorF32 = a.data[(i * a.cols + j)..(i * a.cols + j + VECTOR_SIZE)][0..VECTOR_SIZE].*;
            const x_vec: VectorF32 = x[j..j + VECTOR_SIZE][0..VECTOR_SIZE].*;
            const prod_vec = a_vec * x_vec;
            sum += @reduce(.Add, prod_vec);
        }

        // Handle remaining elements
        while (j < a.cols) : (j += 1) {
            sum += a.data[i * a.cols + j] * x[j];
        }

        y[i] = sum;
    }
}

/// Element-wise activation functions
pub const ActivationFn = enum {
    ReLU,
    Tanh,
    Sigmoid,
    Swish,
    GELU,
};

/// Apply activation function to vector
pub fn applyActivation(input: []const f32, output: []f32, activation: ActivationFn) !void {
    if (input.len != output.len) return error.IncompatibleDimensions;

    var i: u32 = 0;
    
    // SIMD vectorized activation
    while (i + VECTOR_SIZE <= input.len) : (i += VECTOR_SIZE) {
        const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        var output_vec: VectorF32 = undefined;

        switch (activation) {
            .ReLU => {
                const zero_vec: VectorF32 = @splat(0.0);
                output_vec = @max(input_vec, zero_vec);
            },
            .Tanh => {
                for (0..VECTOR_SIZE) |j| {
                    output_vec[j] = math.tanh(input_vec[j]);
                }
            },
            .Sigmoid => {
                for (0..VECTOR_SIZE) |j| {
                    output_vec[j] = 1.0 / (1.0 + math.exp(-input_vec[j]));
                }
            },
            .Swish => {
                for (0..VECTOR_SIZE) |j| {
                    const sigmoid = 1.0 / (1.0 + math.exp(-input_vec[j]));
                    output_vec[j] = input_vec[j] * sigmoid;
                }
            },
            .GELU => {
                for (0..VECTOR_SIZE) |j| {
                    const x = input_vec[j];
                    const cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)));
                    output_vec[j] = x * cdf;
                }
            },
        }

        output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].* = output_vec;
    }

    // Handle remaining elements
    while (i < input.len) : (i += 1) {
        switch (activation) {
            .ReLU => output[i] = @max(input[i], 0.0),
            .Tanh => output[i] = math.tanh(input[i]),
            .Sigmoid => output[i] = 1.0 / (1.0 + math.exp(-input[i])),
            .Swish => {
                const sigmoid = 1.0 / (1.0 + math.exp(-input[i]));
                output[i] = input[i] * sigmoid;
            },
            .GELU => {
                const x = input[i];
                const cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)));
                output[i] = x * cdf;
            },
        }
    }
}

/// Softmax activation
pub fn softmax(input: []const f32, output: []f32) !void {
    if (input.len != output.len) return error.IncompatibleDimensions;

    // Find maximum for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |val| {
        max_val = @max(max_val, val);
    }

    // Compute exponentials and sum
    var sum: f32 = 0.0;
    var i: u32 = 0;

    // SIMD vectorized exponential
    while (i + VECTOR_SIZE <= input.len) : (i += VECTOR_SIZE) {
        const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const max_vec: VectorF32 = @splat(max_val);
        const shifted_vec = input_vec - max_vec;
        
        var exp_vec: VectorF32 = undefined;
        for (0..VECTOR_SIZE) |j| {
            exp_vec[j] = math.exp(shifted_vec[j]);
        }
        
        output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].* = exp_vec;
        sum += @reduce(.Add, exp_vec);
    }

    // Handle remaining elements
    while (i < input.len) : (i += 1) {
        const exp_val = math.exp(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    const inv_sum = 1.0 / sum;
    i = 0;
    while (i + VECTOR_SIZE <= output.len) : (i += VECTOR_SIZE) {
        const output_vec: VectorF32 = output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const inv_vec: VectorF32 = @splat(inv_sum);
        const normalized_vec = output_vec * inv_vec;
        output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].* = normalized_vec;
    }

    while (i < output.len) : (i += 1) {
        output[i] *= inv_sum;
    }
}

/// Layer normalization
pub fn layerNorm(input: []const f32, output: []f32, gamma: []const f32, beta: []const f32, eps: f32) !void {
    if (input.len != output.len or gamma.len != input.len or beta.len != input.len) {
        return error.IncompatibleDimensions;
    }

    // Compute mean
    var sum: f32 = 0.0;
    var i: u32 = 0;

    while (i + VECTOR_SIZE <= input.len) : (i += VECTOR_SIZE) {
        const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        sum += @reduce(.Add, input_vec);
    }

    while (i < input.len) : (i += 1) {
        sum += input[i];
    }

    const mean = sum / @as(f32, @floatFromInt(input.len));

    // Compute variance
    var var_sum: f32 = 0.0;
    i = 0;

    while (i + VECTOR_SIZE <= input.len) : (i += VECTOR_SIZE) {
        const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const mean_vec: VectorF32 = @splat(mean);
        const diff_vec = input_vec - mean_vec;
        const squared_vec = diff_vec * diff_vec;
        var_sum += @reduce(.Add, squared_vec);
    }

    while (i < input.len) : (i += 1) {
        const diff = input[i] - mean;
        var_sum += diff * diff;
    }

    const variance = var_sum / @as(f32, @floatFromInt(input.len));
    const std_dev = math.sqrt(variance + eps);
    const inv_std = 1.0 / std_dev;

    // Normalize and scale
    i = 0;
    while (i + VECTOR_SIZE <= input.len) : (i += VECTOR_SIZE) {
        const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const gamma_vec: VectorF32 = gamma[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const beta_vec: VectorF32 = beta[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
        const mean_vec: VectorF32 = @splat(mean);
        const inv_std_vec: VectorF32 = @splat(inv_std);
        
        const normalized_vec = (input_vec - mean_vec) * inv_std_vec;
        const output_vec = normalized_vec * gamma_vec + beta_vec;
        output[i..i + VECTOR_SIZE][0..VECTOR_SIZE].* = output_vec;
    }

    while (i < input.len) : (i += 1) {
        const normalized = (input[i] - mean) * inv_std;
        output[i] = normalized * gamma[i] + beta[i];
    }
}

/// Batch matrix multiplication for attention
pub fn batchMatmul(a: []const f32, b: []const f32, c: []f32, 
                   batch_size: u32, m: u32, k: u32, n: u32) !void {
    if (a.len != batch_size * m * k or 
        b.len != batch_size * k * n or 
        c.len != batch_size * m * n) {
        return error.IncompatibleDimensions;
    }

    for (0..batch_size) |batch| {
        const a_offset = batch * m * k;
        const b_offset = batch * k * n;
        const c_offset = batch * m * n;

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                var l: u32 = 0;

                // SIMD vectorized inner loop
                while (l + VECTOR_SIZE <= k) : (l += VECTOR_SIZE) {
                    var a_vec: VectorF32 = undefined;
                    var b_vec: VectorF32 = undefined;

                    for (0..VECTOR_SIZE) |idx| {
                        a_vec[idx] = a[a_offset + i * k + l + idx];
                        b_vec[idx] = b[b_offset + (l + idx) * n + j];
                    }

                    const prod_vec = a_vec * b_vec;
                    sum += @reduce(.Add, prod_vec);
                }

                // Handle remaining elements
                while (l < k) : (l += 1) {
                    sum += a[a_offset + i * k + l] * b[b_offset + l * n + j];
                }

                c[c_offset + i * n + j] = sum;
            }
        }
    }
}

// FFI exports for Rust integration
export fn matrix_create(rows: u32, cols: u32) ?*Matrix {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const matrix = allocator.create(Matrix) catch return null;
    matrix.* = Matrix.init(allocator, rows, cols) catch return null;
    
    return matrix;
}

export fn matrix_destroy(matrix: ?*Matrix) void {
    if (matrix) |m| {
        m.deinit();
        m.allocator.destroy(m);
    }
}

export fn matrix_matmul(a: ?*const Matrix, b: ?*const Matrix, c: ?*Matrix) bool {
    if (a == null or b == null or c == null) return false;
    matmul(a.?, b.?, c.?) catch return false;
    return true;
}

export fn matrix_matvec(a: ?*const Matrix, x: [*]const f32, x_len: u32, y: [*]f32, y_len: u32) bool {
    if (a == null) return false;
    const x_slice = x[0..x_len];
    const y_slice = y[0..y_len];
    matvec(a.?, x_slice, y_slice) catch return false;
    return true;
}

export fn matrix_activation(input: [*]const f32, output: [*]f32, len: u32, activation: u32) bool {
    const input_slice = input[0..len];
    const output_slice = output[0..len];
    
    const act_fn: ActivationFn = switch (activation) {
        0 => .ReLU,
        1 => .Tanh,
        2 => .Sigmoid,
        3 => .Swish,
        4 => .GELU,
        else => return false,
    };
    
    applyActivation(input_slice, output_slice, act_fn) catch return false;
    return true;
}

export fn matrix_softmax(input: [*]const f32, output: [*]f32, len: u32) bool {
    const input_slice = input[0..len];
    const output_slice = output[0..len];
    softmax(input_slice, output_slice) catch return false;
    return true;
}

export fn matrix_layer_norm(input: [*]const f32, output: [*]f32, gamma: [*]const f32, 
                           beta: [*]const f32, len: u32, eps: f32) bool {
    const input_slice = input[0..len];
    const output_slice = output[0..len];
    const gamma_slice = gamma[0..len];
    const beta_slice = beta[0..len];
    
    layerNorm(input_slice, output_slice, gamma_slice, beta_slice, eps) catch return false;
    return true;
}

// Unit tests
test "Matrix operations" {
    const testing = std.testing;
    
    var a = try Matrix.init(testing.allocator, 2, 3);
    defer a.deinit();
    var b = try Matrix.init(testing.allocator, 3, 2);
    defer b.deinit();
    var c = try Matrix.init(testing.allocator, 2, 2);
    defer c.deinit();
    
    // Initialize test matrices
    a.data[0] = 1.0; a.data[1] = 2.0; a.data[2] = 3.0;
    a.data[3] = 4.0; a.data[4] = 5.0; a.data[5] = 6.0;
    
    b.data[0] = 1.0; b.data[1] = 2.0;
    b.data[2] = 3.0; b.data[3] = 4.0;
    b.data[4] = 5.0; b.data[5] = 6.0;
    
    try matmul(&a, &b, &c);
    
    // Verify result
    try testing.expect(c.data[0] == 22.0); // 1*1 + 2*3 + 3*5
    try testing.expect(c.data[1] == 28.0); // 1*2 + 2*4 + 3*6
}

test "Activation functions" {
    const testing = std.testing;
    
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var output = [_]f32{0.0} ** 4;
    
    // Test ReLU
    try applyActivation(&input, &output, .ReLU);
    try testing.expect(output[0] == 0.0);
    try testing.expect(output[1] == 0.0);
    try testing.expect(output[2] == 1.0);
    try testing.expect(output[3] == 2.0);
}

test "Softmax" {
    const testing = std.testing;
    
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{0.0} ** 3;
    
    try softmax(&input, &output);
    
    // Check that outputs sum to 1
    const sum = output[0] + output[1] + output[2];
    try testing.expect(@abs(sum - 1.0) < 1e-6);
    
    // Check that outputs are positive
    try testing.expect(output[0] > 0.0);
    try testing.expect(output[1] > 0.0);
    try testing.expect(output[2] > 0.0);
}