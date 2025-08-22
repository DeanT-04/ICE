//! 4-bit quantization kernels for memory-efficient neural network operations
//!
//! Implements INT4 quantization with optimized packing, unpacking, and
//! quantized arithmetic operations for reduced memory footprint.

const std = @import("std");
const math = std.math;

/// 4-bit quantization parameters
pub const QuantParams = extern struct {
    scale: f32,
    zero_point: i8,
    min_val: f32,
    max_val: f32,
};

/// Packed 4-bit integer array (2 values per byte)
pub const PackedInt4 = struct {
    data: []u8,
    length: u32,    // Number of 4-bit values
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create packed int4 array
    pub fn init(allocator: std.mem.Allocator, length: u32) !Self {
        const byte_length = (length + 1) / 2; // Ceiling division
        const data = try allocator.alloc(u8, byte_length);
        @memset(data, 0);

        return Self{
            .data = data,
            .length = length,
            .allocator = allocator,
        };
    }

    /// Cleanup packed array
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }

    /// Get 4-bit value at index
    pub fn get(self: *const Self, index: u32) i8 {
        if (index >= self.length) return 0;
        
        const byte_idx = index / 2;
        const is_upper = (index % 2) == 1;
        
        if (is_upper) {
            return @as(i8, @intCast((self.data[byte_idx] >> 4) & 0x0F));
        } else {
            return @as(i8, @intCast(self.data[byte_idx] & 0x0F));
        }
    }

    /// Set 4-bit value at index
    pub fn set(self: *Self, index: u32, value: i8) void {
        if (index >= self.length) return;
        
        const byte_idx = index / 2;
        const is_upper = (index % 2) == 1;
        const clamped_val = @as(u8, @intCast(@max(0, @min(15, value))));
        
        if (is_upper) {
            self.data[byte_idx] = (self.data[byte_idx] & 0x0F) | (clamped_val << 4);
        } else {
            self.data[byte_idx] = (self.data[byte_idx] & 0xF0) | clamped_val;
        }
    }

    /// Pack array of i8 values (0-15 range)
    pub fn packFromI8(self: *Self, values: []const i8) void {
        const count = @min(values.len, self.length);
        for (0..count) |i| {
            self.set(@intCast(i), values[i]);
        }
    }

    /// Unpack to array of i8 values
    pub fn unpackToI8(self: *const Self, values: []i8) void {
        const count = @min(values.len, self.length);
        for (0..count) |i| {
            values[i] = self.get(@intCast(i));
        }
    }
};

/// Quantized matrix for 4-bit weights
pub const QuantMatrix = struct {
    packed_weights: PackedInt4,
    quant_params: QuantParams,
    rows: u32,
    cols: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create quantized matrix from float weights
    pub fn fromFloat(allocator: std.mem.Allocator, weights: []const f32, rows: u32, cols: u32) !Self {
        if (weights.len != rows * cols) return error.InvalidDimensions;

        // Compute quantization parameters
        var min_val: f32 = weights[0];
        var max_val: f32 = weights[0];
        
        for (weights[1..]) |w| {
            min_val = @min(min_val, w);
            max_val = @max(max_val, w);
        }

        // Symmetric quantization around zero
        const max_abs = @max(@abs(min_val), @abs(max_val));
        const scale = max_abs / 7.0; // Map to [-7, 7] range for 4-bit signed
        const zero_point: i8 = 0;

        const quant_params = QuantParams{
            .scale = scale,
            .zero_point = zero_point,
            .min_val = min_val,
            .max_val = max_val,
        };

        // Quantize weights
        var packed_weights = try PackedInt4.init(allocator, @intCast(weights.len));
        for (weights, 0..) |w, i| {
            const quantized = quantizeValue(w, quant_params);
            packed_weights.set(@intCast(i), quantized);
        }

        return Self{
            .packed_weights = packed_weights,
            .quant_params = quant_params,
            .rows = rows,
            .cols = cols,
            .allocator = allocator,
        };
    }

    /// Cleanup quantized matrix
    pub fn deinit(self: *Self) void {
        self.packed_weights.deinit();
    }

    /// Get dequantized weight at (row, col)
    pub fn getWeight(self: *const Self, row: u32, col: u32) f32 {
        const index = row * self.cols + col;
        const quantized = self.packed_weights.get(index);
        return dequantizeValue(quantized, self.quant_params);
    }

    /// Quantized matrix-vector multiplication
    pub fn quantMatVec(self: *const Self, input: []const f32, output: []f32) !void {
        if (input.len != self.cols or output.len != self.rows) {
            return error.IncompatibleDimensions;
        }

        // Quantize input vector
        var input_quant = try self.allocator.alloc(i8, input.len);
        defer self.allocator.free(input_quant);

        // Use same scale as weights for input quantization
        for (input, 0..) |val, i| {
            input_quant[i] = quantizeValue(val, self.quant_params);
        }

        // Perform quantized multiplication
        for (0..self.rows) |i| {
            var sum: i32 = 0;
            
            for (0..self.cols) |j| {
                const weight_idx = i * self.cols + j;
                const weight_quant = self.packed_weights.get(@intCast(weight_idx));
                sum += @as(i32, weight_quant) * @as(i32, input_quant[j]);
            }

            // Dequantize result (scale^2 factor from weight*input)
            const scale_squared = self.quant_params.scale * self.quant_params.scale;
            output[i] = @as(f32, @floatFromInt(sum)) * scale_squared;
        }
    }
};

/// Quantize float value to 4-bit signed integer
pub fn quantizeValue(value: f32, params: QuantParams) i8 {
    if (params.scale == 0.0) return 0;
    
    const scaled = value / params.scale;
    const rounded = @round(scaled);
    const clamped = @max(-7.0, @min(7.0, rounded)); // 4-bit signed range
    
    return @as(i8, @intFromFloat(clamped));
}

/// Dequantize 4-bit signed integer to float
pub fn dequantizeValue(quantized: i8, params: QuantParams) f32 {
    return @as(f32, @floatFromInt(quantized)) * params.scale;
}

/// Compute optimal quantization parameters for given data
pub fn computeQuantParams(values: []const f32) QuantParams {
    if (values.len == 0) {
        return QuantParams{
            .scale = 1.0,
            .zero_point = 0,
            .min_val = 0.0,
            .max_val = 0.0,
        };
    }

    var min_val = values[0];
    var max_val = values[0];

    for (values[1..]) |val| {
        min_val = @min(min_val, val);
        max_val = @max(max_val, val);
    }

    // Symmetric quantization
    const max_abs = @max(@abs(min_val), @abs(max_val));
    const scale = if (max_abs > 0.0) max_abs / 7.0 else 1.0;

    return QuantParams{
        .scale = scale,
        .zero_point = 0,
        .min_val = min_val,
        .max_val = max_val,
    };
}

/// Quantized convolution operation
pub fn quantConv1d(
    input: []const f32,
    weights: *const QuantMatrix,
    output: []f32,
    input_len: u32,
    kernel_size: u32,
    stride: u32
) !void {
    if (weights.cols != kernel_size) return error.InvalidKernelSize;
    
    const output_len = (input_len - kernel_size) / stride + 1;
    if (output.len != output_len * weights.rows) return error.InvalidOutputSize;

    for (0..output_len) |out_pos| {
        const input_start = out_pos * stride;
        
        for (0..weights.rows) |filter_idx| {
            var sum: i32 = 0;
            
            // Quantize input window
            for (0..kernel_size) |k| {
                const input_val = input[input_start + k];
                const input_quant = quantizeValue(input_val, weights.quant_params);
                
                const weight_idx = filter_idx * kernel_size + k;
                const weight_quant = weights.packed_weights.get(@intCast(weight_idx));
                
                sum += @as(i32, input_quant) * @as(i32, weight_quant);
            }
            
            // Dequantize result
            const scale_squared = weights.quant_params.scale * weights.quant_params.scale;
            output[out_pos * weights.rows + filter_idx] = @as(f32, @floatFromInt(sum)) * scale_squared;
        }
    }
}

/// Quantized activation functions
pub fn quantReLU(input: []const i8, output: []i8) void {
    for (input, 0..) |val, i| {
        output[i] = @max(0, val);
    }
}

pub fn quantClamp(input: []const i8, output: []i8, min_val: i8, max_val: i8) void {
    for (input, 0..) |val, i| {
        output[i] = @max(min_val, @min(max_val, val));
    }
}

/// Batch quantization of float arrays
pub fn batchQuantize(
    inputs: []const f32,
    outputs: []i8,
    params: QuantParams
) void {
    for (inputs, 0..) |val, i| {
        outputs[i] = quantizeValue(val, params);
    }
}

/// Batch dequantization of int arrays
pub fn batchDequantize(
    inputs: []const i8,
    outputs: []f32,
    params: QuantParams
) void {
    for (inputs, 0..) |val, i| {
        outputs[i] = dequantizeValue(val, params);
    }
}

/// Memory usage calculation
pub fn getMemoryUsage(float_count: u32) struct { float_bytes: u32, int4_bytes: u32, compression_ratio: f32 } {
    const float_bytes = float_count * 4; // 32-bit floats
    const int4_bytes = (float_count + 1) / 2; // 4-bit packed
    const compression_ratio = @as(f32, @floatFromInt(float_bytes)) / @as(f32, @floatFromInt(int4_bytes));
    
    return .{
        .float_bytes = float_bytes,
        .int4_bytes = int4_bytes,
        .compression_ratio = compression_ratio,
    };
}

// FFI exports for Rust integration
export fn quant_params_compute(values: [*]const f32, len: u32) QuantParams {
    const values_slice = values[0..len];
    return computeQuantParams(values_slice);
}

export fn quant_matrix_create(weights: [*]const f32, rows: u32, cols: u32) ?*QuantMatrix {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const matrix = allocator.create(QuantMatrix) catch return null;
    const weights_slice = weights[0..(rows * cols)];
    matrix.* = QuantMatrix.fromFloat(allocator, weights_slice, rows, cols) catch return null;
    
    return matrix;
}

export fn quant_matrix_destroy(matrix: ?*QuantMatrix) void {
    if (matrix) |m| {
        m.deinit();
        m.allocator.destroy(m);
    }
}

export fn quant_matrix_matvec(matrix: ?*const QuantMatrix, input: [*]const f32, input_len: u32, output: [*]f32, output_len: u32) bool {
    if (matrix == null) return false;
    
    const input_slice = input[0..input_len];
    const output_slice = output[0..output_len];
    matrix.?.quantMatVec(input_slice, output_slice) catch return false;
    return true;
}

export fn quant_pack_create(length: u32) ?*PackedInt4 {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const packed_int4 = allocator.create(PackedInt4) catch return null;
    packed_int4.* = PackedInt4.init(allocator, length) catch return null;
    
    return packed_int4;
}

export fn quant_pack_destroy(packed_data: ?*PackedInt4) void {
    if (packed_data) |p| {
        p.deinit();
        p.allocator.destroy(p);
    }
}

export fn quant_batch_quantize(inputs: [*]const f32, outputs: [*]i8, len: u32, params: QuantParams) void {
    const inputs_slice = inputs[0..len];
    const outputs_slice = outputs[0..len];
    batchQuantize(inputs_slice, outputs_slice, params);
}

export fn quant_batch_dequantize(inputs: [*]const i8, outputs: [*]f32, len: u32, params: QuantParams) void {
    const inputs_slice = inputs[0..len];
    const outputs_slice = outputs[0..len];
    batchDequantize(inputs_slice, outputs_slice, params);
}

export fn quant_memory_usage(float_count: u32) extern struct { float_bytes: u32, int4_bytes: u32, compression_ratio: f32 } {
    return getMemoryUsage(float_count);
}

// Unit tests
test "PackedInt4 operations" {
    const testing = std.testing;
    
    var packed_data = try PackedInt4.init(testing.allocator, 8);
    defer packed_data.deinit();
    
    // Test set/get
    packed_data.set(0, 5);
    packed_data.set(1, 10);
    packed_data.set(2, 3);
    packed_data.set(3, 15);
    
    try testing.expect(packed_data.get(0) == 5);
    try testing.expect(packed_data.get(1) == 10);
    try testing.expect(packed_data.get(2) == 3);
    try testing.expect(packed_data.get(3) == 15);
    
    // Test packing/unpacking
    const values = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    packed_data.packFromI8(&values);
    
    var unpacked = [_]i8{0} ** 8;
    packed_data.unpackToI8(&unpacked);
    
    for (values, 0..) |expected, i| {
        try testing.expect(unpacked[i] == expected);
    }
}

test "Quantization parameters" {
    const testing = std.testing;
    
    const values = [_]f32{ -3.5, -1.2, 0.0, 1.8, 2.7 };
    const params = computeQuantParams(&values);
    
    try testing.expect(params.scale > 0.0);
    try testing.expect(params.zero_point == 0);
    try testing.expect(params.min_val == -3.5);
    try testing.expect(params.max_val == 2.7);
}

test "Quantize/dequantize values" {
    const testing = std.testing;
    
    const params = QuantParams{
        .scale = 0.5,
        .zero_point = 0,
        .min_val = -3.5,
        .max_val = 3.5,
    };
    
    // Test quantization
    const val = 1.75;
    const quantized = quantizeValue(val, params);
    const dequantized = dequantizeValue(quantized, params);
    
    try testing.expect(quantized == 4); // 1.75 / 0.5 = 3.5, rounded to 4
    try testing.expect(@abs(dequantized - 2.0) < 0.1); // 4 * 0.5 = 2.0
}

test "Quantized matrix operations" {
    const testing = std.testing;
    
    const weights = [_]f32{
        1.0, -2.0, 3.0,
        -1.5, 2.5, -0.5,
    };
    
    var qmatrix = try QuantMatrix.fromFloat(testing.allocator, &weights, 2, 3);
    defer qmatrix.deinit();
    
    const input = [_]f32{ 1.0, 2.0, -1.0 };
    var output = [_]f32{ 0.0, 0.0 };
    
    try qmatrix.quantMatVec(&input, &output);
    
    // Results should be approximately correct (with quantization error)
    // Expected: [1*1 + (-2)*2 + 3*(-1), (-1.5)*1 + 2.5*2 + (-0.5)*(-1)]
    //         = [-6, 4]
    try testing.expect(@abs(output[0] - (-6.0)) < 1.0); // Allow quantization error
    try testing.expect(@abs(output[1] - 4.0) < 1.0);
}

test "Memory usage calculation" {
    const testing = std.testing;
    
    const usage = getMemoryUsage(1000);
    try testing.expect(usage.float_bytes == 4000); // 1000 * 4 bytes
    try testing.expect(usage.int4_bytes == 500);   // 1000 / 2 bytes
    try testing.expect(@abs(usage.compression_ratio - 8.0) < 0.1); // ~8x compression
}