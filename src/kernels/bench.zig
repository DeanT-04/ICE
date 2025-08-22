//! Benchmark suite for ultra-fast AI kernels
//! Measures performance of SNN, matrix, sparse, and quantization operations

const std = @import("std");
const print = std.debug.print;
const time = std.time;

const snn = @import("snn.zig");
const matrix = @import("matrix.zig");
const sparse = @import("sparse.zig");
const quantize = @import("quantize.zig");
const lib = @import("lib.zig");

/// Benchmark configuration
const BenchConfig = struct {
    iterations: u32 = 1000,
    warmup_iterations: u32 = 100,
    input_sizes: []const u32 = &[_]u32{ 64, 128, 256, 512, 1024 },
};

/// Benchmark result
const BenchResult = struct {
    operation: []const u8,
    input_size: u32,
    avg_time_ns: u64,
    ops_per_sec: f64,
    memory_usage_mb: f64,
};

/// Timer utility
const Timer = struct {
    start_time: u64,
    
    const Self = @This();
    
    pub fn start() Self {
        return Self{ .start_time = time.nanoTimestamp() };
    }
    
    pub fn read(self: *const Self) u64 {
        return time.nanoTimestamp() - self.start_time;
    }
};

/// Memory tracking
var memory_tracker = std.heap.GeneralPurposeAllocator(.{
    .enable_memory_limit = true,
}){};
const allocator = memory_tracker.allocator();

/// Run benchmark with timing
fn benchmarkOperation(
    comptime name: []const u8,
    operation: anytype,
    args: anytype,
    config: BenchConfig,
    input_size: u32,
) !BenchResult {
    // Warmup
    for (0..config.warmup_iterations) |_| {
        _ = try @call(.auto, operation, args);
    }
    
    const initial_memory = memory_tracker.total_requested_bytes;
    
    // Actual benchmark
    const timer = Timer.start();
    for (0..config.iterations) |_| {
        _ = try @call(.auto, operation, args);
    }
    const total_time = timer.read();
    
    const final_memory = memory_tracker.total_requested_bytes;
    const memory_used = final_memory - initial_memory;
    
    const avg_time = total_time / config.iterations;
    const ops_per_sec = @as(f64, @floatFromInt(config.iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9);
    const memory_mb = @as(f64, @floatFromInt(memory_used)) / (1024.0 * 1024.0);
    
    return BenchResult{
        .operation = name,
        .input_size = input_size,
        .avg_time_ns = avg_time,
        .ops_per_sec = ops_per_sec,
        .memory_usage_mb = memory_mb,
    };
}

/// Benchmark SNN operations
fn benchmarkSNN(config: BenchConfig, input_size: u32) !BenchResult {
    const snn_config = snn.SnnConfig{
        .input_size = input_size,
        .output_size = input_size / 2,
        .threshold = 1.0,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.3,
    };
    
    var kernel = try snn.SnnKernel.init(allocator, snn_config);
    defer kernel.deinit();
    
    const input = try allocator.alloc(f32, input_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, input_size / 2);
    defer allocator.free(output);
    
    // Initialize random input
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    for (input) |*val| {
        val.* = random.float(f32) * 2.0 - 1.0; // Range [-1, 1]
    }
    
    const snn_forward = struct {
        fn call(k: *snn.SnnKernel, in: []const f32, out: []f32) !void {
            try k.forward(in, out);
        }
    }.call;
    
    return benchmarkOperation("SNN Forward", snn_forward, .{ &kernel, input, output }, config, input_size);
}

/// Benchmark matrix multiplication
fn benchmarkMatmul(config: BenchConfig, input_size: u32) !BenchResult {
    var a = try matrix.Matrix.init(allocator, input_size, input_size);
    defer a.deinit();
    var b = try matrix.Matrix.init(allocator, input_size, input_size);
    defer b.deinit();
    var c = try matrix.Matrix.init(allocator, input_size, input_size);
    defer c.deinit();
    
    // Initialize random matrices
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    
    for (a.data) |*val| val.* = random.float(f32);
    for (b.data) |*val| val.* = random.float(f32);
    
    return benchmarkOperation("Matrix Multiply", matrix.matmul, .{ &a, &b, &c }, config, input_size);
}

/// Benchmark matrix-vector multiplication
fn benchmarkMatvec(config: BenchConfig, input_size: u32) !BenchResult {
    var a = try matrix.Matrix.init(allocator, input_size, input_size);
    defer a.deinit();
    
    const x = try allocator.alloc(f32, input_size);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, input_size);
    defer allocator.free(y);
    
    // Initialize random data
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    
    for (a.data) |*val| val.* = random.float(f32);
    for (x) |*val| val.* = random.float(f32);
    
    return benchmarkOperation("Matrix-Vector Multiply", matrix.matvec, .{ &a, x, y }, config, input_size);
}

/// Benchmark activation functions
fn benchmarkActivation(config: BenchConfig, input_size: u32) !BenchResult {
    const input = try allocator.alloc(f32, input_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, input_size);
    defer allocator.free(output);
    
    // Initialize random input
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    for (input) |*val| {
        val.* = random.float(f32) * 4.0 - 2.0; // Range [-2, 2]
    }
    
    return benchmarkOperation("ReLU Activation", matrix.applyActivation, .{ input, output, matrix.ActivationFn.ReLU }, config, input_size);
}

/// Benchmark sparse operations
fn benchmarkSparse(config: BenchConfig, input_size: u32) !BenchResult {
    const density = 0.1; // 10% sparse
    var sparse_matrix = try sparse.SparseMatrix.init(allocator, input_size, input_size, density);
    defer sparse_matrix.deinit();
    
    const x = try allocator.alloc(f32, input_size);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, input_size);
    defer allocator.free(y);
    
    // Initialize random data
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    
    for (x) |*val| val.* = random.float(f32);
    
    // Add some random sparse entries
    for (0..@as(u32, @intFromFloat(@as(f32, @floatFromInt(input_size * input_size)) * density))) |_| {
        const row = random.intRangeLessThan(u32, 0, input_size);
        const col = random.intRangeLessThan(u32, 0, input_size);
        const val = random.float(f32);
        try sparse_matrix.set(row, col, val);
    }
    
    return benchmarkOperation("Sparse MatVec", sparse_matrix.matvec, .{ x, y }, config, input_size);
}

/// Benchmark quantization
fn benchmarkQuantization(config: BenchConfig, input_size: u32) !BenchResult {
    const input = try allocator.alloc(f32, input_size);
    defer allocator.free(input);
    const output = try allocator.alloc(u8, input_size / 2);
    defer allocator.free(output);
    var scale: f32 = 0.0;
    
    // Initialize random input
    var prng = std.rand.DefaultPrng.init(@intCast(time.timestamp()));
    const random = prng.random();
    for (input) |*val| {
        val.* = random.float(f32) * 10.0 - 5.0; // Range [-5, 5]
    }
    
    const quantize_fn = struct {
        fn call(in: [*]const f32, out: [*]u8, sc: [*]f32, len: usize) void {
            lib.matrix_quantize_4bit(in, out, sc, len);
        }
    }.call;
    
    return benchmarkOperation("4-bit Quantization", quantize_fn, .{ input.ptr, output.ptr, &scale, input.len }, config, input_size);
}

/// Print benchmark results
fn printResults(results: []const BenchResult) void {
    print("\n=== Ultra-Fast AI Kernels Benchmark Results ===\n\n");
    print("{s:<20} {s:<10} {s:<15} {s:<15} {s:<15}\n", .{ "Operation", "Size", "Avg Time (μs)", "Ops/sec", "Memory (MB)" });
    print("{s}\n", .{"-" ** 80});
    
    for (results) |result| {
        const avg_time_us = @as(f64, @floatFromInt(result.avg_time_ns)) / 1000.0;
        print("{s:<20} {d:<10} {d:<15.2} {d:<15.0} {d:<15.2}\n", .{
            result.operation,
            result.input_size,
            avg_time_us,
            result.ops_per_sec,
            result.memory_usage_mb,
        });
    }
    
    print("\n");
}

/// Main benchmark runner
pub fn main() !void {
    print("Starting Ultra-Fast AI Kernels Benchmark Suite...\n");
    
    const config = BenchConfig{};
    var results = std.ArrayList(BenchResult).init(allocator);
    defer results.deinit();
    
    // Run benchmarks for different input sizes
    for (config.input_sizes) |size| {
        print("Benchmarking with input size: {d}\n", .{size});
        
        // SNN benchmarks
        if (benchmarkSNN(config, size)) |result| {
            try results.append(result);
        } else |err| {
            print("SNN benchmark failed: {}\n", .{err});
        }
        
        // Matrix operation benchmarks
        if (benchmarkMatmul(config, size)) |result| {
            try results.append(result);
        } else |err| {
            print("Matmul benchmark failed: {}\n", .{err});
        }
        
        if (benchmarkMatvec(config, size)) |result| {
            try results.append(result);
        } else |err| {
            print("Matvec benchmark failed: {}\n", .{err});
        }
        
        if (benchmarkActivation(config, size)) |result| {
            try results.append(result);
        } else |err| {
            print("Activation benchmark failed: {}\n", .{err});
        }
        
        // Sparse operation benchmarks
        if (benchmarkSparse(config, size)) |result| {
            try results.append(result);
        } else |err| {
            print("Sparse benchmark failed: {}\n", .{err});
        }
        
        // Quantization benchmarks (only for even sizes)
        if (size % 2 == 0) {
            if (benchmarkQuantization(config, size)) |result| {
                try results.append(result);
            } else |err| {
                print("Quantization benchmark failed: {}\n", .{err});
            }
        }
    }
    
    // Print results
    printResults(results.items);
    
    // Memory cleanup check
    const leaked = memory_tracker.deinit();
    if (leaked == .leak) {
        print("WARNING: Memory leaks detected!\n");
    } else {
        print("Memory cleanup successful.\n");
    }
}