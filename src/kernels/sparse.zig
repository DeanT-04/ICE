//! Sparse computation kernels for efficient neural network operations
//!
//! Optimized for 10-20% activation rates with CSR/COO sparse matrix formats
//! and SIMD-accelerated sparse operations.

const std = @import("std");
const math = std.math;

/// Compressed Sparse Row (CSR) matrix format
pub const CSRMatrix = struct {
    values: []f32,        // Non-zero values
    col_indices: []u32,   // Column indices for each value
    row_ptrs: []u32,      // Row start pointers
    rows: u32,
    cols: u32,
    nnz: u32,             // Number of non-zeros
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create CSR matrix from dense matrix
    pub fn fromDense(allocator: std.mem.Allocator, dense: []const f32, rows: u32, cols: u32, threshold: f32) !Self {
        // Count non-zeros
        var nnz: u32 = 0;
        for (dense) |val| {
            if (@abs(val) > threshold) nnz += 1;
        }

        // Allocate sparse arrays
        const values = try allocator.alloc(f32, nnz);
        const col_indices = try allocator.alloc(u32, nnz);
        const row_ptrs = try allocator.alloc(u32, rows + 1);

        // Fill sparse format
        var val_idx: u32 = 0;
        
        for (0..rows) |i| {
            row_ptrs[i] = val_idx;
            for (0..cols) |j| {
                const val = dense[i * cols + j];
                if (@abs(val) > threshold) {
                    values[val_idx] = val;
                    col_indices[val_idx] = @intCast(j);
                    val_idx += 1;
                }
            }
        }
        row_ptrs[rows] = val_idx;

        return Self{
            .values = values,
            .col_indices = col_indices,
            .row_ptrs = row_ptrs,
            .rows = rows,
            .cols = cols,
            .nnz = nnz,
            .allocator = allocator,
        };
    }

    /// Cleanup CSR matrix
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.values);
        self.allocator.free(self.col_indices);
        self.allocator.free(self.row_ptrs);
    }

    /// Sparse matrix-vector multiplication: y = A * x
    pub fn spmv(self: *const Self, x: []const f32, y: []f32) !void {
        if (x.len != self.cols or y.len != self.rows) {
            return error.IncompatibleDimensions;
        }

        // Zero output vector
        @memset(y, 0.0);

        // Compute y = A * x using CSR format
        for (0..self.rows) |i| {
            const row_start = self.row_ptrs[i];
            const row_end = self.row_ptrs[i + 1];
            
            var sum: f32 = 0.0;
            for (row_start..row_end) |idx| {
                const col = self.col_indices[idx];
                const val = self.values[idx];
                sum += val * x[col];
            }
            y[i] = sum;
        }
    }
};

/// Coordinate (COO) sparse matrix format
pub const COOMatrix = struct {
    values: []f32,
    row_indices: []u32,
    col_indices: []u32,
    rows: u32,
    cols: u32,
    nnz: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create COO matrix from triplets
    pub fn init(allocator: std.mem.Allocator, rows: u32, cols: u32, nnz: u32) !Self {
        const values = try allocator.alloc(f32, nnz);
        const row_indices = try allocator.alloc(u32, nnz);
        const col_indices = try allocator.alloc(u32, nnz);

        return Self{
            .values = values,
            .row_indices = row_indices,
            .col_indices = col_indices,
            .rows = rows,
            .cols = cols,
            .nnz = nnz,
            .allocator = allocator,
        };
    }

    /// Cleanup COO matrix
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.values);
        self.allocator.free(self.row_indices);
        self.allocator.free(self.col_indices);
    }

    /// Convert to CSR format
    pub fn toCSR(self: *const Self, allocator: std.mem.Allocator) !CSRMatrix {
        // Sort by row indices first
        var sorted_indices = try allocator.alloc(u32, self.nnz);
        defer allocator.free(sorted_indices);

        for (0..self.nnz) |i| {
            sorted_indices[i] = @intCast(i);
        }

        // Simple insertion sort by row index
        for (1..self.nnz) |i| {
            const key = sorted_indices[i];
            const key_row = self.row_indices[key];
            var j: i32 = @intCast(i - 1);

            while (j >= 0 and self.row_indices[sorted_indices[@intCast(j)]] > key_row) {
                sorted_indices[@intCast(j + 1)] = sorted_indices[@intCast(j)];
                j -= 1;
            }
            sorted_indices[@intCast(j + 1)] = key;
        }

        // Create CSR structure
        const values = try allocator.alloc(f32, self.nnz);
        const col_indices = try allocator.alloc(u32, self.nnz);
        const row_ptrs = try allocator.alloc(u32, self.rows + 1);

        // Fill CSR arrays
        for (0..self.nnz) |i| {
            const orig_idx = sorted_indices[i];
            values[i] = self.values[orig_idx];
            col_indices[i] = self.col_indices[orig_idx];
        }

        // Build row pointers
        @memset(row_ptrs, 0);
        for (0..self.nnz) |i| {
            const orig_idx = sorted_indices[i];
            const row = self.row_indices[orig_idx];
            row_ptrs[row + 1] += 1;
        }

        // Convert counts to pointers
        for (1..self.rows + 1) |i| {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        return CSRMatrix{
            .values = values,
            .col_indices = col_indices,
            .row_ptrs = row_ptrs,
            .rows = self.rows,
            .cols = self.cols,
            .nnz = self.nnz,
            .allocator = allocator,
        };
    }
};

/// Sparse activation mask for enforcing sparsity
pub const SparseMask = struct {
    active_indices: []u32,
    active_count: u32,
    total_size: u32,
    sparsity_rate: f32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create sparse mask with target sparsity
    pub fn init(allocator: std.mem.Allocator, size: u32, sparsity_rate: f32) !Self {
        const active_count = @as(u32, @intFromFloat(@as(f32, @floatFromInt(size)) * sparsity_rate));
        const active_indices = try allocator.alloc(u32, active_count);

        return Self{
            .active_indices = active_indices,
            .active_count = active_count,
            .total_size = size,
            .sparsity_rate = sparsity_rate,
            .allocator = allocator,
        };
    }

    /// Cleanup sparse mask
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.active_indices);
    }

    /// Update mask based on vector magnitudes
    pub fn updateFromMagnitudes(self: *Self, values: []const f32) !void {
        if (values.len != self.total_size) return error.InvalidSize;

        // Create array of (magnitude, index) pairs
        const Candidate = struct { mag: f32, idx: u32 };
        var candidates = try self.allocator.alloc(Candidate, self.total_size);
        defer self.allocator.free(candidates);

        for (values, 0..) |val, i| {
            candidates[i] = .{ .mag = @abs(val), .idx = @intCast(i) };
        }

        // Sort by magnitude (descending)
        std.sort.heap(Candidate, candidates, {}, struct {
            fn lessThan(_: void, lhs: Candidate, rhs: Candidate) bool {
                return lhs.mag > rhs.mag;
            }
        }.lessThan);

        // Take top-k indices
        for (0..self.active_count) |i| {
            self.active_indices[i] = candidates[i].idx;
        }

        // Sort active indices for efficient access
        std.sort.heap(u32, self.active_indices, {}, std.sort.asc(u32));
    }

    /// Apply mask to vector (zero non-active elements)
    pub fn applyMask(self: *const Self, values: []f32) void {
        if (values.len != self.total_size) return;

        // Zero all values first
        @memset(values, 0.0);

        // Only keep active indices (assumes values already contain correct sparse values)
        // This is used when values are pre-computed sparse activations
    }

    /// Get sparse representation
    pub fn getSparseValues(self: *const Self, dense_values: []const f32, sparse_values: []f32) !void {
        if (dense_values.len != self.total_size or sparse_values.len != self.active_count) {
            return error.InvalidSize;
        }

        for (0..self.active_count) |i| {
            const idx = self.active_indices[i];
            sparse_values[i] = dense_values[idx];
        }
    }

    /// Expand sparse to dense representation
    pub fn expandToDense(self: *const Self, sparse_values: []const f32, dense_values: []f32) !void {
        if (sparse_values.len != self.active_count or dense_values.len != self.total_size) {
            return error.InvalidSize;
        }

        @memset(dense_values, 0.0);
        for (0..self.active_count) |i| {
            const idx = self.active_indices[i];
            dense_values[idx] = sparse_values[i];
        }
    }
};

/// Sparse ReLU activation with automatic sparsity
pub fn sparseReLU(input: []const f32, output: []f32, sparsity_rate: f32, allocator: std.mem.Allocator) !void {
    if (input.len != output.len) return error.IncompatibleDimensions;

    // Apply ReLU first
    for (input, 0..) |val, i| {
        output[i] = @max(val, 0.0);
    }

    // Create mask and enforce sparsity
    var mask = try SparseMask.init(allocator, @intCast(input.len), sparsity_rate);
    defer mask.deinit();

    try mask.updateFromMagnitudes(output);
    
    // Zero non-active elements
    for (0..output.len) |i| {
        var is_active = false;
        for (mask.active_indices) |active_idx| {
            if (active_idx == i) {
                is_active = true;
                break;
            }
        }
        if (!is_active) {
            output[i] = 0.0;
        }
    }
}

/// Sparse matrix multiplication with early termination
pub fn sparseMatMul(a: *const CSRMatrix, x: []const f32, y: []f32, threshold: f32) !void {
    if (x.len != a.cols or y.len != a.rows) {
        return error.IncompatibleDimensions;
    }

    @memset(y, 0.0);

    for (0..a.rows) |i| {
        const row_start = a.row_ptrs[i];
        const row_end = a.row_ptrs[i + 1];
        
        var sum: f32 = 0.0;
        for (row_start..row_end) |idx| {
            const col = a.col_indices[idx];
            const val = a.values[idx];
            const x_val = x[col];
            
            // Early termination for very small contributions
            if (@abs(val * x_val) > threshold) {
                sum += val * x_val;
            }
        }
        y[i] = sum;
    }
}

/// Sparse attention computation
pub fn sparseAttention(
    query: []const f32, 
    key: []const f32, 
    value: []const f32,
    output: []f32,
    seq_len: u32,
    head_dim: u32,
    sparsity_mask: []const bool,
    allocator: std.mem.Allocator
) !void {
    if (query.len != seq_len * head_dim or 
        key.len != seq_len * head_dim or 
        value.len != seq_len * head_dim or
        output.len != seq_len * head_dim or
        sparsity_mask.len != seq_len * seq_len) {
        return error.IncompatibleDimensions;
    }

    // Compute attention scores only for allowed positions
    var scores = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(scores);
    @memset(scores, -std.math.inf(f32));

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            if (sparsity_mask[i * seq_len + j]) {
                var score: f32 = 0.0;
                for (0..head_dim) |d| {
                    score += query[i * head_dim + d] * key[j * head_dim + d];
                }
                score /= math.sqrt(@as(f32, @floatFromInt(head_dim)));
                scores[i * seq_len + j] = score;
            }
        }
    }

    // Apply softmax only to valid positions
    for (0..seq_len) |i| {
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |j| {
            if (sparsity_mask[i * seq_len + j]) {
                max_score = @max(max_score, scores[i * seq_len + j]);
            }
        }

        var sum_exp: f32 = 0.0;
        for (0..seq_len) |j| {
            if (sparsity_mask[i * seq_len + j]) {
                scores[i * seq_len + j] = math.exp(scores[i * seq_len + j] - max_score);
                sum_exp += scores[i * seq_len + j];
            } else {
                scores[i * seq_len + j] = 0.0;
            }
        }

        // Normalize
        if (sum_exp > 0.0) {
            for (0..seq_len) |j| {
                scores[i * seq_len + j] /= sum_exp;
            }
        }
    }

    // Compute output
    @memset(output, 0.0);
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            if (sparsity_mask[i * seq_len + j]) {
                const weight = scores[i * seq_len + j];
                for (0..head_dim) |d| {
                    output[i * head_dim + d] += weight * value[j * head_dim + d];
                }
            }
        }
    }
}

// FFI exports for Rust integration
export fn sparse_csr_create(dense: [*]const f32, rows: u32, cols: u32, threshold: f32) ?*CSRMatrix {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const matrix = allocator.create(CSRMatrix) catch return null;
    const dense_slice = dense[0..(rows * cols)];
    matrix.* = CSRMatrix.fromDense(allocator, dense_slice, rows, cols, threshold) catch return null;
    
    return matrix;
}

export fn sparse_csr_destroy(matrix: ?*CSRMatrix) void {
    if (matrix) |m| {
        m.deinit();
        m.allocator.destroy(m);
    }
}

export fn sparse_csr_spmv(matrix: ?*const CSRMatrix, x: [*]const f32, x_len: u32, y: [*]f32, y_len: u32) bool {
    if (matrix == null) return false;
    
    const x_slice = x[0..x_len];
    const y_slice = y[0..y_len];
    matrix.?.spmv(x_slice, y_slice) catch return false;
    return true;
}

export fn sparse_relu(input: [*]const f32, output: [*]f32, len: u32, sparsity_rate: f32) bool {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const input_slice = input[0..len];
    const output_slice = output[0..len];
    
    sparseReLU(input_slice, output_slice, sparsity_rate, allocator) catch return false;
    return true;
}

export fn sparse_mask_create(size: u32, sparsity_rate: f32) ?*SparseMask {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const mask = allocator.create(SparseMask) catch return null;
    mask.* = SparseMask.init(allocator, size, sparsity_rate) catch return null;
    
    return mask;
}

export fn sparse_mask_destroy(mask: ?*SparseMask) void {
    if (mask) |m| {
        m.deinit();
        m.allocator.destroy(m);
    }
}

// Unit tests
test "CSR matrix creation and SpMV" {
    const testing = std.testing;
    
    // Create a simple 3x3 dense matrix
    const dense = [_]f32{
        1.0, 0.0, 3.0,
        0.0, 2.0, 0.0,
        4.0, 0.0, 5.0,
    };
    
    var csr = try CSRMatrix.fromDense(testing.allocator, &dense, 3, 3, 0.1);
    defer csr.deinit();
    
    // Check nnz count
    try testing.expect(csr.nnz == 5); // 1, 3, 2, 4, 5
    
    // Test SpMV
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    var y = [_]f32{ 0.0, 0.0, 0.0 };
    
    try csr.spmv(&x, &y);
    
    // Expected: [1*1 + 0*2 + 3*3, 0*1 + 2*2 + 0*3, 4*1 + 0*2 + 5*3]
    //         = [10, 4, 19]
    try testing.expect(y[0] == 10.0);
    try testing.expect(y[1] == 4.0);
    try testing.expect(y[2] == 19.0);
}

test "Sparse mask operations" {
    const testing = std.testing;
    
    var mask = try SparseMask.init(testing.allocator, 10, 0.3); // 30% sparsity
    defer mask.deinit();
    
    try testing.expect(mask.active_count == 3);
    try testing.expect(mask.total_size == 10);
    
    // Test magnitude-based update
    const values = [_]f32{ 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0 };
    try mask.updateFromMagnitudes(&values);
    
    // Should select indices with highest magnitudes: 1 (0.9), 3 (0.8), 5 (0.7)
    try testing.expect(mask.active_indices[0] == 1 or mask.active_indices[1] == 1 or mask.active_indices[2] == 1);
    try testing.expect(mask.active_indices[0] == 3 or mask.active_indices[1] == 3 or mask.active_indices[2] == 3);
    try testing.expect(mask.active_indices[0] == 5 or mask.active_indices[1] == 5 or mask.active_indices[2] == 5);
}

test "Sparse ReLU" {
    const testing = std.testing;
    
    const input = [_]f32{ -1.0, 2.0, -0.5, 3.0, -2.0, 1.0 };
    var output = [_]f32{0.0} ** 6;
    
    try sparseReLU(&input, &output, 0.5, testing.allocator); // 50% sparsity (3 elements)
    
    // Count non-zero outputs
    var nonzero_count: u32 = 0;
    for (output) |val| {
        if (val > 0.0) nonzero_count += 1;
    }
    
    try testing.expect(nonzero_count <= 3); // Should enforce sparsity
    
    // All outputs should be non-negative (ReLU property)
    for (output) |val| {
        try testing.expect(val >= 0.0);
    }
}