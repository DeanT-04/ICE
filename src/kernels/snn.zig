//! SNN (Spiking Neural Network) kernels for event-driven processing
//!
//! High-performance Zig implementations for spike generation, membrane dynamics,
//! and sparse activation processing. Optimized for CPU execution with SIMD.

const std = @import("std");
const math = std.math;
const simd = std.simd;

/// SIMD vector size for parallel processing
const VECTOR_SIZE = 8;
const VectorF32 = @Vector(VECTOR_SIZE, f32);
const VectorU8 = @Vector(VECTOR_SIZE, u8);
const VectorU32 = @Vector(VECTOR_SIZE, u32);

/// SNN configuration structure
pub const SnnConfig = extern struct {
    input_size: u32,
    output_size: u32,
    threshold: f32,
    decay_rate: f32,
    refractory_period: u32,
    sparse_rate: f32,
};

/// Neuron state structure for efficient memory layout
pub const NeuronState = extern struct {
    membrane_potential: f32,
    spike_output: u8,
    refractory_counter: u32,
    last_spike_time: u32,
};

/// Spike event structure for event-driven processing
pub const SpikeEvent = extern struct {
    neuron_id: u32,
    timestamp: u32,
    amplitude: f32,
};

/// Sparse connection structure
pub const SparseConnection = extern struct {
    pre_neuron: u32,
    post_neuron: u32,
    weight: f32,
    delay: u32,
};

/// Error types for SNN operations
pub const SnnError = error{
    InvalidConfiguration,
    MemoryAllocation,
    InvalidInput,
    BufferOverflow,
};

/// SNN kernel context
pub const SnnKernel = struct {
    config: SnnConfig,
    neuron_states: []NeuronState,
    weights: []f32,
    spike_buffer: []SpikeEvent,
    connections: []SparseConnection,
    current_time: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Initialize SNN kernel
    pub fn init(allocator: std.mem.Allocator, config: SnnConfig) !Self {
        // Validate configuration
        if (config.input_size == 0 or config.output_size == 0) {
            return SnnError.InvalidConfiguration;
        }
        if (config.threshold <= 0 or config.decay_rate <= 0 or config.decay_rate >= 1) {
            return SnnError.InvalidConfiguration;
        }

        // Allocate memory for neuron states
        const total_neurons = config.input_size + config.output_size;
        const neuron_states = try allocator.alloc(NeuronState, total_neurons);
        
        // Initialize neuron states
        for (neuron_states) |*state| {
            state.* = NeuronState{
                .membrane_potential = 0.0,
                .spike_output = 0,
                .refractory_counter = 0,
                .last_spike_time = 0,
            };
        }

        // Allocate weight matrix (input_size x output_size)
        const weight_count = config.input_size * config.output_size;
        const weights = try allocator.alloc(f32, weight_count);
        
        // Initialize weights with small random values
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();
        for (weights) |*weight| {
            weight.* = (random.float(f32) - 0.5) * 0.2; // Range [-0.1, 0.1]
        }

        // Allocate spike event buffer
        const spike_buffer_size = total_neurons * 10; // Buffer for spike events
        const spike_buffer = try allocator.alloc(SpikeEvent, spike_buffer_size);

        // Create sparse connections (30% connectivity)
        const connection_count = @as(u32, @intFromFloat(@as(f32, @floatFromInt(weight_count)) * 0.3));
        const connections = try allocator.alloc(SparseConnection, connection_count);
        
        // Initialize sparse connections
        var conn_idx: u32 = 0;
        for (0..config.input_size) |i| {
            for (0..config.output_size) |j| {
                if (random.float(f32) < 0.3 and conn_idx < connection_count) { // 30% sparsity
                    connections[conn_idx] = SparseConnection{
                        .pre_neuron = @intCast(i),
                        .post_neuron = @intCast(config.input_size + j),
                        .weight = weights[i * config.output_size + j],
                        .delay = random.intRangeAtMost(u32, 1, 5), // 1-5 timestep delays
                    };
                    conn_idx += 1;
                }
            }
        }

        return Self{
            .config = config,
            .neuron_states = neuron_states,
            .weights = weights,
            .spike_buffer = spike_buffer,
            .connections = connections[0..conn_idx],
            .current_time = 0,
            .allocator = allocator,
        };
    }

    /// Cleanup allocated memory
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.neuron_states);
        self.allocator.free(self.weights);
        self.allocator.free(self.spike_buffer);
        self.allocator.free(self.connections);
    }

    /// Process input through SNN with event-driven dynamics
    pub fn forward(self: *Self, input: []const f32, output: []f32) !void {
        if (input.len != self.config.input_size or output.len != self.config.output_size) {
            return SnnError.InvalidInput;
        }

        self.current_time += 1;

        // Step 1: Update input neuron states
        try self.updateInputNeurons(input);

        // Step 2: Process spike propagation through connections
        try self.propagateSpikes();

        // Step 3: Update output neuron membrane dynamics
        try self.updateMembraneDynamics();

        // Step 4: Generate output spikes and apply sparsity
        try self.generateOutputSpikes(output);

        // Step 5: Apply refractory periods
        self.updateRefractoryStates();
    }

    /// Update input neuron states
    fn updateInputNeurons(self: *Self, input: []const f32) !void {
        const input_neurons = self.neuron_states[0..self.config.input_size];
        
        // Process in SIMD vectors for performance
        var i: u32 = 0;
        while (i + VECTOR_SIZE <= self.config.input_size) : (i += VECTOR_SIZE) {
            const input_vec: VectorF32 = input[i..i + VECTOR_SIZE][0..VECTOR_SIZE].*;
            const threshold_vec: VectorF32 = @splat(self.config.threshold);
            
            // Check which neurons spike
            const spike_mask = input_vec > threshold_vec;
            
            // Update neuron states
            for (0..VECTOR_SIZE) |j| {
                const neuron_idx = i + j;
                const neuron = &input_neurons[neuron_idx];
                
                if (spike_mask[j]) {
                    neuron.spike_output = 1;
                    neuron.membrane_potential = 0.0; // Reset after spike
                    neuron.last_spike_time = self.current_time;
                } else {
                    neuron.spike_output = 0;
                    neuron.membrane_potential = input[neuron_idx];
                }
            }
        }
        
        // Handle remaining neurons
        while (i < self.config.input_size) : (i += 1) {
            const neuron = &input_neurons[i];
            if (input[i] > self.config.threshold) {
                neuron.spike_output = 1;
                neuron.membrane_potential = 0.0;
                neuron.last_spike_time = self.current_time;
            } else {
                neuron.spike_output = 0;
                neuron.membrane_potential = input[i];
            }
        }
    }

    /// Propagate spikes through sparse connections
    fn propagateSpikes(self: *Self) !void {
        for (self.connections) |connection| {
            const pre_neuron = &self.neuron_states[connection.pre_neuron];
            const post_neuron = &self.neuron_states[connection.post_neuron];
            
            // Check if pre-neuron spiked and connection delay is satisfied
            if (pre_neuron.spike_output == 1) {
                const spike_time_diff = self.current_time - pre_neuron.last_spike_time;
                if (spike_time_diff >= connection.delay) {
                    // Apply synaptic current to post-neuron
                    post_neuron.membrane_potential += connection.weight;
                }
            }
        }
    }

    /// Update membrane dynamics for output neurons
    fn updateMembraneDynamics(self: *Self) !void {
        const output_start = self.config.input_size;
        const output_neurons = self.neuron_states[output_start..];
        const decay_rate = self.config.decay_rate;
        
        // Vectorized membrane potential decay
        var i: u32 = 0;
        while (i + VECTOR_SIZE <= self.config.output_size) : (i += VECTOR_SIZE) {
            for (0..VECTOR_SIZE) |j| {
                const neuron_idx = i + j;
                const neuron = &output_neurons[neuron_idx];
                
                // Skip if in refractory period
                if (neuron.refractory_counter > 0) continue;
                
                // Apply exponential decay: V(t+1) = V(t) * decay_rate
                neuron.membrane_potential *= decay_rate;
            }
        }
        
        // Handle remaining neurons
        while (i < self.config.output_size) : (i += 1) {
            const neuron = &output_neurons[i];
            if (neuron.refractory_counter == 0) {
                neuron.membrane_potential *= decay_rate;
            }
        }
    }

    /// Generate output spikes with sparsity control
    fn generateOutputSpikes(self: *Self, output: []f32) !void {
        const output_start = self.config.input_size;
        const output_neurons = self.neuron_states[output_start..];
        var spike_count: u32 = 0;
        
        // Generate spikes based on threshold crossing
        for (output_neurons, 0..) |*neuron, i| {
            if (neuron.refractory_counter > 0) {
                neuron.spike_output = 0;
                output[i] = 0.0;
                continue;
            }
            
            if (neuron.membrane_potential >= self.config.threshold) {
                neuron.spike_output = 1;
                neuron.membrane_potential = 0.0; // Reset potential
                neuron.last_spike_time = self.current_time;
                neuron.refractory_counter = self.config.refractory_period;
                spike_count += 1;
                output[i] = 1.0;
            } else {
                neuron.spike_output = 0;
                output[i] = 0.0;
            }
        }
        
        // Apply sparsity constraint
        const target_spikes = @as(u32, @intFromFloat(@as(f32, @floatFromInt(self.config.output_size)) * self.config.sparse_rate));
        if (spike_count > target_spikes) {
            try self.enforceSparsity(output, spike_count, target_spikes);
        }
    }

    /// Enforce sparsity by suppressing weakest spikes
    fn enforceSparsity(self: *Self, output: []f32, current_spikes: u32, target_spikes: u32) !void {
        if (current_spikes <= target_spikes) return;
        
        const output_start = self.config.input_size;
        const output_neurons = self.neuron_states[output_start..];
        
        // Create array of (potential, index) pairs for sorting
        const SpikeEntry = struct { potential: f32, index: u32 };
        var spike_strengths = try self.allocator.alloc(SpikeEntry, current_spikes);
        defer self.allocator.free(spike_strengths);
        
        var spike_idx: u32 = 0;
        for (output_neurons, 0..) |neuron, i| {
            if (neuron.spike_output == 1) {
                spike_strengths[spike_idx] = .{
                    .potential = neuron.membrane_potential,
                    .index = @intCast(i),
                };
                spike_idx += 1;
            }
        }
        
        // Sort by potential strength (descending)
        std.sort.heap(SpikeEntry, spike_strengths, {}, struct {
            fn lessThan(_: void, lhs: SpikeEntry, rhs: SpikeEntry) bool {
                return lhs.potential > rhs.potential; // Descending order
            }
        }.lessThan);
        
        // Keep only the strongest spikes
        for (spike_strengths[target_spikes..]) |spike_info| {
            const neuron = &output_neurons[spike_info.index];
            neuron.spike_output = 0;
            neuron.refractory_counter = 0;
            output[spike_info.index] = 0.0;
        }
    }

    /// Update refractory states
    fn updateRefractoryStates(self: *Self) void {
        for (self.neuron_states) |*neuron| {
            if (neuron.refractory_counter > 0) {
                neuron.refractory_counter -= 1;
            }
        }
    }

    /// Reset all neuron states
    pub fn reset(self: *Self) void {
        for (self.neuron_states) |*state| {
            state.membrane_potential = 0.0;
            state.spike_output = 0;
            state.refractory_counter = 0;
            state.last_spike_time = 0;
        }
        self.current_time = 0;
    }

    /// Get kernel statistics
    pub fn getStats(self: *Self) SnnStats {
        var active_neurons: u32 = 0;
        var total_potential: f32 = 0.0;
        var refractory_neurons: u32 = 0;
        
        for (self.neuron_states) |neuron| {
            if (neuron.spike_output == 1) active_neurons += 1;
            if (neuron.refractory_counter > 0) refractory_neurons += 1;
            total_potential += neuron.membrane_potential;
        }
        
        const total_neurons = self.config.input_size + self.config.output_size;
        
        return SnnStats{
            .active_neurons = active_neurons,
            .total_neurons = total_neurons,
            .activation_rate = @as(f32, @floatFromInt(active_neurons)) / @as(f32, @floatFromInt(total_neurons)),
            .avg_potential = total_potential / @as(f32, @floatFromInt(total_neurons)),
            .refractory_neurons = refractory_neurons,
            .current_time = self.current_time,
        };
    }
};

/// SNN statistics structure
pub const SnnStats = extern struct {
    active_neurons: u32,
    total_neurons: u32,
    activation_rate: f32,
    avg_potential: f32,
    refractory_neurons: u32,
    current_time: u32,
};

// FFI exports for Rust integration
export fn snn_kernel_create(config: *const SnnConfig) ?*SnnKernel {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    const kernel = allocator.create(SnnKernel) catch return null;
    kernel.* = SnnKernel.init(allocator, config.*) catch return null;
    
    return kernel;
}

export fn snn_kernel_destroy(kernel: ?*SnnKernel) void {
    if (kernel) |k| {
        k.deinit();
        k.allocator.destroy(k);
    }
}

export fn snn_kernel_forward(kernel: ?*SnnKernel, input: [*]const f32, input_len: u32, output: [*]f32, output_len: u32) bool {
    if (kernel == null) return false;
    
    const k = kernel.?;
    const input_slice = input[0..input_len];
    const output_slice = output[0..output_len];
    
    k.forward(input_slice, output_slice) catch return false;
    return true;
}

export fn snn_kernel_reset(kernel: ?*SnnKernel) void {
    if (kernel) |k| {
        k.reset();
    }
}

export fn snn_kernel_get_stats(kernel: ?*SnnKernel) SnnStats {
    if (kernel) |k| {
        return k.getStats();
    }
    return SnnStats{
        .active_neurons = 0,
        .total_neurons = 0,
        .activation_rate = 0.0,
        .avg_potential = 0.0,
        .refractory_neurons = 0,
        .current_time = 0,
    };
}

// Unit tests
test "SNN kernel creation and basic operations" {
    const testing = std.testing;
    
    const config = SnnConfig{
        .input_size = 10,
        .output_size = 5,
        .threshold = 0.5,
        .decay_rate = 0.9,
        .refractory_period = 2,
        .sparse_rate = 0.2,
    };
    
    var kernel = try SnnKernel.init(testing.allocator, config);
    defer kernel.deinit();
    
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
    try testing.expect(spike_count <= max_spikes + 1); // Allow small variance
}

test "SNN kernel reset functionality" {
    const testing = std.testing;
    
    const config = SnnConfig{
        .input_size = 5,
        .output_size = 3,
        .threshold = 0.3,
        .decay_rate = 0.8,
        .refractory_period = 1,
        .sparse_rate = 0.5,
    };
    
    var kernel = try SnnKernel.init(testing.allocator, config);
    defer kernel.deinit();
    
    // Process some input
    const input = [_]f32{ 0.5, 0.4, 0.6, 0.2, 0.7 };
    var output = [_]f32{0.0} ** 3;
    try kernel.forward(&input, &output);
    
    // Reset kernel
    kernel.reset();
    
    // Verify reset state
    const stats = kernel.getStats();
    try testing.expect(stats.current_time == 0);
    try testing.expect(stats.active_neurons == 0);
}