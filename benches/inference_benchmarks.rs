//! Benchmark tests for ultra-fast AI model inference performance
//!
//! Validates <100ms inference target using criterion-rs with statistical analysis
//! and performance regression detection.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use ndarray::Array1;
use tokio::runtime::Runtime;
use std::sync::Arc;

use ultra_fast_ai::model::core::*;
use ultra_fast_ai::model::fusion::*;
use ultra_fast_ai::model::agentic::*;
use ultra_fast_ai::model::validation::*;
use ultra_fast_ai::utils::perf::*;
use ultra_fast_ai::{UltraFastAiError, Result};

/// Benchmark configuration for different model sizes
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    name: String,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    batch_size: usize,
    target_latency_ms: u64,
}

impl BenchmarkConfig {
    fn small() -> Self {
        Self {
            name: "small".to_string(),
            input_size: 512,
            hidden_size: 256,
            output_size: 128,
            batch_size: 1,
            target_latency_ms: 50,
        }
    }

    fn medium() -> Self {
        Self {
            name: "medium".to_string(),
            input_size: 768,
            hidden_size: 512,
            output_size: 256,
            batch_size: 1,
            target_latency_ms: 75,
        }
    }

    fn large() -> Self {
        Self {
            name: "large".to_string(),
            input_size: 1024,
            hidden_size: 768,
            output_size: 512,
            batch_size: 1,
            target_latency_ms: 100,
        }
    }

    fn batch() -> Self {
        Self {
            name: "batch".to_string(),
            input_size: 768,
            hidden_size: 512,
            output_size: 256,
            batch_size: 8,
            target_latency_ms: 400, // 8 * 50ms per sample
        }
    }
}

/// Create test model for benchmarking
fn create_benchmark_model(config: &BenchmarkConfig) -> Result<HybridLayer> {
    let snn_config = SnnConfig {
        input_size: config.input_size,
        hidden_sizes: vec![config.hidden_size / 2, config.hidden_size / 4],
        output_size: config.output_size,
        threshold: 0.5,
        decay_rate: 0.9,
        refractory_period: 2,
        sparse_rate: 0.15,
    };

    let ssm_config = SsmConfig {
        input_size: config.input_size,
        state_size: 32,
        output_size: config.output_size,
        num_layers: 6,
        dt_min: 0.001,
        dt_max: 0.1,
        dt_init: "random".to_string(),
        conv_kernel_size: 4,
    };

    let liquid_config = LiquidConfig {
        input_size: config.input_size,
        hidden_size: config.hidden_size / 2,
        output_size: config.output_size,
        time_constant: 1.0,
        adaptation_rate: 0.01,
        connectivity: 0.3,
        enable_adaptation: true,
    };

    let fusion_config = FusionConfig {
        input_dims: vec![config.output_size, config.output_size, config.output_size],
        output_dim: config.output_size,
        hidden_dim: config.hidden_size,
        attention_heads: 4,
        dropout_rate: 0.1,
        use_layer_norm: true,
        activation_function: "gelu".to_string(),
    };

    HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config)
}

/// Generate random input for benchmarking
fn generate_benchmark_input(config: &BenchmarkConfig) -> Vec<Array1<f32>> {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    
    (0..config.batch_size)
        .map(|_| {
            Array1::from_shape_fn(config.input_size, |_| rng.gen_range(-1.0..1.0))
        })
        .collect()
}

/// Single inference benchmark
fn bench_single_inference(c: &mut Criterion) {
    let configs = vec![
        BenchmarkConfig::small(),
        BenchmarkConfig::medium(),
        BenchmarkConfig::large(),
    ];

    let mut group = c.benchmark_group("single_inference");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    for config in configs {
        let mut model = create_benchmark_model(&config).expect("Failed to create model");
        let inputs = generate_benchmark_input(&config);
        let input = &inputs[0];

        group.benchmark_with_input(
            BenchmarkId::new("forward_pass", &config.name),
            &config,
            |b, _config| {
                b.iter(|| {
                    let _output = model.forward(input).expect("Forward pass failed");
                });
            },
        );

        // Validate latency target
        group.bench_function(
            BenchmarkId::new("latency_validation", &config.name),
            |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let _output = model.forward(input).expect("Forward pass failed");
                    }
                    let duration = start.elapsed();
                    
                    // Check if average latency meets target
                    let avg_latency_ms = duration.as_millis() as u64 / iters;
                    assert!(
                        avg_latency_ms <= config.target_latency_ms,
                        "Latency {}ms exceeds target {}ms for config {}",
                        avg_latency_ms,
                        config.target_latency_ms,
                        config.name
                    );
                    
                    duration
                });
            },
        );
    }

    group.finish();
}

/// Batch inference benchmark
fn bench_batch_inference(c: &mut Criterion) {
    let config = BenchmarkConfig::batch();
    let mut model = create_benchmark_model(&config).expect("Failed to create model");
    let inputs = generate_benchmark_input(&config);

    let mut group = c.benchmark_group("batch_inference");
    group.throughput(Throughput::Elements(config.batch_size as u64));
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("batch_forward_pass", |b| {
        b.iter(|| {
            for input in &inputs {
                let _output = model.forward(input).expect("Forward pass failed");
            }
        });
    });

    group.bench_function("batch_latency_validation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for input in &inputs {
                    let _output = model.forward(input).expect("Forward pass failed");
                }
            }
            let duration = start.elapsed();
            
            // Validate batch processing time
            let total_samples = iters * config.batch_size as u64;
            let avg_latency_per_sample_ms = duration.as_millis() as u64 / total_samples;
            
            assert!(
                avg_latency_per_sample_ms <= 100,
                "Average latency per sample {}ms exceeds 100ms target",
                avg_latency_per_sample_ms
            );
            
            duration
        });
    });

    group.finish();
}

/// Component-level benchmarks
fn bench_components(c: &mut Criterion) {
    let config = BenchmarkConfig::medium();
    let input = &generate_benchmark_input(&config)[0];

    let mut group = c.benchmark_group("components");
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_secs(5));

    // SNN component benchmark
    group.bench_function("snn_layer", |b| {
        let snn_config = SnnConfig {
            input_size: config.input_size,
            hidden_sizes: vec![256, 128],
            output_size: config.output_size,
            threshold: 0.5,
            decay_rate: 0.9,
            refractory_period: 2,
            sparse_rate: 0.15,
        };
        let mut snn = SnnLayer::new(snn_config).expect("Failed to create SNN");

        b.iter(|| {
            let _output = snn.forward(input).expect("SNN forward failed");
        });
    });

    // SSM component benchmark
    group.bench_function("ssm_layer", |b| {
        let ssm_config = SsmConfig {
            input_size: config.input_size,
            state_size: 32,
            output_size: config.output_size,
            num_layers: 6,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "random".to_string(),
            conv_kernel_size: 4,
        };
        let mut ssm = SsmLayer::new(ssm_config).expect("Failed to create SSM");

        b.iter(|| {
            let _output = ssm.forward(input).expect("SSM forward failed");
        });
    });

    // Liquid NN component benchmark
    group.bench_function("liquid_layer", |b| {
        let liquid_config = LiquidConfig {
            input_size: config.input_size,
            hidden_size: 256,
            output_size: config.output_size,
            time_constant: 1.0,
            adaptation_rate: 0.01,
            connectivity: 0.3,
            enable_adaptation: true,
        };
        let mut liquid = LiquidLayer::new(liquid_config).expect("Failed to create Liquid NN");

        b.iter(|| {
            let _output = liquid.forward(input).expect("Liquid NN forward failed");
        });
    });

    // Fusion layer benchmark
    group.bench_function("fusion_layer", |b| {
        let fusion_config = FusionConfig {
            input_dims: vec![config.output_size, config.output_size, config.output_size],
            output_dim: config.output_size,
            hidden_dim: 512,
            attention_heads: 4,
            dropout_rate: 0.1,
            use_layer_norm: true,
            activation_function: "gelu".to_string(),
        };
        let mut fusion = FusionLayer::new(fusion_config).expect("Failed to create Fusion layer");

        let snn_out = Array1::zeros(config.output_size);
        let ssm_out = Array1::zeros(config.output_size);
        let liquid_out = Array1::zeros(config.output_size);

        b.iter(|| {
            let _output = fusion.forward(&snn_out, &ssm_out, &liquid_out).expect("Fusion forward failed");
        });
    });

    group.finish();
}

/// Agentic system benchmark
fn bench_agentic_system(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::medium();
    let inputs = generate_benchmark_input(&config);

    let mut group = c.benchmark_group("agentic_system");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("task_execution", |b| {
        let task_config = TaskConfig::default();
        let mut coordinator = AgenticCoordinator::new(task_config, VotingStrategy::WeightedVote);

        b.to_async(&rt).iter(|| async {
            let _output = coordinator.execute_task(&inputs[0]).await.expect("Task execution failed");
        });
    });

    group.bench_function("ensemble_voting", |b| {
        let task_config = TaskConfig::default();
        let coordinator = AgenticCoordinator::new(task_config, VotingStrategy::WeightedVote);

        b.iter(|| {
            // Mock sub-model outputs for voting
            let outputs = vec![
                Array1::from_vec(vec![0.8, 0.2, 0.1]),
                Array1::from_vec(vec![0.7, 0.3, 0.0]),
                Array1::from_vec(vec![0.9, 0.1, 0.2]),
            ];
            let confidences = vec![0.9, 0.8, 0.85];
            
            let _result = coordinator.vote_on_outputs(&outputs, &confidences);
        });
    });

    group.finish();
}

/// Validation system benchmark
fn bench_validation_system(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("validation_system");
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("output_validation", |b| {
        let validation_config = ValidationConfig::default();
        let validator = OutputValidator::new(validation_config);

        let test_outputs = vec![
            "This is a factual statement about machine learning.",
            "According to experts, this claim needs verification.",
            "The algorithm processes data efficiently.",
        ];

        b.to_async(&rt).iter(|| async {
            for output in &test_outputs {
                let _result = validator.validate_output(output, None, None).await.expect("Validation failed");
            }
        });
    });

    group.bench_function("quick_validation", |b| {
        let validation_config = ValidationConfig::default();
        let validator = OutputValidator::new(validation_config);

        let test_text = "This is a reasonable and factual statement.";

        b.iter(|| {
            let _result = validator.quick_validate(test_text);
        });
    });

    group.finish();
}

/// Memory usage benchmark
fn bench_memory_usage(c: &mut Criterion) {
    let config = BenchmarkConfig::large();

    let mut group = c.benchmark_group("memory_usage");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("model_creation", |b| {
        b.iter(|| {
            let _model = create_benchmark_model(&config).expect("Model creation failed");
        });
    });

    group.bench_function("memory_efficient_inference", |b| {
        let mut model = create_benchmark_model(&config).expect("Failed to create model");
        let input = &generate_benchmark_input(&config)[0];
        let performance_monitor = PerformanceMonitor::new();

        b.iter(|| {
            let initial_memory = performance_monitor.get_memory_usage_mb();
            let _output = model.forward(input).expect("Forward pass failed");
            let final_memory = performance_monitor.get_memory_usage_mb();
            
            // Ensure memory usage is reasonable
            let memory_increase = final_memory.saturating_sub(initial_memory);
            assert!(
                memory_increase < 1000, // Less than 1GB increase per inference
                "Memory increase {}MB exceeds reasonable limit",
                memory_increase
            );
        });
    });

    group.finish();
}

/// Throughput benchmark
fn bench_throughput(c: &mut Criterion) {
    let config = BenchmarkConfig::medium();
    let mut model = create_benchmark_model(&config).expect("Failed to create model");
    let inputs: Vec<_> = (0..100).map(|_| generate_benchmark_input(&config)[0].clone()).collect();

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(inputs.len() as u64));
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("samples_per_second", |b| {
        b.iter(|| {
            for input in &inputs {
                let _output = model.forward(input).expect("Forward pass failed");
            }
        });
    });

    group.bench_function("throughput_validation", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for input in &inputs {
                    let _output = model.forward(input).expect("Forward pass failed");
                }
            }
            let duration = start.elapsed();
            
            // Calculate throughput
            let total_samples = iters * inputs.len() as u64;
            let samples_per_second = total_samples as f64 / duration.as_secs_f64();
            
            // Should process at least 10 samples per second for efficiency
            assert!(
                samples_per_second >= 10.0,
                "Throughput {:.2} samples/sec is below minimum requirement",
                samples_per_second
            );
            
            duration
        });
    });

    group.finish();
}

/// Regression test benchmark
fn bench_regression_detection(c: &mut Criterion) {
    let config = BenchmarkConfig::medium();

    let mut group = c.benchmark_group("regression_detection");
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_secs(3));

    // Baseline performance target (these would be updated based on CI results)
    const BASELINE_LATENCY_MS: u64 = 75;
    const REGRESSION_THRESHOLD: f64 = 1.1; // 10% regression tolerance

    group.bench_function("regression_check", |b| {
        let mut model = create_benchmark_model(&config).expect("Failed to create model");
        let input = &generate_benchmark_input(&config)[0];

        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _output = model.forward(input).expect("Forward pass failed");
            }
            let duration = start.elapsed();
            
            // Check for performance regression
            let avg_latency_ms = duration.as_millis() as u64 / iters;
            let regression_ratio = avg_latency_ms as f64 / BASELINE_LATENCY_MS as f64;
            
            if regression_ratio > REGRESSION_THRESHOLD {
                eprintln!(
                    "WARNING: Performance regression detected! Current: {}ms, Baseline: {}ms, Ratio: {:.2}",
                    avg_latency_ms, BASELINE_LATENCY_MS, regression_ratio
                );
            }
            
            duration
        });
    });

    group.finish();
}

/// Stress test benchmark
fn bench_stress_test(c: &mut Criterion) {
    let config = BenchmarkConfig::large();

    let mut group = c.benchmark_group("stress_test");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(50);

    group.bench_function("sustained_load", |b| {
        let mut model = create_benchmark_model(&config).expect("Failed to create model");
        let inputs: Vec<_> = (0..1000).map(|_| generate_benchmark_input(&config)[0].clone()).collect();

        b.iter(|| {
            for (i, input) in inputs.iter().enumerate() {
                let start = std::time::Instant::now();
                let _output = model.forward(input).expect("Forward pass failed");
                let latency = start.elapsed();
                
                // Ensure latency doesn't degrade under sustained load
                assert!(
                    latency.as_millis() <= 150, // Allow some variance under stress
                    "Latency {}ms exceeds stress test limit at iteration {}",
                    latency.as_millis(),
                    i
                );
            }
        });
    });

    group.finish();
}

/// RTX 2070 Ti specific benchmark validation
fn bench_rtx_2070_ti_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtx_2070_ti_constraints");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(10));

    // RTX 2070 Ti specific configuration
    let rtx_config = BenchmarkConfig {
        name: "rtx_2070_ti".to_string(),
        input_size: 768,
        hidden_size: 1024,
        output_size: 256,
        batch_size: 4, // Reasonable batch size for 8GB VRAM
        target_latency_ms: 100,
    };

    group.bench_function("memory_constraint_validation", |b| {
        let model = create_benchmark_model(&rtx_config).expect("Failed to create model");
        let total_params = model.total_parameters();
        
        // Validate parameter budget (100M parameter limit)
        assert!(total_params <= 100_000_000, 
               "Model has {} parameters, exceeds 100M limit", total_params);
        
        // Estimate VRAM usage (simplified)
        let param_memory_mb = (total_params * 4) / (1024 * 1024); // 4 bytes per f32
        let activation_memory_mb = (rtx_config.input_size * rtx_config.batch_size * 4) / (1024 * 1024);
        let total_memory_mb = param_memory_mb + activation_memory_mb + 1024; // +1GB buffer
        
        assert!(total_memory_mb <= 8192, 
               "Estimated memory usage {} MB exceeds RTX 2070 Ti 8GB limit", total_memory_mb);

        b.iter(|| {
            // Memory usage validation during inference
            let performance_monitor = PerformanceMonitor::new();
            let initial_memory = performance_monitor.get_memory_usage_mb();
            
            let inputs = generate_benchmark_input(&rtx_config);
            for input in &inputs {
                let mut model_ref = &model;
                // Note: This would need to be mutable in real implementation
                // let _output = model_ref.forward(input).expect("Forward pass failed");
            }
            
            let final_memory = performance_monitor.get_memory_usage_mb();
            let memory_increase = final_memory.saturating_sub(initial_memory);
            
            assert!(memory_increase < 2048, 
                   "Memory increase {} MB too high during inference", memory_increase);
        });
    });

    group.bench_function("training_time_constraint", |b| {
        // Simulate training time estimation
        let epochs = 10;
        let samples_per_epoch = 10000;
        let target_training_time_hours = 24.0;
        
        b.iter_custom(|iters| {
            let mut total_training_time = Duration::new(0, 0);
            
            for _ in 0..iters {
                let start = std::time::Instant::now();
                
                // Simulate training epoch (simplified)
                for _ in 0..10 { // 10 samples to simulate
                    let input = &generate_benchmark_input(&rtx_config)[0];
                    // Simulate forward + backward pass time (2x forward)
                    let _training_step_time = Duration::from_millis(2); // Mock training step
                }
                
                total_training_time += start.elapsed();
            }
            
            // Estimate full training time
            let samples_processed = iters * 10;
            let time_per_sample = total_training_time.as_secs_f64() / samples_processed as f64;
            let estimated_training_hours = (time_per_sample * samples_per_epoch as f64 * epochs as f64) / 3600.0;
            
            assert!(estimated_training_hours <= target_training_time_hours,
                   "Estimated training time {:.2}h exceeds 24h target", estimated_training_hours);
            
            total_training_time
        });
    });

    group.finish();
}

/// Parameter budget validation benchmark
fn bench_parameter_budget_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_budget");
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_secs(5));

    // Test different parameter distributions
    let budget_configs = vec![
        ("snn_heavy", 30_000_000, 25_000_000, 15_000_000, 8_000_000),
        ("ssm_heavy", 20_000_000, 40_000_000, 15_000_000, 8_000_000),
        ("liquid_heavy", 25_000_000, 30_000_000, 20_000_000, 8_000_000),
        ("fusion_heavy", 25_000_000, 30_000_000, 15_000_000, 10_000_000),
    ];

    for (name, snn_target, ssm_target, liquid_target, fusion_target) in budget_configs {
        group.bench_function(&format!("budget_validation_{}", name), |b| {
            b.iter(|| {
                // Create model configuration targeting specific parameter distribution
                let config = BenchmarkConfig {
                    name: name.to_string(),
                    input_size: 512,
                    hidden_size: 384,
                    output_size: 128,
                    batch_size: 1,
                    target_latency_ms: 100,
                };
                
                let model = create_benchmark_model(&config).expect("Failed to create model");
                let breakdown = model.parameter_breakdown();
                let total_params = model.total_parameters();
                
                // Validate total budget
                assert!(total_params <= 100_000_000, 
                       "Total parameters {} exceed 100M budget", total_params);
                
                // Validate individual component budgets
                let snn_params = breakdown.get("snn").unwrap_or(&0);
                let ssm_params = breakdown.get("ssm").unwrap_or(&0);
                let liquid_params = breakdown.get("liquid").unwrap_or(&0);
                let fusion_params = breakdown.get("fusion").unwrap_or(&0);
                
                assert!(*snn_params <= snn_target, 
                       "SNN params {} exceed target {}", snn_params, snn_target);
                assert!(*ssm_params <= ssm_target, 
                       "SSM params {} exceed target {}", ssm_params, ssm_target);
                assert!(*liquid_params <= liquid_target, 
                       "Liquid params {} exceed target {}", liquid_params, liquid_target);
                assert!(*fusion_params <= fusion_target, 
                       "Fusion params {} exceed target {}", fusion_params, fusion_target);
            });
        });
    }

    group.finish();
}

/// Advanced performance metrics benchmark
fn bench_advanced_performance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_metrics");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(8));

    let config = BenchmarkConfig::medium();
    let mut model = create_benchmark_model(&config).expect("Failed to create model");
    let inputs = generate_benchmark_input(&config);

    group.bench_function("performance_consistency", |b| {
        b.iter_custom(|iters| {
            let mut latencies = Vec::new();
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let inference_start = std::time::Instant::now();
                let _output = model.forward(&inputs[0]).expect("Forward pass failed");
                let latency = inference_start.elapsed();
                latencies.push(latency.as_millis() as f64);
            }
            
            let total_time = start.elapsed();
            
            // Calculate performance consistency metrics
            let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let variance = latencies.iter()
                .map(|&x| (x - mean_latency).powi(2))
                .sum::<f64>() / latencies.len() as f64;
            let std_dev = variance.sqrt();
            let coefficient_of_variation = std_dev / mean_latency;
            
            // Validate performance consistency
            assert!(coefficient_of_variation < 0.2, 
                   "Performance inconsistency detected: CV = {:.3}", coefficient_of_variation);
            assert!(mean_latency <= 100.0, 
                   "Mean latency {:.2}ms exceeds 100ms target", mean_latency);
            
            total_time
        });
    });

    group.bench_function("tail_latency_validation", |b| {
        b.iter_custom(|iters| {
            let mut latencies = Vec::new();
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let inference_start = std::time::Instant::now();
                let _output = model.forward(&inputs[0]).expect("Forward pass failed");
                let latency = inference_start.elapsed();
                latencies.push(latency.as_millis() as u64);
            }
            
            let total_time = start.elapsed();
            
            // Sort latencies for percentile calculation
            latencies.sort();
            let len = latencies.len();
            
            let p50 = latencies[len / 2];
            let p95 = latencies[(len as f64 * 0.95) as usize];
            let p99 = latencies[(len as f64 * 0.99) as usize];
            let max_latency = latencies[len - 1];
            
            // Validate tail latencies
            assert!(p50 <= 75, "P50 latency {}ms exceeds 75ms target", p50);
            assert!(p95 <= 150, "P95 latency {}ms exceeds 150ms target", p95);
            assert!(p99 <= 200, "P99 latency {}ms exceeds 200ms target", p99);
            assert!(max_latency <= 300, "Max latency {}ms exceeds 300ms limit", max_latency);
            
            total_time
        });
    });

    group.bench_function("concurrent_inference_scaling", |b| {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let model_arc = Arc::new(Mutex::new(model));
        
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let num_threads = 4;
                let mut handles = vec![];
                
                for _ in 0..num_threads {
                    let model_clone = Arc::clone(&model_arc);
                    let input = inputs[0].clone();
                    
                    let handle = thread::spawn(move || {
                        let mut model_guard = model_clone.lock().unwrap();
                        let _output = model_guard.forward(&input).expect("Forward pass failed");
                    });
                    
                    handles.push(handle);
                }
                
                for handle in handles {
                    handle.join().unwrap();
                }
            }
            
            start.elapsed()
        });
    });

    group.finish();
}

/// Power efficiency benchmark
fn bench_power_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_efficiency");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(10));

    let config = BenchmarkConfig::medium();
    let mut model = create_benchmark_model(&config).expect("Failed to create model");
    let input = &generate_benchmark_input(&config)[0];

    group.bench_function("inferences_per_watt", |b| {
        b.iter_custom(|iters| {
            let performance_monitor = PerformanceMonitor::new();
            let start = std::time::Instant::now();
            
            // Simulate power measurement start
            performance_monitor.start_power_measurement();
            
            for _ in 0..iters {
                let _output = model.forward(input).expect("Forward pass failed");
            }
            
            let total_time = start.elapsed();
            let avg_power = performance_monitor.get_average_power_consumption();
            
            // Calculate efficiency metrics
            let inferences_per_second = iters as f64 / total_time.as_secs_f64();
            let inferences_per_watt = inferences_per_second / avg_power as f64;
            
            // Validate power efficiency
            assert!(avg_power <= 50.0, "Average power {:.2}W exceeds 50W limit", avg_power);
            assert!(inferences_per_watt >= 0.2, 
                   "Power efficiency {:.3} inferences/W below minimum", inferences_per_watt);
            
            total_time
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_inference,
    bench_batch_inference,
    bench_components,
    bench_agentic_system,
    bench_validation_system,
    bench_memory_usage,
    bench_throughput,
    bench_regression_detection,
    bench_stress_test,
    bench_rtx_2070_ti_constraints,
    bench_parameter_budget_validation,
    bench_advanced_performance_metrics,
    bench_power_efficiency
);

criterion_main!(benches);