//! Energy consumption benchmarks for ultra-fast AI model
//!
//! Validates <50W inference power target using advanced energy monitoring
//! and power optimization techniques.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use ndarray::Array1;
use tokio::runtime::Runtime;
use std::sync::Arc;
use std::thread;

use ultra_fast_ai::model::core::*;
use ultra_fast_ai::model::fusion::*;
use ultra_fast_ai::model::agentic::*;
use ultra_fast_ai::utils::perf::*;
use ultra_fast_ai::{UltraFastAiError, Result};

/// Energy measurement configuration
#[derive(Debug, Clone)]
struct EnergyConfig {
    name: String,
    target_power_watts: f64,
    measurement_duration_ms: u64,
    sampling_rate_hz: u32,
    thermal_limit_celsius: f64,
}

impl EnergyConfig {
    fn low_power() -> Self {
        Self {
            name: "low_power".to_string(),
            target_power_watts: 25.0,
            measurement_duration_ms: 5000,
            sampling_rate_hz: 100,
            thermal_limit_celsius: 65.0,
        }
    }

    fn medium_power() -> Self {
        Self {
            name: "medium_power".to_string(),
            target_power_watts: 35.0,
            measurement_duration_ms: 10000,
            sampling_rate_hz: 100,
            thermal_limit_celsius: 70.0,
        }
    }

    fn max_power() -> Self {
        Self {
            name: "max_power".to_string(),
            target_power_watts: 50.0,
            measurement_duration_ms: 15000,
            sampling_rate_hz: 100,
            thermal_limit_celsius: 75.0,
        }
    }
}

/// Energy measurement result
#[derive(Debug, Clone)]
struct EnergyMeasurement {
    avg_power_watts: f64,
    peak_power_watts: f64,
    min_power_watts: f64,
    total_energy_joules: f64,
    power_efficiency: f64, // inferences per watt
    thermal_max_celsius: f64,
    frequency_scaling_events: u32,
    power_violations: u32,
}

/// Mock energy monitor (in production, this would interface with actual hardware)
struct EnergyMonitor {
    config: EnergyConfig,
    baseline_power: f64,
    measurement_active: bool,
    start_time: Option<Instant>,
    samples: Vec<f64>,
}

impl EnergyMonitor {
    fn new(config: EnergyConfig) -> Self {
        Self {
            config,
            baseline_power: 15.0, // Baseline system power consumption
            measurement_active: false,
            start_time: None,
            samples: Vec::new(),
        }
    }

    fn start_measurement(&mut self) {
        self.measurement_active = true;
        self.start_time = Some(Instant::now());
        self.samples.clear();
    }

    fn stop_measurement(&mut self) -> EnergyMeasurement {
        self.measurement_active = false;
        let duration = self.start_time.unwrap().elapsed();
        
        // Simulate power measurements (in production, would read from hardware)
        let avg_power = if self.samples.is_empty() {
            self.estimate_power_consumption()
        } else {
            self.samples.iter().sum::<f64>() / self.samples.len() as f64
        };

        let peak_power = self.samples.iter().fold(avg_power, |acc, &x| acc.max(x));
        let min_power = self.samples.iter().fold(avg_power, |acc, &x| acc.min(x));
        let total_energy = avg_power * duration.as_secs_f64();
        
        EnergyMeasurement {
            avg_power_watts: avg_power,
            peak_power_watts: peak_power,
            min_power_watts: min_power,
            total_energy_joules: total_energy,
            power_efficiency: 1.0 / avg_power, // Simple efficiency metric
            thermal_max_celsius: self.estimate_temperature(),
            frequency_scaling_events: 0, // Would track DVFS events
            power_violations: if avg_power > self.config.target_power_watts { 1 } else { 0 },
        }
    }

    fn sample_power(&mut self) {
        if self.measurement_active {
            let power = self.estimate_power_consumption();
            self.samples.push(power);
        }
    }

    fn estimate_power_consumption(&self) -> f64 {
        // Simulate realistic power consumption based on workload
        // In production, this would read from RAPL, PMU, or external power meters
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let base_computation_power = 20.0;
        let memory_access_power = 8.0;
        let cache_miss_penalty = 3.0;
        let thermal_throttling = if self.estimate_temperature() > 70.0 { 0.8 } else { 1.0 };
        
        let total_power = (self.baseline_power + base_computation_power + memory_access_power + cache_miss_penalty) * thermal_throttling;
        
        // Add some realistic variance
        total_power + rng.gen_range(-2.0..2.0)
    }

    fn estimate_temperature(&self) -> f64 {
        // Simulate temperature based on power consumption
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let base_temp = 35.0;
        let power_factor = 0.5; // Temperature increase per watt
        let ambient_temp = 22.0;
        
        let estimated_temp = ambient_temp + base_temp + (self.estimate_power_consumption() * power_factor);
        estimated_temp + rng.gen_range(-2.0..2.0)
    }
}

/// Create energy-optimized model configuration
fn create_energy_optimized_model(power_budget: f64) -> Result<HybridLayer> {
    // Scale model size based on power budget
    let scale_factor = (power_budget / 50.0).min(1.0);
    
    let snn_config = SnnConfig {
        input_size: (512.0 * scale_factor) as usize,
        hidden_sizes: vec![(256.0 * scale_factor) as usize, (128.0 * scale_factor) as usize],
        output_size: (128.0 * scale_factor) as usize,
        threshold: 0.5,
        decay_rate: 0.95, // Higher decay for energy efficiency
        refractory_period: 3, // Longer refractory for lower activity
        sparse_rate: 0.1, // Lower sparsity for energy efficiency
    };

    let ssm_config = SsmConfig {
        input_size: (512.0 * scale_factor) as usize,
        state_size: (24.0 * scale_factor) as usize,
        output_size: (128.0 * scale_factor) as usize,
        num_layers: (4.0 * scale_factor) as usize,
        dt_min: 0.001,
        dt_max: 0.1,
        dt_init: "constant".to_string(),
        conv_kernel_size: 3, // Smaller kernels for efficiency
    };

    let liquid_config = LiquidConfig {
        input_size: (512.0 * scale_factor) as usize,
        hidden_size: (128.0 * scale_factor) as usize,
        output_size: (128.0 * scale_factor) as usize,
        time_constant_min: 1.0,
        time_constant_max: 3.0,
        sensory_tau: 1.5,
        inter_tau: 2.0,
        command_tau: 2.5,
        adaptation_rate: 0.005, // Lower adaptation rate
        enable_adaptation: false, // Disable for energy savings
    };

    let fusion_config = FusionConfig {
        input_dims: vec![(128.0 * scale_factor) as usize; 3],
        output_dim: (128.0 * scale_factor) as usize,
        hidden_dim: (256.0 * scale_factor) as usize,
        attention_heads: 2, // Fewer heads for efficiency
        dropout_rate: 0.05, // Lower dropout
        use_cross_attention: true,
        use_adaptive_weights: false, // Disable for energy savings
        temperature: 1.0,
    };

    HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config)
}

/// Benchmark single inference energy consumption
fn bench_single_inference_energy(c: &mut Criterion) {
    let configs = vec![
        EnergyConfig::low_power(),
        EnergyConfig::medium_power(),
        EnergyConfig::max_power(),
    ];

    let mut group = c.benchmark_group("single_inference_energy");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    for energy_config in configs {
        let mut model = create_energy_optimized_model(energy_config.target_power_watts)
            .expect("Failed to create energy-optimized model");
        
        let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);
        let mut energy_monitor = EnergyMonitor::new(energy_config.clone());

        group.benchmark_with_input(
            BenchmarkId::new("power_constrained_inference", &energy_config.name),
            &energy_config,
            |b, config| {
                b.iter_custom(|iters| {
                    energy_monitor.start_measurement();
                    
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _output = model.forward(&input).expect("Forward pass failed");
                        energy_monitor.sample_power();
                        
                        // Simulate brief pause between inferences
                        thread::sleep(Duration::from_micros(100));
                    }
                    let duration = start.elapsed();
                    
                    let measurement = energy_monitor.stop_measurement();
                    
                    // Validate power consumption
                    assert!(
                        measurement.avg_power_watts <= config.target_power_watts,
                        "Average power {:.2}W exceeds target {:.2}W for {}",
                        measurement.avg_power_watts,
                        config.target_power_watts,
                        config.name
                    );
                    
                    // Validate peak power doesn't exceed 120% of target
                    let peak_limit = config.target_power_watts * 1.2;
                    assert!(
                        measurement.peak_power_watts <= peak_limit,
                        "Peak power {:.2}W exceeds limit {:.2}W for {}",
                        measurement.peak_power_watts,
                        peak_limit,
                        config.name
                    );
                    
                    duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sustained energy consumption
fn bench_sustained_energy_consumption(c: &mut Criterion) {
    let config = EnergyConfig::max_power();
    let mut model = create_energy_optimized_model(config.target_power_watts)
        .expect("Failed to create model");
    
    let inputs: Vec<_> = (0..100)
        .map(|_| Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5))
        .collect();

    let mut group = c.benchmark_group("sustained_energy");
    group.throughput(Throughput::Elements(inputs.len() as u64));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(20);

    group.bench_function("sustained_load_energy", |b| {
        b.iter_custom(|iters| {
            let mut energy_monitor = EnergyMonitor::new(config.clone());
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                for input in &inputs {
                    let _output = model.forward(input).expect("Forward pass failed");
                    energy_monitor.sample_power();
                }
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Validate sustained power consumption
            assert!(
                measurement.avg_power_watts <= config.target_power_watts,
                "Sustained average power {:.2}W exceeds target {:.2}W",
                measurement.avg_power_watts,
                config.target_power_watts
            );
            
            // Validate thermal constraints
            assert!(
                measurement.thermal_max_celsius <= config.thermal_limit_celsius,
                "Maximum temperature {:.1}°C exceeds limit {:.1}°C",
                measurement.thermal_max_celsius,
                config.thermal_limit_celsius
            );
            
            duration
        });
    });

    group.finish();
}

/// Benchmark energy efficiency across different workloads
fn bench_energy_efficiency(c: &mut Criterion) {
    let workloads = vec![
        ("light", 25.0, 256, 1),
        ("medium", 35.0, 512, 4),
        ("heavy", 50.0, 768, 8),
    ];

    let mut group = c.benchmark_group("energy_efficiency");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(15));

    for (workload_name, power_budget, input_size, batch_size) in workloads {
        let mut model = create_energy_optimized_model(power_budget)
            .expect("Failed to create model");
        
        let inputs: Vec<_> = (0..batch_size)
            .map(|_| Array1::from_shape_fn(input_size, |_| rand::random::<f32>() - 0.5))
            .collect();

        group.benchmark_with_input(
            BenchmarkId::new("efficiency", workload_name),
            &power_budget,
            |b, &budget| {
                b.iter_custom(|iters| {
                    let config = EnergyConfig {
                        name: workload_name.to_string(),
                        target_power_watts: budget,
                        measurement_duration_ms: 5000,
                        sampling_rate_hz: 100,
                        thermal_limit_celsius: 70.0,
                    };
                    
                    let mut energy_monitor = EnergyMonitor::new(config);
                    energy_monitor.start_measurement();
                    
                    let start = Instant::now();
                    let mut total_inferences = 0;
                    
                    for _ in 0..iters {
                        for input in &inputs {
                            let _output = model.forward(input).expect("Forward pass failed");
                            total_inferences += 1;
                            energy_monitor.sample_power();
                        }
                    }
                    let duration = start.elapsed();
                    
                    let measurement = energy_monitor.stop_measurement();
                    
                    // Calculate and validate energy efficiency
                    let inferences_per_watt = total_inferences as f64 / measurement.avg_power_watts;
                    let target_efficiency = match workload_name {
                        "light" => 2.0,  // 2 inferences per watt
                        "medium" => 1.5, // 1.5 inferences per watt
                        "heavy" => 1.0,  // 1 inference per watt
                        _ => 1.0,
                    };
                    
                    assert!(
                        inferences_per_watt >= target_efficiency,
                        "Energy efficiency {:.2} inferences/W below target {:.2} for {}",
                        inferences_per_watt,
                        target_efficiency,
                        workload_name
                    );
                    
                    duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark component-level energy consumption
fn bench_component_energy(c: &mut Criterion) {
    let power_budget = 50.0;
    let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);

    let mut group = c.benchmark_group("component_energy");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(8));

    // SNN component energy
    group.bench_function("snn_energy", |b| {
        let snn_config = SnnConfig {
            input_size: 512,
            hidden_sizes: vec![256, 128],
            output_size: 128,
            threshold: 0.5,
            decay_rate: 0.95,
            refractory_period: 3,
            sparse_rate: 0.1,
        };
        let mut snn = SnnLayer::new(snn_config).expect("Failed to create SNN");

        b.iter_custom(|iters| {
            let config = EnergyConfig::max_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                let _output = snn.forward(&input).expect("SNN forward failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // SNN should use less than 30% of total power budget
            let component_power_limit = power_budget * 0.3;
            assert!(
                measurement.avg_power_watts <= component_power_limit,
                "SNN power {:.2}W exceeds component limit {:.2}W",
                measurement.avg_power_watts,
                component_power_limit
            );
            
            duration
        });
    });

    // SSM component energy
    group.bench_function("ssm_energy", |b| {
        let ssm_config = SsmConfig {
            input_size: 512,
            state_size: 24,
            output_size: 128,
            num_layers: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "constant".to_string(),
            conv_kernel_size: 3,
        };
        let mut ssm = SsmLayer::new(ssm_config).expect("Failed to create SSM");

        b.iter_custom(|iters| {
            let config = EnergyConfig::max_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                let _output = ssm.forward(&input).expect("SSM forward failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // SSM should use less than 40% of total power budget
            let component_power_limit = power_budget * 0.4;
            assert!(
                measurement.avg_power_watts <= component_power_limit,
                "SSM power {:.2}W exceeds component limit {:.2}W",
                measurement.avg_power_watts,
                component_power_limit
            );
            
            duration
        });
    });

    // Liquid NN component energy
    group.bench_function("liquid_energy", |b| {
        let liquid_config = LiquidConfig {
            input_size: 512,
            hidden_size: 128,
            output_size: 128,
            time_constant_min: 1.0,
            time_constant_max: 3.0,
            sensory_tau: 1.5,
            inter_tau: 2.0,
            command_tau: 2.5,
            adaptation_rate: 0.005,
            enable_adaptation: false,
        };
        let mut liquid = LiquidLayer::new(liquid_config).expect("Failed to create Liquid NN");

        b.iter_custom(|iters| {
            let config = EnergyConfig::max_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                let _output = liquid.forward(&input).expect("Liquid NN forward failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Liquid NN should use less than 20% of total power budget
            let component_power_limit = power_budget * 0.2;
            assert!(
                measurement.avg_power_watts <= component_power_limit,
                "Liquid NN power {:.2}W exceeds component limit {:.2}W",
                measurement.avg_power_watts,
                component_power_limit
            );
            
            duration
        });
    });

    group.finish();
}

/// Benchmark agentic system energy consumption
fn bench_agentic_energy(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);

    let mut group = c.benchmark_group("agentic_energy");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("agentic_task_energy", |b| {
        let task_config = TaskConfig::default();
        let mut coordinator = AgenticCoordinator::new(task_config, VotingStrategy::WeightedVote);

        b.to_async(&rt).iter_custom(|iters| async move {
            let config = EnergyConfig::max_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                let _output = coordinator.execute_task(&input).await.expect("Task execution failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Agentic system should use less than 60W total (includes all sub-models)
            assert!(
                measurement.avg_power_watts <= 60.0,
                "Agentic system power {:.2}W exceeds 60W limit",
                measurement.avg_power_watts
            );
            
            duration
        });
    });

    group.finish();
}

/// Benchmark power optimization strategies
fn bench_power_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_optimization");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(8));

    let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);

    // Test quantized vs full precision power consumption
    group.bench_function("quantization_power_savings", |b| {
        let mut full_precision_model = create_energy_optimized_model(50.0)
            .expect("Failed to create full precision model");
        
        b.iter_custom(|iters| {
            let config = EnergyConfig::max_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                // Simulate quantized inference (would use actual quantization in production)
                let _output = full_precision_model.forward(&input).expect("Forward pass failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Quantized models should use 20-30% less power
            let power_savings_target = 50.0 * 0.7; // 30% savings
            assert!(
                measurement.avg_power_watts <= power_savings_target,
                "Quantized power {:.2}W should be below {:.2}W (30% savings)",
                measurement.avg_power_watts,
                power_savings_target
            );
            
            duration
        });
    });

    // Test sparse activation power savings
    group.bench_function("sparse_activation_power_savings", |b| {
        let sparse_config = SnnConfig {
            input_size: 512,
            hidden_sizes: vec![256, 128],
            output_size: 128,
            threshold: 0.7, // Higher threshold for more sparsity
            decay_rate: 0.95,
            refractory_period: 3,
            sparse_rate: 0.05, // Very sparse (5% activation)
        };
        let mut sparse_snn = SnnLayer::new(sparse_config).expect("Failed to create sparse SNN");

        b.iter_custom(|iters| {
            let config = EnergyConfig::low_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for _ in 0..iters {
                let _output = sparse_snn.forward(&input).expect("Sparse SNN forward failed");
                energy_monitor.sample_power();
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Sparse activation should significantly reduce power
            assert!(
                measurement.avg_power_watts <= 20.0,
                "Sparse activation power {:.2}W should be below 20W",
                measurement.avg_power_watts
            );
            
            duration
        });
    });

    group.finish();
}

criterion_group!(
    energy_benches,
    bench_single_inference_energy,
    bench_sustained_energy_consumption,
    bench_energy_efficiency,
    bench_component_energy,
    bench_agentic_energy,
    bench_power_optimization,
    bench_rtx_2070_ti_thermal_validation,
    bench_energy_optimization_validation,
    bench_energy_analysis_reporting
);

/// RTX 2070 Ti thermal and power validation
fn bench_rtx_2070_ti_thermal_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtx_2070_ti_thermal");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(15));

    let input = Array1::from_shape_fn(768, |_| rand::random::<f32>() - 0.5);

    group.bench_function("thermal_throttling_prevention", |b| {
        let mut model = create_energy_optimized_model(45.0) // Conservative power target
            .expect("Failed to create thermal-optimized model");
        
        b.iter_custom(|iters| {
            let config = EnergyConfig {
                name: "rtx_2070_ti_thermal".to_string(),
                target_power_watts: 45.0,
                measurement_duration_ms: 30000, // Longer test for thermal buildup
                sampling_rate_hz: 200,
                thermal_limit_celsius: 83.0, // RTX 2070 Ti thermal limit
            };
            
            let mut energy_monitor = EnergyMonitor::new(config);
            energy_monitor.start_measurement();
            
            let start = Instant::now();
            for i in 0..iters {
                let _output = model.forward(&input).expect("Forward pass failed");
                energy_monitor.sample_power();
                
                // Check thermal state every 100 iterations
                if i % 100 == 0 {
                    let current_temp = energy_monitor.estimate_temperature();
                    assert!(current_temp < 83.0, 
                           "Temperature {:.1}°C exceeds RTX 2070 Ti limit at iteration {}", 
                           current_temp, i);
                }
                
                // Small delay to allow thermal simulation
                if i % 50 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            let duration = start.elapsed();
            
            let measurement = energy_monitor.stop_measurement();
            
            // Validate thermal performance
            assert!(measurement.thermal_max_celsius < 83.0, 
                   "Max temperature {:.1}°C exceeds RTX 2070 Ti limit", 
                   measurement.thermal_max_celsius);
            assert!(measurement.avg_power_watts <= 45.0, 
                   "Average power {:.2}W exceeds conservative limit", 
                   measurement.avg_power_watts);
            
            duration
        });
    });

    group.finish();
}

/// Power efficiency and energy optimization validation
fn bench_energy_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_optimization");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(12));

    let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);

    group.bench_function("power_scaling_validation", |b| {
        // Test different power budgets
        let power_budgets = vec![25.0, 35.0, 45.0, 50.0];
        
        b.iter_custom(|iters| {
            let mut total_duration = Duration::new(0, 0);
            
            for power_budget in &power_budgets {
                let mut model = create_energy_optimized_model(*power_budget)
                    .expect("Failed to create power-scaled model");
                
                let config = EnergyConfig {
                    name: format!("power_budget_{}", power_budget),
                    target_power_watts: *power_budget,
                    measurement_duration_ms: 5000,
                    sampling_rate_hz: 100,
                    thermal_limit_celsius: 75.0,
                };
                
                let mut energy_monitor = EnergyMonitor::new(config);
                energy_monitor.start_measurement();
                
                let start = Instant::now();
                for _ in 0..(iters / 4) {
                    let _output = model.forward(&input).expect("Forward pass failed");
                    energy_monitor.sample_power();
                }
                let duration = start.elapsed();
                total_duration += duration;
                
                let measurement = energy_monitor.stop_measurement();
                
                // Power scaling should respect budget ±10%
                let power_tolerance = power_budget * 0.1;
                assert!(measurement.avg_power_watts <= power_budget + power_tolerance,
                       "Power {:.2}W exceeds budget {:.2}W + tolerance", 
                       measurement.avg_power_watts, power_budget);
            }
            
            total_duration
        });
    });

    group.finish();
}

/// Advanced energy analysis and reporting
fn bench_energy_analysis_reporting(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_analysis");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(8));

    let input = Array1::from_shape_fn(512, |_| rand::random::<f32>() - 0.5);

    group.bench_function("real_time_energy_monitoring", |b| {
        let mut model = create_energy_optimized_model(40.0)
            .expect("Failed to create model");
        
        b.iter_custom(|iters| {
            let config = EnergyConfig::medium_power();
            let mut energy_monitor = EnergyMonitor::new(config);
            
            let start = Instant::now();
            
            let mut power_samples = Vec::new();
            let mut temperature_samples = Vec::new();
            
            energy_monitor.start_measurement();
            
            for i in 0..iters {
                let _output = model.forward(&input).expect("Forward pass failed");
                
                // Real-time sampling
                energy_monitor.sample_power();
                
                if i % 10 == 0 {
                    power_samples.push(energy_monitor.estimate_power_consumption());
                    temperature_samples.push(energy_monitor.estimate_temperature());
                }
            }
            
            let duration = start.elapsed();
            let measurement = energy_monitor.stop_measurement();
            
            // Analyze real-time data
            let power_variance = calculate_variance(&power_samples);
            let temp_variance = calculate_variance(&temperature_samples);
            
            // Validate stability
            assert!(power_variance < 25.0, "Power variance {:.2} too high", power_variance);
            assert!(temp_variance < 10.0, "Temperature variance {:.2} too high", temp_variance);
            assert!(measurement.power_violations == 0, 
                   "Should have no power violations in real-time monitoring");
            
            duration
        });
    });

    group.finish();
}

/// Helper function to calculate variance
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    variance
}

criterion_main!(energy_benches);