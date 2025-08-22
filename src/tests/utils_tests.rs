//! Utility system tests
//!
//! Comprehensive tests for performance monitoring, validation utilities,
//! and other support components.

use super::test_utils::*;
use super::*;
use crate::utils::perf::*;
use ndarray::Array1;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(test)]
mod performance_monitor_tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        
        // Should create without error
        let memory_usage = monitor.get_memory_usage_mb();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_latency_tracking() {
        let mut monitor = PerformanceMonitor::new();
        
        // Simulate some work
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let duration = start.elapsed();
        
        monitor.record_inference_latency(duration);
        
        let stats = monitor.get_performance_stats();
        assert!(stats.contains_key("avg_latency_ms"));
        assert!(stats.contains_key("max_latency_ms"));
        assert!(stats.contains_key("min_latency_ms"));
    }

    #[test]
    fn test_energy_monitoring() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record energy usage
        monitor.record_energy_consumption(45.5); // 45.5W
        monitor.record_energy_consumption(42.3);
        monitor.record_energy_consumption(48.1);
        
        let stats = monitor.get_performance_stats();
        assert!(stats.contains_key("avg_power_w"));
        assert!(stats.contains_key("max_power_w"));
        assert!(stats.contains_key("min_power_w"));
        
        let avg_power = stats.get("avg_power_w").copied().unwrap_or(0.0);
        assert!(avg_power > 40.0 && avg_power < 50.0);
    }

    #[test]
    fn test_constraint_violation_detection() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record values that violate constraints
        monitor.record_inference_latency(Duration::from_millis(150)); // > 100ms limit
        monitor.record_energy_consumption(60.0); // > 50W limit
        
        let violations = monitor.get_constraint_violations();
        assert!(!violations.is_empty());
        
        // Should detect latency violation
        assert!(violations.iter().any(|v| v.constraint_type == "latency"));
        // Should detect energy violation
        assert!(violations.iter().any(|v| v.constraint_type == "energy"));
    }

    #[test]
    fn test_memory_tracking() {
        let monitor = PerformanceMonitor::new();
        
        let memory_usage = monitor.get_memory_usage_mb();
        assert!(memory_usage > 0);
        assert!(memory_usage < 32768); // Should be reasonable (< 32GB)
        
        let memory_breakdown = monitor.get_memory_breakdown();
        assert!(memory_breakdown.contains_key("total_mb"));
    }

    #[test]
    fn test_throughput_measurement() {
        let mut monitor = PerformanceMonitor::new();
        
        // Simulate processing samples
        let start = Instant::now();
        for _ in 0..100 {
            // Simulate some processing
            std::thread::sleep(Duration::from_micros(100));
        }
        let duration = start.elapsed();
        
        monitor.record_throughput(100, duration);
        
        let stats = monitor.get_performance_stats();
        assert!(stats.contains_key("throughput_samples_per_sec"));
        
        let throughput = stats.get("throughput_samples_per_sec").copied().unwrap_or(0.0);
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_performance_alerts() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record concerning values
        monitor.record_inference_latency(Duration::from_millis(95)); // Close to limit
        monitor.record_energy_consumption(48.0); // Close to limit
        
        let alerts = monitor.get_performance_alerts();
        
        // Should have alerts for values approaching limits
        assert!(alerts.iter().any(|a| a.alert_type == "warning"));
    }

    #[test]
    fn test_performance_history() {
        let mut monitor = PerformanceMonitor::new();
        
        // Record several measurements
        for i in 0..10 {
            monitor.record_inference_latency(Duration::from_millis(50 + i * 2));
            monitor.record_energy_consumption(30.0 + i as f32);
        }
        
        let history = monitor.get_performance_history();
        assert!(!history.is_empty());
        assert!(history.len() <= 10); // Should track recent history
        
        // History should be in chronological order
        let timestamps: Vec<_> = history.iter().map(|h| h.timestamp).collect();
        let mut sorted_timestamps = timestamps.clone();
        sorted_timestamps.sort();
        assert_eq!(timestamps, sorted_timestamps);
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;

    #[test]
    fn test_latency_constraint() {
        let constraint = PerformanceConstraint {
            constraint_type: "latency".to_string(),
            limit: 100.0,
            unit: "ms".to_string(),
            severity: "critical".to_string(),
        };
        
        assert_eq!(constraint.constraint_type, "latency");
        assert_eq!(constraint.limit, 100.0);
        assert_eq!(constraint.unit, "ms");
        assert_eq!(constraint.severity, "critical");
    }

    #[test]
    fn test_energy_constraint() {
        let constraint = PerformanceConstraint {
            constraint_type: "energy".to_string(),
            limit: 50.0,
            unit: "W".to_string(),
            severity: "critical".to_string(),
        };
        
        assert_eq!(constraint.constraint_type, "energy");
        assert_eq!(constraint.limit, 50.0);
    }

    #[test]
    fn test_memory_constraint() {
        let constraint = PerformanceConstraint {
            constraint_type: "memory".to_string(),
            limit: 8192.0,
            unit: "MB".to_string(),
            severity: "critical".to_string(),
        };
        
        assert_eq!(constraint.constraint_type, "memory");
        assert_eq!(constraint.limit, 8192.0);
    }

    #[test]
    fn test_constraint_violation() {
        let violation = ConstraintViolation {
            constraint_type: "latency".to_string(),
            measured_value: 120.0,
            limit_value: 100.0,
            severity: "critical".to_string(),
            timestamp: std::time::SystemTime::now(),
            description: "Inference latency exceeded 100ms limit".to_string(),
        };
        
        assert_eq!(violation.constraint_type, "latency");
        assert!(violation.measured_value > violation.limit_value);
        assert_eq!(violation.severity, "critical");
        assert!(!violation.description.is_empty());
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics {
            latency_ms: 75.5,
            energy_w: 42.3,
            memory_mb: 6144,
            throughput_samples_per_sec: 150.0,
            cpu_utilization: 0.65,
            gpu_utilization: 0.85,
            timestamp: std::time::SystemTime::now(),
        };
        
        assert!(metrics.latency_ms > 0.0);
        assert!(metrics.energy_w > 0.0);
        assert!(metrics.memory_mb > 0);
        assert!(metrics.throughput_samples_per_sec > 0.0);
        assert!(metrics.cpu_utilization >= 0.0 && metrics.cpu_utilization <= 1.0);
        assert!(metrics.gpu_utilization >= 0.0 && metrics.gpu_utilization <= 1.0);
    }

    #[test]
    fn test_performance_alert() {
        let alert = PerformanceAlert {
            alert_type: "warning".to_string(),
            message: "High energy consumption detected".to_string(),
            measured_value: 47.5,
            threshold: 45.0,
            severity: "medium".to_string(),
            timestamp: std::time::SystemTime::now(),
        };
        
        assert_eq!(alert.alert_type, "warning");
        assert!(!alert.message.is_empty());
        assert!(alert.measured_value > alert.threshold);
        assert_eq!(alert.severity, "medium");
    }

    #[test]
    fn test_system_resources() {
        let resources = SystemResources {
            total_memory_mb: 16384,
            available_memory_mb: 8192,
            cpu_cores: 8,
            gpu_memory_mb: 8192,
            gpu_name: "RTX 2070 Ti".to_string(),
        };
        
        assert!(resources.total_memory_mb > 0);
        assert!(resources.available_memory_mb <= resources.total_memory_mb);
        assert!(resources.cpu_cores > 0);
        assert!(resources.gpu_memory_mb > 0);
        assert!(!resources.gpu_name.is_empty());
    }
}

#[cfg(test)]
mod optimization_utils_tests {
    use super::*;

    #[test]
    fn test_parameter_counting() {
        // Test parameter counting utilities
        let weights = random_weights(100, 50);
        let param_count = weights.len();
        
        assert_eq!(param_count, 5000); // 100 * 50
        assert!(param_count > 0);
    }

    #[test]
    fn test_memory_estimation() {
        // Test memory estimation for parameters
        let param_count = 1_000_000; // 1M parameters
        let bytes_per_param = 4; // f32
        let estimated_mb = (param_count * bytes_per_param) as f32 / (1024.0 * 1024.0);
        
        assert!(estimated_mb > 0.0);
        assert!(estimated_mb < 100.0); // Should be reasonable for 1M params
    }

    #[test]
    fn test_sparsity_calculation() {
        let dense_array = Array1::ones(1000);
        let mut sparse_array = Array1::zeros(1000);
        
        // Set 10% of values to non-zero
        for i in 0..100 {
            sparse_array[i] = 1.0;
        }
        
        let dense_sparsity = calculate_sparsity(&dense_array);
        let sparse_sparsity = calculate_sparsity(&sparse_array);
        
        assert_eq!(dense_sparsity, 0.0); // No zeros
        assert!((sparse_sparsity - 0.9).abs() < 0.01); // ~90% zeros
    }

    fn calculate_sparsity(array: &Array1<f32>) -> f32 {
        let zeros = array.iter().filter(|&&x| x == 0.0).count();
        zeros as f32 / array.len() as f32
    }

    #[test]
    fn test_activation_rate_calculation() {
        let activations = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let activation_rate = calculate_activation_rate(&activations);
        
        assert!((activation_rate - 0.3).abs() < 0.01); // 3/10 = 0.3
    }

    fn calculate_activation_rate(activations: &Array1<f32>) -> f32 {
        let active = activations.iter().filter(|&&x| x > 0.0).count();
        active as f32 / activations.len() as f32
    }
}

#[cfg(test)]
mod validation_utils_tests {
    use super::*;

    #[test]
    fn test_array_validation() {
        let valid_array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let invalid_array = Array1::from_vec(vec![1.0, f32::NAN, 3.0, f32::INFINITY]);
        
        assert!(is_valid_array(&valid_array));
        assert!(!is_valid_array(&invalid_array));
    }

    fn is_valid_array(array: &Array1<f32>) -> bool {
        array.iter().all(|&x| x.is_finite())
    }

    #[test]
    fn test_range_validation() {
        assert!(is_in_range(0.5, 0.0, 1.0));
        assert!(is_in_range(0.0, 0.0, 1.0));
        assert!(is_in_range(1.0, 0.0, 1.0));
        assert!(!is_in_range(-0.1, 0.0, 1.0));
        assert!(!is_in_range(1.1, 0.0, 1.0));
    }

    fn is_in_range(value: f32, min: f32, max: f32) -> bool {
        value >= min && value <= max
    }

    #[test]
    fn test_gradient_validation() {
        let good_gradients = Array1::from_vec(vec![0.1, -0.05, 0.02, -0.08]);
        let bad_gradients = Array1::from_vec(vec![100.0, -200.0, f32::NAN, 0.01]);
        
        assert!(are_gradients_healthy(&good_gradients));
        assert!(!are_gradients_healthy(&bad_gradients));
    }

    fn are_gradients_healthy(gradients: &Array1<f32>) -> bool {
        gradients.iter().all(|&g| g.is_finite() && g.abs() < 10.0)
    }
}

#[cfg(test)]
mod benchmark_utils_tests {
    use super::*;
    use super::perf_test_utils::*;

    #[test]
    fn test_timing_utilities() {
        let (result, duration) = measure_time(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
        assert!(duration < Duration::from_millis(50)); // Should be reasonable
    }

    #[test]
    fn test_memory_benchmarking() {
        let initial_memory = get_memory_usage();
        
        // Allocate some memory
        let _large_array = vec![0u8; 1024 * 1024]; // 1MB
        
        let final_memory = get_memory_usage();
        
        // Memory usage should have increased (or at least not decreased significantly)
        assert!(final_memory >= initial_memory);
    }

    #[test]
    fn test_throughput_calculation() {
        let samples_processed = 1000;
        let duration = Duration::from_secs(2);
        
        let throughput = calculate_throughput(samples_processed, duration);
        assert_eq!(throughput, 500.0); // 1000 samples / 2 seconds = 500 samples/sec
    }

    fn calculate_throughput(samples: usize, duration: Duration) -> f32 {
        samples as f32 / duration.as_secs_f32()
    }

    #[test]
    fn test_efficiency_metrics() {
        let energy_consumed = 100.0; // Joules
        let samples_processed = 1000;
        
        let energy_per_sample = energy_consumed / samples_processed as f32;
        assert_eq!(energy_per_sample, 0.1); // 0.1 J per sample
        
        let efficiency = 1.0 / energy_per_sample; // Samples per Joule
        assert_eq!(efficiency, 10.0);
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::*;

    #[test]
    fn test_performance_config_validation() {
        let config = PerformanceConfig {
            max_latency_ms: 100.0,
            max_energy_w: 50.0,
            max_memory_mb: 8192,
            min_throughput: 10.0,
            enable_monitoring: true,
            alert_thresholds: HashMap::new(),
        };
        
        assert!(config.max_latency_ms > 0.0);
        assert!(config.max_energy_w > 0.0);
        assert!(config.max_memory_mb > 0);
        assert!(config.min_throughput > 0.0);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_monitoring_config() {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("latency_warning".to_string(), 80.0);
        alert_thresholds.insert("energy_warning".to_string(), 40.0);
        
        let config = MonitoringConfig {
            sample_interval_ms: 1000,
            history_size: 100,
            enable_alerts: true,
            alert_thresholds,
        };
        
        assert!(config.sample_interval_ms > 0);
        assert!(config.history_size > 0);
        assert!(config.enable_alerts);
        assert!(!config.alert_thresholds.is_empty());
    }
}

// Additional utility structs for testing
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub max_latency_ms: f32,
    pub max_energy_w: f32,
    pub max_memory_mb: usize,
    pub min_throughput: f32,
    pub enable_monitoring: bool,
    pub alert_thresholds: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub sample_interval_ms: u64,
    pub history_size: usize,
    pub enable_alerts: bool,
    pub alert_thresholds: HashMap<String, f32>,
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_constraint_violation_error() {
        let error = crate::UltraFastAiError::ConstraintViolation {
            constraint_type: "latency".to_string(),
            measured: 120.0,
            limit: 100.0,
        };
        
        match error {
            crate::UltraFastAiError::ConstraintViolation { constraint_type, measured, limit } => {
                assert_eq!(constraint_type, "latency");
                assert_eq!(measured, 120.0);
                assert_eq!(limit, 100.0);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_performance_error() {
        let error = crate::UltraFastAiError::PerformanceError("High latency detected".to_string());
        
        match error {
            crate::UltraFastAiError::PerformanceError(msg) => {
                assert_eq!(msg, "High latency detected");
            }
            _ => panic!("Wrong error type"),
        }
    }
}

#[cfg(test)]
mod integration_utils_tests {
    use super::*;
    use super::integration_setup::*;

    #[tokio::test]
    async fn test_performance_monitoring_integration() {
        init_test_env();
        let test_env = setup_test_environment();
        
        // Test that performance monitoring works with the full system
        let monitor = PerformanceMonitor::new();
        let stats = monitor.get_performance_stats();
        
        assert!(!stats.is_empty());
        assert!(stats.contains_key("memory_usage_mb"));
    }

    #[test]
    fn test_utility_integration_with_model() {
        let snn_config = create_test_snn_config();
        let snn = SnnLayer::new(snn_config.clone()).unwrap();
        
        let param_count = snn.parameter_count();
        let input = random_input(snn_config.input_size);
        
        // Test utilities work with model components
        assert!(param_count >= 0);
        assert_eq!(input.len(), snn_config.input_size);
    }
}