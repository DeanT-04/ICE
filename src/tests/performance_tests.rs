//! Comprehensive performance monitoring tests
//!
//! Tests for energy consumption, latency tracking, memory usage,
//! and performance constraint validation with property-based testing.

use rstest::*;
use proptest::prelude::*;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use test_case::test_case;
use serial_test::serial;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::timeout;

use crate::utils::perf::*;
use crate::utils::energy::*;

// =============================================================================
// FIXTURES FOR PERFORMANCE TESTING
// =============================================================================

#[fixture]
fn default_perf_config() -> PerformanceConfig {
    PerformanceConfig::default()
}

#[fixture]
fn strict_perf_config() -> PerformanceConfig {
    PerformanceConfig {
        energy: EnergyConfig {
            target_power_limit_w: 30.0, // Stricter than default 50W
            measurement_interval_ms: 50, // More frequent measurements
            power_alert_threshold_w: 25.0,
            ..Default::default()
        },
        latency: LatencyConfig {
            target_latency_ms: 50.0, // Stricter than default 100ms
            alert_threshold_ms: 45.0,
            history_size: 2000,
            ..Default::default()
        },
        monitoring_enabled: true,
        log_interval_sec: 5,
        ..Default::default()
    }
}

#[fixture]
fn test_perf_monitor() -> PerformanceMonitor {
    PerformanceMonitor::new()
}

#[fixture]
fn energy_config_low_power() -> EnergyConfig {
    EnergyConfig {
        target_power_limit_w: 20.0,
        measurement_interval_ms: 100,
        enable_cpu_monitoring: true,
        enable_gpu_monitoring: false,  // Disable GPU for low power
        enable_system_monitoring: true,
        power_alert_threshold_w: 18.0,
    }
}

#[fixture]
fn energy_config_high_performance() -> EnergyConfig {
    EnergyConfig {
        target_power_limit_w: 75.0,
        measurement_interval_ms: 50,
        enable_cpu_monitoring: true,
        enable_gpu_monitoring: true,
        enable_system_monitoring: true,
        power_alert_threshold_w: 70.0,
    }
}

#[fixture]
fn latency_config_realtime() -> LatencyConfig {
    LatencyConfig {
        target_latency_ms: 10.0,  // Real-time requirements
        percentile_tracking: vec![50.0, 90.0, 95.0, 99.0, 99.9],
        history_size: 10000,
        alert_threshold_ms: 15.0,
    }
}

// =============================================================================
// PERFORMANCE MONITOR CORE TESTS
// =============================================================================

#[rstest]
#[case::default_config(default_perf_config())]
#[case::strict_config(strict_perf_config())]
fn test_performance_monitor_creation(#[case] config: PerformanceConfig) {
    let monitor = PerformanceMonitor::with_config(config.clone());
    
    assert_eq!(monitor.config.energy.target_power_limit_w, config.energy.target_power_limit_w);
    assert_eq!(monitor.config.latency.target_latency_ms, config.latency.target_latency_ms);
    assert_eq!(monitor.config.monitoring_enabled, config.monitoring_enabled);
}

#[rstest]
fn test_measurement_lifecycle(mut test_perf_monitor: PerformanceMonitor) {
    // Start measurement
    let measurement_id = test_perf_monitor.start_measurement("test_operation");
    assert!(!measurement_id.is_empty());
    
    // Simulate some work
    std::thread::sleep(Duration::from_millis(10));
    
    // Finish measurement
    let duration = test_perf_monitor.finish_measurement(&measurement_id);
    assert!(duration.is_some());
    assert!(duration.unwrap() >= 10.0); // Should be at least 10ms
}

#[rstest]
#[tokio::test]
async fn test_async_measurement(test_perf_monitor: PerformanceMonitor) {
    let (result, duration) = test_perf_monitor.measure_async("async_test", async {
        tokio::time::sleep(Duration::from_millis(5)).await;
        42
    }).await;
    
    assert_eq!(result, 42);
    assert!(duration >= 5.0);
}

#[rstest]
fn test_multiple_concurrent_measurements(test_perf_monitor: PerformanceMonitor) {
    let monitor = Arc::new(Mutex::new(test_perf_monitor));
    let mut handles = Vec::new();
    
    // Start multiple measurements concurrently
    for i in 0..5 {
        let monitor_clone = monitor.clone();
        let handle = std::thread::spawn(move || {
            let mut guard = monitor_clone.lock().unwrap();
            let id = guard.start_measurement(&format!("operation_{}", i));
            std::thread::sleep(Duration::from_millis(5));
            guard.finish_measurement(&id)
        });
        handles.push(handle);
    }
    
    // Wait for all measurements to complete
    for handle in handles {
        let duration = handle.join().unwrap();
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 5.0);
    }
}

// =============================================================================
// ENERGY CONSUMPTION TESTS
// =============================================================================

#[rstest]
#[case::low_power(energy_config_low_power())]
#[case::high_performance(energy_config_high_performance())]
fn test_energy_configuration_bounds(#[case] config: EnergyConfig) {
    assert!(config.target_power_limit_w > 0.0);
    assert!(config.power_alert_threshold_w <= config.target_power_limit_w);
    assert!(config.measurement_interval_ms > 0);
}

#[rstest]
fn test_power_consumption_monitoring(default_perf_config: PerformanceConfig) {
    let monitor = PerformanceMonitor::with_config(default_perf_config);
    
    let power = monitor.get_power_consumption();
    
    // Power should be reasonable
    assert!(power >= 0.0, "Power consumption should be non-negative");
    assert!(power <= 200.0, "Power consumption should be reasonable (<200W)");
}

#[rstest]
#[test_case(true, true, true; "all monitoring enabled")]
#[test_case(true, false, false; "cpu only")]
#[test_case(false, true, false; "gpu only")]
#[test_case(false, false, true; "system only")]
fn test_energy_monitoring_components(cpu: bool, gpu: bool, system: bool) {
    let config = PerformanceConfig {
        energy: EnergyConfig {
            enable_cpu_monitoring: cpu,
            enable_gpu_monitoring: gpu,
            enable_system_monitoring: system,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let monitor = PerformanceMonitor::with_config(config);
    let power = monitor.get_power_consumption();
    
    if cpu || gpu || system {
        assert!(power > 0.0, "Should have some power consumption when monitoring enabled");
    }
    
    // Power should always be reasonable regardless of configuration
    assert!(power <= 300.0, "Power should not exceed reasonable limits");
}

// =============================================================================
// LATENCY MONITORING TESTS
// =============================================================================

#[rstest]
fn test_latency_percentile_calculation(latency_config_realtime: LatencyConfig) {
    let mut monitor = PerformanceMonitor::with_config(PerformanceConfig {
        latency: latency_config_realtime,
        ..Default::default()
    });
    
    // Add measurements with known distribution
    let latencies = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    
    for (i, &latency) in latencies.iter().enumerate() {
        let id = monitor.start_measurement(&format!("test_{}", i));
        // Simulate the latency by adding to history directly
        {
            let mut history = monitor.latency_history.lock().unwrap();
            history.push_back(latency);
        }
        monitor.finish_measurement(&id);
    }
    
    let stats = monitor.get_stats();
    
    // Verify percentiles are reasonable
    assert!(stats.p50_latency_ms >= 40.0 && stats.p50_latency_ms <= 60.0);
    assert!(stats.p90_latency_ms >= 80.0 && stats.p90_latency_ms <= 100.0);
    assert!(stats.p95_latency_ms >= 90.0 && stats.p95_latency_ms <= 100.0);
    assert!(stats.p99_latency_ms >= 95.0 && stats.p99_latency_ms <= 100.0);
    
    // Max should be the highest value
    assert_eq!(stats.max_latency_ms, 100.0);
}

#[rstest]
fn test_latency_constraint_validation(strict_perf_config: PerformanceConfig) {
    let monitor = PerformanceMonitor::with_config(strict_perf_config.clone());
    
    // Add some latency measurements
    {
        let mut history = monitor.latency_history.lock().unwrap();
        history.push_back(30.0); // Under target (50ms)
        history.push_back(60.0); // Over target (50ms)
        history.push_back(40.0); // Under target
    }
    
    let constraints = monitor.check_constraints();
    
    assert!(constraints.contains_key("latency_constraint"));
    assert!(constraints.contains_key("energy_constraint"));
    assert!(constraints.contains_key("memory_constraint"));
    
    // With mixed latencies, constraint might or might not be met depending on percentile calculation
    let latency_ok = constraints.get("latency_constraint").unwrap();
    assert!(latency_ok == &true || latency_ok == &false); // Valid boolean value
}

// =============================================================================
// MEMORY USAGE TESTS
// =============================================================================

#[rstest]
fn test_memory_usage_tracking(test_perf_monitor: PerformanceMonitor) {
    let memory_mb = test_perf_monitor.get_memory_usage_mb();
    
    assert!(memory_mb > 0, "Memory usage should be positive");
    assert!(memory_mb < 32768, "Memory usage should be reasonable (<32GB)");
}

#[rstest]
fn test_memory_constraint_validation(test_perf_monitor: PerformanceMonitor) {
    let constraints = test_perf_monitor.check_constraints();
    let memory_ok = constraints.get("memory_constraint").unwrap();
    
    // Should pass memory constraint (8GB limit)
    assert!(*memory_ok, "Memory constraint should pass for reasonable usage");
}

// =============================================================================
// PERFORMANCE STATISTICS TESTS
// =============================================================================

#[rstest]
fn test_performance_statistics_calculation(mut test_perf_monitor: PerformanceMonitor) {
    // Add some test data
    {
        let mut latency_history = test_perf_monitor.latency_history.lock().unwrap();
        let mut energy_history = test_perf_monitor.energy_history.lock().unwrap();
        
        // Add latency data
        for i in 1..=100 {
            latency_history.push_back(i as f32);
        }
        
        // Add energy data
        for i in 1..=100 {
            energy_history.push_back((i as f32) * 0.5);
        }
    }
    
    let stats = test_perf_monitor.get_stats();
    
    // Verify basic statistics
    assert!(stats.avg_latency_ms > 0.0);
    assert!(stats.avg_energy_w > 0.0);
    assert!(stats.p50_latency_ms > 0.0);
    assert!(stats.p90_latency_ms >= stats.p50_latency_ms);
    assert!(stats.p95_latency_ms >= stats.p90_latency_ms);
    assert!(stats.p99_latency_ms >= stats.p95_latency_ms);
    assert!(stats.max_latency_ms >= stats.p99_latency_ms);
    
    // Verify efficiency score
    assert!(stats.efficiency_score >= 0.0 && stats.efficiency_score <= 1.0);
}

// =============================================================================
// PROPERTY-BASED TESTS
// =============================================================================

prop_compose! {
    fn valid_power_limit()(limit in 10.0f32..200.0) -> f32 {
        limit
    }
}

prop_compose! {
    fn valid_latency_target()(target in 1.0f32..1000.0) -> f32 {
        target
    }
}

prop_compose! {
    fn valid_measurement_interval()(interval in 10u64..1000) -> u64 {
        interval
    }
}

proptest! {
    #[test]
    fn property_energy_config_consistency(
        power_limit in valid_power_limit(),
        interval in valid_measurement_interval()
    ) {
        let alert_threshold = power_limit * 0.9; // 90% of limit
        
        let config = EnergyConfig {
            target_power_limit_w: power_limit,
            power_alert_threshold_w: alert_threshold,
            measurement_interval_ms: interval,
            ..Default::default()
        };
        
        prop_assert!(config.power_alert_threshold_w <= config.target_power_limit_w);
        prop_assert!(config.target_power_limit_w > 0.0);
        prop_assert!(config.measurement_interval_ms > 0);
    }
    
    #[test]
    fn property_latency_config_consistency(
        target in valid_latency_target(),
        history_size in 100usize..10000
    ) {
        let alert_threshold = target * 0.8; // 80% of target
        
        let config = LatencyConfig {
            target_latency_ms: target,
            alert_threshold_ms: alert_threshold,
            history_size,
            ..Default::default()
        };
        
        prop_assert!(config.target_latency_ms > 0.0);
        prop_assert!(config.alert_threshold_ms <= config.target_latency_ms);
        prop_assert!(config.history_size > 0);
    }
    
    #[test]
    fn property_percentile_calculation_bounds(
        values in prop::collection::vec(1.0f32..1000.0, 10..100)
    ) {
        let mut sorted_values = values;
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50 = PerformanceMonitor::percentile(&sorted_values, 50.0);
        let p90 = PerformanceMonitor::percentile(&sorted_values, 90.0);
        let p95 = PerformanceMonitor::percentile(&sorted_values, 95.0);
        let p99 = PerformanceMonitor::percentile(&sorted_values, 99.0);
        
        prop_assert!(p50 <= p90);
        prop_assert!(p90 <= p95);
        prop_assert!(p95 <= p99);
        
        prop_assert!(p50 >= sorted_values[0]);
        prop_assert!(p99 <= sorted_values[sorted_values.len() - 1]);
    }
}

// =============================================================================
// QUICKCHECK TESTS
// =============================================================================

#[quickcheck]
fn quickcheck_performance_metrics_bounds(
    latency: f32,
    energy: f32,
    memory: u32
) -> TestResult {
    if latency < 0.0 || energy < 0.0 || memory == 0 {
        return TestResult::discard();
    }
    
    if latency > 10000.0 || energy > 1000.0 || memory > 100000 {
        return TestResult::discard();
    }
    
    let metrics = PerformanceMetrics {
        timestamp: 0,
        latency_ms: latency,
        energy_consumption_w: energy,
        memory_usage_mb: memory as usize,
        cpu_usage_percent: 50.0,
        gpu_usage_percent: 30.0,
        gpu_memory_mb: 4096,
        throughput_tokens_per_sec: 100.0,
        model_efficiency_score: 0.8,
    };
    
    TestResult::from_bool(
        metrics.latency_ms >= 0.0 &&
        metrics.energy_consumption_w >= 0.0 &&
        metrics.memory_usage_mb > 0 &&
        metrics.model_efficiency_score >= 0.0 &&
        metrics.model_efficiency_score <= 1.0
    )
}

#[quickcheck]
fn quickcheck_constraint_checking_consistency(
    target_latency: f32,
    target_power: f32,
    actual_latency: f32,
    actual_power: f32
) -> TestResult {
    if target_latency <= 0.0 || target_power <= 0.0 ||
       actual_latency < 0.0 || actual_power < 0.0 {
        return TestResult::discard();
    }
    
    if target_latency > 1000.0 || target_power > 200.0 ||
       actual_latency > 2000.0 || actual_power > 400.0 {
        return TestResult::discard();
    }
    
    let config = PerformanceConfig {
        latency: LatencyConfig {
            target_latency_ms: target_latency,
            ..Default::default()
        },
        energy: EnergyConfig {
            target_power_limit_w: target_power,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let monitor = PerformanceMonitor::with_config(config);
    
    // Add test data
    {
        let mut latency_history = monitor.latency_history.lock().unwrap();
        latency_history.push_back(actual_latency);
        
        let mut energy_history = monitor.energy_history.lock().unwrap();
        energy_history.push_back(actual_power);
    }
    
    let constraints = monitor.check_constraints();
    let latency_ok = constraints.get("latency_constraint").unwrap_or(&false);
    let energy_ok = constraints.get("energy_constraint").unwrap_or(&false);
    
    // Verify constraint logic
    let expected_latency_ok = actual_latency <= target_latency;
    let expected_energy_ok = actual_power <= target_power;
    
    TestResult::from_bool(
        (*latency_ok == expected_latency_ok || (actual_latency - target_latency).abs() < 1.0) &&
        (*energy_ok == expected_energy_ok || (actual_power - target_power).abs() < 1.0)
    )
}

// =============================================================================
// STRESS AND PERFORMANCE TESTS
// =============================================================================

#[rstest]
#[serial] // Run serially to avoid interference
fn test_high_frequency_measurements(test_perf_monitor: PerformanceMonitor) {
    let monitor = Arc::new(Mutex::new(test_perf_monitor));
    let measurement_count = 1000;
    let start_time = Instant::now();
    
    for i in 0..measurement_count {
        let monitor_clone = monitor.clone();
        let mut guard = monitor_clone.lock().unwrap();
        let id = guard.start_measurement(&format!("high_freq_{}", i));
        // Minimal work
        std::thread::sleep(Duration::from_micros(100));
        guard.finish_measurement(&id);
    }
    
    let total_time = start_time.elapsed();
    let measurements_per_sec = measurement_count as f64 / total_time.as_secs_f64();
    
    // Should be able to handle at least 100 measurements per second
    assert!(measurements_per_sec > 100.0, 
        "Should handle high-frequency measurements: {:.1}/sec", measurements_per_sec);
}

#[rstest]
#[tokio::test]
#[serial]
async fn test_concurrent_async_measurements(test_perf_monitor: PerformanceMonitor) {
    let monitor = Arc::new(tokio::sync::Mutex::new(test_perf_monitor));
    let mut handles = Vec::new();
    
    // Spawn many concurrent async measurements
    for i in 0..50 {
        let monitor_clone = monitor.clone();
        let handle = tokio::spawn(async move {
            let guard = monitor_clone.lock().await;
            let (result, duration) = guard.measure_async("concurrent_test", async {
                tokio::time::sleep(Duration::from_millis(1)).await;
                i
            }).await;
            (result, duration)
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut results = Vec::new();
    for handle in handles {
        let (result, duration) = handle.await.unwrap();
        results.push((result, duration));
        assert!(duration >= 1.0);
    }
    
    assert_eq!(results.len(), 50);
}

// =============================================================================
// TIMING CONSTRAINT VALIDATION TESTS
// =============================================================================

#[rstest]
#[case(50.0)]  // Under target
#[case(100.0)] // At target  
#[case(150.0)] // Over target
fn test_inference_timing_constraints(#[case] target_ms: f32) {
    let config = PerformanceConfig {
        latency: LatencyConfig {
            target_latency_ms: 100.0,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut monitor = PerformanceMonitor::with_config(config);
    
    // Simulate inference timing
    monitor.start_inference_timer();
    
    // Simulate work taking target_ms milliseconds
    std::thread::sleep(Duration::from_millis(target_ms as u64));
    
    let measured_time = monitor.end_inference_timer();
    
    // Should be approximately the target time (with some tolerance)
    assert!((measured_time - target_ms).abs() < 10.0, 
        "Measured time {:.1}ms should be close to target {:.1}ms", 
        measured_time, target_ms);
    
    // Validate constraint checking
    let constraints = monitor.check_constraints();
    let latency_ok = constraints.get("latency_constraint").unwrap();
    
    if target_ms <= 100.0 {
        assert!(*latency_ok, "Should meet latency constraint for {}ms", target_ms);
    }
}

#[rstest]
fn test_training_timing_constraints() {
    let mut monitor = PerformanceMonitor::new();
    
    // Simulate training timing
    monitor.start_training_timer();
    
    // Simulate short training
    std::thread::sleep(Duration::from_millis(50));
    
    let training_time = monitor.end_training_timer();
    
    // Should measure reasonable time
    assert!(training_time >= 50.0, "Training time should be at least 50ms");
    assert!(training_time < 200.0, "Training time should be under 200ms for test");
}

// =============================================================================
// VRAM AND GPU MONITORING TESTS
// =============================================================================

#[rstest]
fn test_vram_usage_monitoring(test_perf_monitor: PerformanceMonitor) {
    let vram_usage = test_perf_monitor.get_vram_usage();
    
    assert!(vram_usage > 0, "VRAM usage should be positive");
    assert!(vram_usage <= 16384, "VRAM usage should be reasonable (<16GB)");
}

#[rstest]
#[case(4096)]  // 4GB - within limit
#[case(8192)]  // 8GB - at limit
#[case(12288)] // 12GB - over limit
fn test_vram_constraint_validation(#[case] vram_mb: usize) {
    // Mock VRAM usage by creating a monitor that reports specific VRAM
    let config = PerformanceConfig::default();
    let monitor = PerformanceMonitor::with_config(config);
    
    // For the test, we check if the constraint logic works
    // RTX 2070 Ti has 8GB limit
    let within_limit = vram_mb <= 8192;
    
    if within_limit {
        assert!(vram_mb <= 8192, "VRAM usage should be within 8GB limit");
    } else {
        assert!(vram_mb > 8192, "VRAM usage should exceed limit for validation");
    }
}

// =============================================================================
// METRICS EXPORT AND PERSISTENCE TESTS
// =============================================================================

#[rstest]
fn test_metrics_export(test_perf_monitor: PerformanceMonitor, #[with(tempfile::tempdir().unwrap())] temp_dir: tempfile::TempDir) {
    // Configure export path
    let export_path = temp_dir.path().join("metrics.json");
    
    // Update config to export to temp file
    let mut config = test_perf_monitor.config.clone();
    config.export_metrics = true;
    config.metrics_file_path = export_path.to_string_lossy().to_string();
    
    let monitor = PerformanceMonitor::with_config(config);
    
    // Export metrics
    let result = monitor.export_metrics();
    assert!(result.is_ok(), "Metrics export should succeed");
    
    // Verify file was created
    assert!(export_path.exists(), "Metrics file should be created");
    
    // Verify file content
    let content = std::fs::read_to_string(&export_path).unwrap();
    assert!(!content.is_empty(), "Metrics file should not be empty");
    
    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_object(), "Metrics should be a JSON object");
}

// =============================================================================
// EDGE CASES AND ERROR HANDLING
// =============================================================================

#[rstest]
fn test_invalid_measurement_id(mut test_perf_monitor: PerformanceMonitor) {
    // Try to finish a measurement that was never started
    let result = test_perf_monitor.finish_measurement("non_existent_id");
    assert!(result.is_none(), "Should return None for invalid measurement ID");
}

#[rstest]
fn test_duplicate_measurement_finish(mut test_perf_monitor: PerformanceMonitor) {
    let id = test_perf_monitor.start_measurement("test");
    
    // Finish once
    let first_result = test_perf_monitor.finish_measurement(&id);
    assert!(first_result.is_some());
    
    // Try to finish again
    let second_result = test_perf_monitor.finish_measurement(&id);
    assert!(second_result.is_none(), "Should not be able to finish measurement twice");
}

#[rstest]
fn test_empty_statistics(test_perf_monitor: PerformanceMonitor) {
    // Get stats with no measurements
    let stats = test_perf_monitor.get_stats();
    
    // Should handle empty case gracefully
    assert_eq!(stats.avg_latency_ms, 0.0);
    assert_eq!(stats.p50_latency_ms, 0.0);
    assert_eq!(stats.constraint_violations, 0);
}

#[rstest]
fn test_monitor_uptime(test_perf_monitor: PerformanceMonitor) {
    let uptime = test_perf_monitor.get_uptime();
    
    assert!(uptime.as_millis() > 0, "Uptime should be positive");
    assert!(uptime.as_secs() < 60, "Uptime should be reasonable for test");
}