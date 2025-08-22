//! Performance monitoring for energy and latency tracking
//!
//! Provides comprehensive monitoring of energy consumption, latency,
//! memory usage, and throughput to ensure <50W power and <100ms inference targets.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::fs;
use serde::{Deserialize, Serialize};
use tokio::time::interval;

/// Performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: u64,
    pub latency_ms: f32,
    pub energy_consumption_w: f32,
    pub memory_usage_mb: usize,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub gpu_memory_mb: usize,
    pub throughput_tokens_per_sec: f32,
    pub model_efficiency_score: f32,
}

/// Energy monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConfig {
    pub target_power_limit_w: f32,
    pub measurement_interval_ms: u64,
    pub enable_cpu_monitoring: bool,
    pub enable_gpu_monitoring: bool,
    pub enable_system_monitoring: bool,
    pub power_alert_threshold_w: f32,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            target_power_limit_w: 50.0,
            measurement_interval_ms: 100,
            enable_cpu_monitoring: true,
            enable_gpu_monitoring: true,
            enable_system_monitoring: true,
            power_alert_threshold_w: 45.0,
        }
    }
}

/// Latency monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    pub target_latency_ms: f32,
    pub percentile_tracking: Vec<f32>,
    pub history_size: usize,
    pub alert_threshold_ms: f32,
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 100.0,
            percentile_tracking: vec![50.0, 90.0, 95.0, 99.0],
            history_size: 1000,
            alert_threshold_ms: 90.0,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub energy: EnergyConfig,
    pub latency: LatencyConfig,
    pub monitoring_enabled: bool,
    pub log_interval_sec: u64,
    pub metrics_retention_hours: u32,
    pub export_metrics: bool,
    pub metrics_file_path: String,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            energy: EnergyConfig::default(),
            latency: LatencyConfig::default(),
            monitoring_enabled: true,
            log_interval_sec: 30,
            metrics_retention_hours: 24,
            export_metrics: true,
            metrics_file_path: "metrics/performance.json".to_string(),
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub avg_latency_ms: f32,
    pub p50_latency_ms: f32,
    pub p90_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub max_latency_ms: f32,
    pub avg_energy_w: f32,
    pub peak_energy_w: f32,
    pub total_energy_wh: f32,
    pub avg_memory_mb: usize,
    pub peak_memory_mb: usize,
    pub efficiency_score: f32,
    pub constraint_violations: usize,
}

/// Energy measurement source
#[derive(Debug, Clone)]
pub enum EnergySource {
    CPU,
    GPU,
    System,
    Mock(f32), // For testing
}

/// Latency measurement
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    pub operation: String,
    pub start_time: Instant,
    pub duration: Option<Duration>,
    pub context: HashMap<String, String>,
}

impl LatencyMeasurement {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            start_time: Instant::now(),
            duration: None,
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    pub fn finish(mut self) -> Self {
        self.duration = Some(self.start_time.elapsed());
        self
    }

    pub fn get_duration_ms(&self) -> f32 {
        self.duration
            .unwrap_or_else(|| self.start_time.elapsed())
            .as_millis() as f32
    }
}

/// Performance monitor
pub struct PerformanceMonitor {
    config: PerformanceConfig,
    metrics_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    latency_history: Arc<Mutex<VecDeque<f32>>>,
    energy_history: Arc<Mutex<VecDeque<f32>>>,
    active_measurements: Arc<Mutex<HashMap<String, LatencyMeasurement>>>,
    monitoring_thread: Option<thread::JoinHandle<()>>,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self::with_config(PerformanceConfig::default())
    }

    pub fn with_config(config: PerformanceConfig) -> Self {
        Self {
            config,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            latency_history: Arc::new(Mutex::new(VecDeque::new())),
            energy_history: Arc::new(Mutex::new(VecDeque::new())),
            active_measurements: Arc::new(Mutex::new(HashMap::new())),
            monitoring_thread: None,
            start_time: Instant::now(),
        }
    }

    /// Start background monitoring
    pub fn start_monitoring(&mut self) -> crate::Result<()> {
        if !self.config.monitoring_enabled {
            return Ok(());
        }

        let config = self.config.clone();
        let metrics_history = self.metrics_history.clone();
        let energy_history = self.energy_history.clone();

        let handle = thread::spawn(move || {
            Self::monitoring_loop(config, metrics_history, energy_history);
        });

        self.monitoring_thread = Some(handle);
        log::info!("Performance monitoring started");
        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) {
        if let Some(_handle) = self.monitoring_thread.take() {
            // Note: In a real implementation, we'd use a shutdown signal
            log::info!("Performance monitoring stopped");
        }
    }

    /// Background monitoring loop
    fn monitoring_loop(
        config: PerformanceConfig,
        metrics_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
        energy_history: Arc<Mutex<VecDeque<f32>>>,
    ) {
        let interval = std::time::Duration::from_millis(config.energy.measurement_interval_ms);
        let mut last_log = Instant::now();

        loop {
            let start = Instant::now();

            // Collect metrics
            let metrics = Self::collect_system_metrics(&config);

            // Store metrics
            {
                let mut history = metrics_history.lock().unwrap();
                history.push_back(metrics.clone());

                // Cleanup old metrics
                let retention_duration = Duration::from_secs(config.metrics_retention_hours as u64 * 3600);
                let cutoff_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
                    - retention_duration.as_secs();

                history.retain(|m| m.timestamp > cutoff_time);
            }

            // Store energy data
            {
                let mut energy = energy_history.lock().unwrap();
                energy.push_back(metrics.energy_consumption_w);
                if energy.len() > 10000 {
                    energy.pop_front();
                }
            }

            // Periodic logging
            if last_log.elapsed() >= Duration::from_secs(config.log_interval_sec) {
                log::info!("Performance: {:.1}W, {:.1}ms latency, {}MB memory",
                    metrics.energy_consumption_w,
                    metrics.latency_ms,
                    metrics.memory_usage_mb
                );
                last_log = Instant::now();
            }

            // Check constraints
            if metrics.energy_consumption_w > config.energy.power_alert_threshold_w {
                log::warn!("Power consumption alert: {:.1}W exceeds threshold {:.1}W",
                    metrics.energy_consumption_w, config.energy.power_alert_threshold_w);
            }

            if metrics.latency_ms > config.latency.alert_threshold_ms {
                log::warn!("Latency alert: {:.1}ms exceeds threshold {:.1}ms",
                    metrics.latency_ms, config.latency.alert_threshold_ms);
            }

            // Sleep until next measurement
            let elapsed = start.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }
    }

    /// Collect system metrics
    fn collect_system_metrics(config: &PerformanceConfig) -> PerformanceMetrics {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        PerformanceMetrics {
            timestamp,
            latency_ms: Self::measure_inference_latency(),
            energy_consumption_w: Self::measure_energy_consumption(config),
            memory_usage_mb: Self::get_memory_usage_mb_static(),
            cpu_usage_percent: Self::get_cpu_usage(),
            gpu_usage_percent: Self::get_gpu_usage(),
            gpu_memory_mb: Self::get_gpu_memory_usage(),
            throughput_tokens_per_sec: Self::calculate_throughput(),
            model_efficiency_score: Self::calculate_efficiency_score(),
        }
    }

    /// Measure current inference latency
    fn measure_inference_latency() -> f32 {
        // Mock implementation - in real scenario, this would measure actual inference
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(50.0..120.0) // Simulate latency between 50-120ms
    }

    /// Measure energy consumption
    fn measure_energy_consumption(config: &PerformanceConfig) -> f32 {
        let mut total_power = 0.0;

        if config.energy.enable_cpu_monitoring {
            total_power += Self::measure_cpu_power();
        }

        if config.energy.enable_gpu_monitoring {
            total_power += Self::measure_gpu_power();
        }

        if config.energy.enable_system_monitoring {
            total_power += Self::measure_system_power();
        }

        total_power
    }

    /// Measure CPU power consumption
    fn measure_cpu_power() -> f32 {
        // Mock implementation - real implementation would use:
        // - Intel RAPL (Running Average Power Limit)
        // - AMD Energy Counters
        // - System power sensors
        
        // Simulate CPU power based on usage
        let cpu_usage = Self::get_cpu_usage();
        let base_power = 15.0; // Base CPU power in watts
        let max_additional = 25.0; // Max additional power under load
        
        base_power + (cpu_usage / 100.0) * max_additional
    }

    /// Measure GPU power consumption
    fn measure_gpu_power() -> f32 {
        // Mock implementation - real implementation would use:
        // - NVIDIA Management Library (NVML)
        // - nvidia-smi output parsing
        // - GPU sensors
        
        let gpu_usage = Self::get_gpu_usage();
        let base_power = 10.0; // Idle GPU power
        let max_power = 120.0; // RTX 2070 Ti TDP
        
        base_power + (gpu_usage / 100.0) * (max_power - base_power)
    }

    /// Measure system power consumption
    fn measure_system_power() -> f32 {
        // Mock implementation - real implementation would use:
        // - Hardware power sensors
        // - BMC (Baseboard Management Controller)
        // - Smart power strips
        
        8.0 // Estimated system overhead (RAM, motherboard, etc.)
    }

    /// Get memory usage in MB
    pub fn get_memory_usage_mb(&self) -> usize {
        Self::get_memory_usage_mb_static()
    }

    fn get_memory_usage_mb_static() -> usize {
        // Cross-platform memory usage
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = fs::read_to_string("/proc/self/status") {
                for line in content.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Mock implementation for Windows
            use rand::Rng;
            let mut rng = rand::thread_rng();
            return rng.gen_range(2000..7500); // Simulate 2-7.5GB usage
        }

        #[cfg(target_os = "macos")]
        {
            // Mock implementation for macOS
            use rand::Rng;
            let mut rng = rand::thread_rng();
            return rng.gen_range(2000..7500);
        }

        4096 // Default 4GB
    }

    /// Get CPU usage percentage
    fn get_cpu_usage() -> f32 {
        // Mock implementation - real implementation would use:
        // - /proc/stat on Linux
        // - Performance counters on Windows
        // - System APIs on macOS
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(20.0..80.0)
    }

    /// Get GPU usage percentage
    fn get_gpu_usage() -> f32 {
        // Mock implementation - real implementation would use nvidia-ml-py or similar
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(30.0..95.0)
    }

    /// Get GPU memory usage in MB
    fn get_gpu_memory_usage() -> usize {
        // Mock implementation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(3000..7500) // 3-7.5GB GPU memory usage
    }

    /// Calculate tokens per second throughput
    fn calculate_throughput() -> f32 {
        // Mock implementation based on current performance
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(100.0..500.0) // 100-500 tokens/sec
    }

    /// Calculate model efficiency score
    fn calculate_efficiency_score() -> f32 {
        // Composite score: throughput / (power * latency)
        let throughput = Self::calculate_throughput();
        let power = 45.0; // Assume current power
        let latency = 80.0; // Assume current latency
        
        let efficiency = throughput / (power * latency / 1000.0);
        efficiency.min(1.0) // Normalize to 0-1
    }

    /// Start latency measurement
    pub fn start_measurement(&self, operation: &str) -> String {
        let measurement_id = format!("{}_{}", operation, self.get_timestamp_ns());
        let measurement = LatencyMeasurement::new(operation);
        
        {
            let mut measurements = self.active_measurements.lock().unwrap();
            measurements.insert(measurement_id.clone(), measurement);
        }
        
        measurement_id
    }

    /// Finish latency measurement
    pub fn finish_measurement(&self, measurement_id: &str) -> Option<f32> {
        let mut measurements = self.active_measurements.lock().unwrap();
        if let Some(mut measurement) = measurements.remove(measurement_id) {
            measurement = measurement.finish();
            let duration_ms = measurement.get_duration_ms();
            
            // Store in latency history
            {
                let mut history = self.latency_history.lock().unwrap();
                history.push_back(duration_ms);
                if history.len() > self.config.latency.history_size {
                    history.pop_front();
                }
            }
            
            Some(duration_ms)
        } else {
            None
        }
    }

    /// Measure operation with automatic timing
    pub async fn measure_async<F, T>(&self, operation: &str, func: F) -> (T, f32)
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = func.await;
        let duration_ms = start.elapsed().as_millis() as f32;
        
        // Store measurement
        {
            let mut history = self.latency_history.lock().unwrap();
            history.push_back(duration_ms);
            if history.len() > self.config.latency.history_size {
                history.pop_front();
            }
        }
        
        log::debug!("Operation '{}' took {:.2}ms", operation, duration_ms);
        (result, duration_ms)
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let latency_history = self.latency_history.lock().unwrap();
        let energy_history = self.energy_history.lock().unwrap();
        
        // Calculate latency statistics
        let mut latencies: Vec<f32> = latency_history.iter().copied().collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<f32>() / latencies.len() as f32
        } else {
            0.0
        };
        
        let p50 = Self::percentile(&latencies, 50.0);
        let p90 = Self::percentile(&latencies, 90.0);
        let p95 = Self::percentile(&latencies, 95.0);
        let p99 = Self::percentile(&latencies, 99.0);
        let max_latency = latencies.last().copied().unwrap_or(0.0);
        
        // Calculate energy statistics
        let avg_energy = if !energy_history.is_empty() {
            energy_history.iter().sum::<f32>() / energy_history.len() as f32
        } else {
            0.0
        };
        
        let peak_energy = energy_history.iter().fold(0.0f32, |a, &b| a.max(b));
        let total_energy_wh = avg_energy * self.start_time.elapsed().as_secs_f32() / 3600.0;
        
        // Count constraint violations
        let latency_violations = latencies.iter()
            .filter(|&&l| l > self.config.latency.target_latency_ms)
            .count();
        let energy_violations = energy_history.iter()
            .filter(|&&e| e > self.config.energy.target_power_limit_w)
            .count();
        
        PerformanceStats {
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            max_latency_ms: max_latency,
            avg_energy_w: avg_energy,
            peak_energy_w: peak_energy,
            total_energy_wh,
            avg_memory_mb: Self::get_memory_usage_mb_static(),
            peak_memory_mb: Self::get_memory_usage_mb_static(), // Would track max in real impl
            efficiency_score: Self::calculate_efficiency_score(),
            constraint_violations: latency_violations + energy_violations,
        }
    }

    /// Calculate percentile
    fn percentile(sorted_values: &[f32], percentile: f32) -> f32 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        if sorted_values.len() == 1 {
            return sorted_values[0];
        }
        
        // Using proper percentile calculation with linear interpolation
        let n = sorted_values.len() as f32;
        let rank = (percentile / 100.0) * (n - 1.0);
        let lower_index = rank.floor() as usize;
        let upper_index = lower_index + 1;
        let weight = rank.fract();
        
        if upper_index >= sorted_values.len() {
            sorted_values[lower_index]
        } else {
            sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
        }
    }

    /// Export metrics to file
    pub fn export_metrics(&self) -> crate::Result<()> {
        if !self.config.export_metrics {
            return Ok(());
        }

        let stats = self.get_stats();
        let metrics_json = serde_json::to_string_pretty(&stats)?;
        
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&self.config.metrics_file_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(&self.config.metrics_file_path, metrics_json)?;
        log::info!("Metrics exported to {}", self.config.metrics_file_path);
        
        Ok(())
    }

    /// Check if performance constraints are met
    pub fn check_constraints(&self) -> HashMap<String, bool> {
        let stats = self.get_stats();
        let mut constraints = HashMap::new();
        
        constraints.insert("latency_constraint".to_string(), 
            stats.p95_latency_ms <= self.config.latency.target_latency_ms);
        constraints.insert("energy_constraint".to_string(), 
            stats.avg_energy_w <= self.config.energy.target_power_limit_w);
        constraints.insert("memory_constraint".to_string(), 
            stats.avg_memory_mb <= 8192); // 8GB limit
        
        constraints
    }

    /// Get monitoring uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get timestamp in nanoseconds
    fn get_timestamp_ns(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    /// Start inference timer
    pub fn start_inference_timer(&mut self) {
        self.start_measurement("inference");
    }

    /// End inference timer and return duration
    pub fn end_inference_timer(&mut self) -> f32 {
        self.finish_measurement("inference").unwrap_or(0.0)
    }

    /// Start training timer
    pub fn start_training_timer(&mut self) {
        self.start_measurement("training");
    }

    /// End training timer and return duration
    pub fn end_training_timer(&mut self) -> f32 {
        self.finish_measurement("training").unwrap_or(0.0)
    }

    /// Get current power consumption
    pub fn get_power_consumption(&self) -> f32 {
        Self::measure_energy_consumption(&self.config)
    }

    /// Get VRAM usage in MB
    pub fn get_vram_usage(&self) -> usize {
        Self::get_gpu_memory_usage()
    }
}

impl Drop for PerformanceMonitor {
    fn drop(&mut self) {
        self.stop_monitoring();
        if let Err(e) = self.export_metrics() {
            log::warn!("Failed to export metrics on drop: {}", e);
        }
    }
}

/// Utility macro for easy performance measurement
#[macro_export]
macro_rules! measure_perf {
    ($monitor:expr, $operation:expr, $code:block) => {{
        let id = $monitor.start_measurement($operation);
        let result = $code;
        $monitor.finish_measurement(&id);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert!(monitor.config.monitoring_enabled);
        assert_eq!(monitor.config.energy.target_power_limit_w, 50.0);
        assert_eq!(monitor.config.latency.target_latency_ms, 100.0);
    }

    #[test]
    fn test_latency_measurement() {
        let monitor = PerformanceMonitor::new();
        
        let id = monitor.start_measurement("test_operation");
        std::thread::sleep(Duration::from_millis(10));
        let duration = monitor.finish_measurement(&id);
        
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 10.0);
    }

    #[tokio::test]
    async fn test_async_measurement() {
        let monitor = PerformanceMonitor::new();
        
        let (result, duration) = monitor.measure_async("async_test", async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            42
        }).await;
        
        assert_eq!(result, 42);
        assert!(duration >= 5.0);
    }

    #[test]
    fn test_performance_stats() {
        let monitor = PerformanceMonitor::new();
        
        // Add some measurements
        for i in 0..10 {
            let mut history = monitor.latency_history.lock().unwrap();
            history.push_back((i as f32 + 1.0) * 10.0); // 10, 20, 30, ... 100ms
        }
        
        let stats = monitor.get_stats();
        assert!(stats.avg_latency_ms > 0.0);
        assert!(stats.p95_latency_ms > stats.p50_latency_ms);
    }

    #[test]
    fn test_constraint_checking() {
        let monitor = PerformanceMonitor::new();
        let constraints = monitor.check_constraints();
        
        assert!(constraints.contains_key("latency_constraint"));
        assert!(constraints.contains_key("energy_constraint"));
        assert!(constraints.contains_key("memory_constraint"));
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        
        let p50 = PerformanceMonitor::percentile(&values, 50.0);
        let p90 = PerformanceMonitor::percentile(&values, 90.0);
        let p95 = PerformanceMonitor::percentile(&values, 95.0);
        
        // Using proper percentile calculation with linear interpolation:
        // For 10 elements (indices 0-9):
        // 50th percentile = index 4.5 → (50.0 + 60.0) / 2 = 55.0
        // 90th percentile = index 8.1 → 90.0 * 0.9 + 100.0 * 0.1 = 91.0
        // 95th percentile = index 8.55 → 90.0 * 0.45 + 100.0 * 0.55 = 95.5
        assert_eq!(p50, 55.0);
        assert_eq!(p90, 91.0);
        assert_eq!(p95, 95.5);
    }

    #[test]
    fn test_memory_usage() {
        let monitor = PerformanceMonitor::new();
        let memory_mb = monitor.get_memory_usage_mb();
        assert!(memory_mb > 0);
        assert!(memory_mb < 32768); // Reasonable upper bound
    }
}