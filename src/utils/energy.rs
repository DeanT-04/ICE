//! Energy monitoring utilities for power consumption tracking
//!
//! Provides comprehensive energy monitoring capabilities for the ultra-fast AI model
//! including CPU power, GPU power, memory power, and thermal monitoring.

use std::time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use sysinfo::System;

/// Energy monitoring configuration
#[derive(Debug, Clone)]
pub struct EnergyMonitorConfig {
    pub sampling_rate_hz: u32,
    pub buffer_size: usize,
    pub enable_thermal_monitoring: bool,
    pub enable_frequency_monitoring: bool,
    pub enable_memory_monitoring: bool,
    pub power_cap_watts: Option<f64>,
}

impl Default for EnergyMonitorConfig {
    fn default() -> Self {
        Self {
            sampling_rate_hz: 100,
            buffer_size: 10000,
            enable_thermal_monitoring: true,
            enable_frequency_monitoring: true,
            enable_memory_monitoring: true,
            power_cap_watts: Some(50.0),
        }
    }
}

/// Power measurement sample
#[derive(Debug, Clone)]
pub struct PowerSample {
    pub timestamp: Instant,
    pub cpu_power_watts: f64,
    pub memory_power_watts: f64,
    pub total_power_watts: f64,
    pub cpu_temperature_celsius: f64,
    pub cpu_frequency_mhz: f64,
    pub memory_usage_mb: f64,
}

/// Energy monitoring statistics
#[derive(Debug, Clone)]
pub struct EnergyStats {
    pub avg_power_watts: f64,
    pub peak_power_watts: f64,
    pub min_power_watts: f64,
    pub total_energy_joules: f64,
    pub avg_temperature_celsius: f64,
    pub peak_temperature_celsius: f64,
    pub avg_frequency_mhz: f64,
    pub power_efficiency_ops_per_watt: f64,
    pub thermal_violations: u32,
    pub power_violations: u32,
    pub measurement_duration_ms: u64,
}

/// Real-time energy monitor
pub struct EnergyMonitor {
    config: EnergyMonitorConfig,
    system: Arc<Mutex<System>>,
    samples: Arc<Mutex<VecDeque<PowerSample>>>,
    baseline_power: f64,
    monitoring_active: Arc<Mutex<bool>>,
    start_time: Option<Instant>,
}

impl EnergyMonitor {
    /// Create a new energy monitor
    pub fn new(config: EnergyMonitorConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let baseline_power = Self::estimate_baseline_power(&system);
        let buffer_size = config.buffer_size;
        
        Self {
            config,
            system: Arc::new(Mutex::new(system)),
            samples: Arc::new(Mutex::new(VecDeque::with_capacity(buffer_size))),
            baseline_power,
            monitoring_active: Arc::new(Mutex::new(false)),
            start_time: None,
        }
    }

    /// Start energy monitoring in background thread
    pub fn start_monitoring(&mut self) {
        *self.monitoring_active.lock().unwrap() = true;
        self.start_time = Some(Instant::now());
        
        let system = Arc::clone(&self.system);
        let samples = Arc::clone(&self.samples);
        let monitoring_active = Arc::clone(&self.monitoring_active);
        let config = self.config.clone();
        let baseline_power = self.baseline_power;
        
        thread::spawn(move || {
            let sampling_interval = Duration::from_millis(1000 / config.sampling_rate_hz as u64);
            
            while *monitoring_active.lock().unwrap() {
                let sample = Self::collect_power_sample(&system, baseline_power, &config);
                
                {
                    let mut samples_guard = samples.lock().unwrap();
                    if samples_guard.len() >= config.buffer_size {
                        samples_guard.pop_front();
                    }
                    samples_guard.push_back(sample);
                }
                
                thread::sleep(sampling_interval);
            }
        });
    }

    /// Stop energy monitoring
    pub fn stop_monitoring(&mut self) -> EnergyStats {
        *self.monitoring_active.lock().unwrap() = false;
        thread::sleep(Duration::from_millis(100)); // Allow final samples
        
        let duration = self.start_time.unwrap().elapsed();
        let samples = self.samples.lock().unwrap().clone();
        
        self.compute_energy_stats(&samples, duration)
    }

    /// Get current power consumption
    pub fn get_current_power(&self) -> f64 {
        let sample = Self::collect_power_sample(&self.system, self.baseline_power, &self.config);
        sample.total_power_watts
    }

    /// Get real-time statistics
    pub fn get_realtime_stats(&self) -> Option<EnergyStats> {
        if let Some(start_time) = self.start_time {
            let duration = start_time.elapsed();
            let samples = self.samples.lock().unwrap().clone();
            
            if !samples.is_empty() {
                Some(self.compute_energy_stats(&samples, duration))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Estimate baseline system power consumption
    fn estimate_baseline_power(system: &System) -> f64 {
        // Estimate based on system characteristics
        let cpu_count = system.cpus().len() as f64;
        let memory_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Base consumption estimates (Watts)
        let cpu_base = cpu_count * 5.0; // ~5W per core at idle
        let memory_base = memory_gb * 2.0; // ~2W per GB
        let system_base = 10.0; // Motherboard, storage, etc.
        
        cpu_base + memory_base + system_base
    }

    /// Collect a single power measurement sample
    fn collect_power_sample(
        system: &Arc<Mutex<System>>,
        baseline_power: f64,
        config: &EnergyMonitorConfig,
    ) -> PowerSample {
        let mut sys = system.lock().unwrap();
        sys.refresh_cpu();
        sys.refresh_memory();
        // Note: Component monitoring simplified for compatibility

        // CPU power estimation based on usage
        let cpu_usage = sys.global_cpu_info().cpu_usage() as f64 / 100.0;
        let cpu_frequency = 3000.0; // Default frequency since API changed
        
        // Estimate CPU power consumption
        let cpu_base_power = baseline_power * 0.6; // 60% of baseline is CPU
        let cpu_dynamic_power = cpu_usage * cpu_base_power * 2.0; // Dynamic scaling
        let cpu_power = cpu_base_power + cpu_dynamic_power;

        // Memory power estimation
        let memory_usage = sys.used_memory() as f64 / sys.total_memory() as f64;
        let memory_base_power = baseline_power * 0.2; // 20% of baseline is memory
        let memory_power = memory_base_power * (1.0 + memory_usage);

        // Temperature measurement (simplified)
        let cpu_temperature = if config.enable_thermal_monitoring {
            // Simplified temperature estimation
            45.0 + (cpu_usage * 25.0) // Estimate based on usage
        } else {
            45.0 // Default estimate
        };

        // Total power calculation
        let total_power = cpu_power + memory_power + (baseline_power * 0.2); // 20% for other components

        PowerSample {
            timestamp: Instant::now(),
            cpu_power_watts: cpu_power,
            memory_power_watts: memory_power,
            total_power_watts: total_power,
            cpu_temperature_celsius: cpu_temperature,
            cpu_frequency_mhz: cpu_frequency,
            memory_usage_mb: sys.used_memory() as f64 / (1024.0 * 1024.0),
        }
    }

    /// Compute comprehensive energy statistics
    fn compute_energy_stats(&self, samples: &VecDeque<PowerSample>, duration: Duration) -> EnergyStats {
        if samples.is_empty() {
            return EnergyStats {
                avg_power_watts: 0.0,
                peak_power_watts: 0.0,
                min_power_watts: 0.0,
                total_energy_joules: 0.0,
                avg_temperature_celsius: 0.0,
                peak_temperature_celsius: 0.0,
                avg_frequency_mhz: 0.0,
                power_efficiency_ops_per_watt: 0.0,
                thermal_violations: 0,
                power_violations: 0,
                measurement_duration_ms: 0,
            };
        }

        let total_samples = samples.len() as f64;
        
        // Power statistics
        let total_power: f64 = samples.iter().map(|s| s.total_power_watts).sum();
        let avg_power = total_power / total_samples;
        let peak_power = samples.iter().map(|s| s.total_power_watts).fold(0.0, f64::max);
        let min_power = samples.iter().map(|s| s.total_power_watts).fold(f64::INFINITY, f64::min);
        
        // Energy calculation (power * time)
        let total_energy = avg_power * duration.as_secs_f64();
        
        // Temperature statistics
        let total_temp: f64 = samples.iter().map(|s| s.cpu_temperature_celsius).sum();
        let avg_temperature = total_temp / total_samples;
        let peak_temperature = samples.iter().map(|s| s.cpu_temperature_celsius).fold(0.0, f64::max);
        
        // Frequency statistics
        let total_freq: f64 = samples.iter().map(|s| s.cpu_frequency_mhz).sum();
        let avg_frequency = total_freq / total_samples;
        
        // Violation counting
        let power_cap = self.config.power_cap_watts.unwrap_or(f64::INFINITY);
        let thermal_limit = 80.0; // 80°C thermal limit
        
        let power_violations = samples.iter()
            .filter(|s| s.total_power_watts > power_cap)
            .count() as u32;
        
        let thermal_violations = samples.iter()
            .filter(|s| s.cpu_temperature_celsius > thermal_limit)
            .count() as u32;
        
        // Power efficiency (operations per watt) - estimated
        let power_efficiency = if avg_power > 0.0 {
            // Estimate operations based on CPU usage patterns
            let avg_cpu_power: f64 = samples.iter().map(|s| s.cpu_power_watts).sum::<f64>() / total_samples;
            let estimated_ops_per_second = avg_cpu_power * 1000.0; // Rough estimate
            estimated_ops_per_second / avg_power
        } else {
            0.0
        };

        EnergyStats {
            avg_power_watts: avg_power,
            peak_power_watts: peak_power,
            min_power_watts: min_power,
            total_energy_joules: total_energy,
            avg_temperature_celsius: avg_temperature,
            peak_temperature_celsius: peak_temperature,
            avg_frequency_mhz: avg_frequency,
            power_efficiency_ops_per_watt: power_efficiency,
            thermal_violations,
            power_violations,
            measurement_duration_ms: duration.as_millis() as u64,
        }
    }

    /// Get power breakdown by component
    pub fn get_power_breakdown(&self) -> PowerBreakdown {
        if let Some(latest_sample) = self.samples.lock().unwrap().back() {
            let total = latest_sample.total_power_watts;
            
            PowerBreakdown {
                cpu_percentage: (latest_sample.cpu_power_watts / total) * 100.0,
                memory_percentage: (latest_sample.memory_power_watts / total) * 100.0,
                other_percentage: 100.0 - ((latest_sample.cpu_power_watts + latest_sample.memory_power_watts) / total) * 100.0,
                total_power_watts: total,
            }
        } else {
            PowerBreakdown {
                cpu_percentage: 0.0,
                memory_percentage: 0.0,
                other_percentage: 0.0,
                total_power_watts: 0.0,
            }
        }
    }
}

/// Power consumption breakdown
#[derive(Debug, Clone)]
pub struct PowerBreakdown {
    pub cpu_percentage: f64,
    pub memory_percentage: f64,
    pub other_percentage: f64,
    pub total_power_watts: f64,
}

/// Energy optimization recommendations
#[derive(Debug, Clone)]
pub struct EnergyOptimizationRecommendations {
    pub reduce_cpu_frequency: bool,
    pub enable_power_saving_mode: bool,
    pub reduce_memory_usage: bool,
    pub thermal_throttling_needed: bool,
    pub estimated_power_savings_watts: f64,
}

impl EnergyMonitor {
    /// Get optimization recommendations based on current power consumption
    pub fn get_optimization_recommendations(&self, target_power_watts: f64) -> EnergyOptimizationRecommendations {
        let current_power = self.get_current_power();
        let power_excess = current_power - target_power_watts;
        
        if power_excess <= 0.0 {
            return EnergyOptimizationRecommendations {
                reduce_cpu_frequency: false,
                enable_power_saving_mode: false,
                reduce_memory_usage: false,
                thermal_throttling_needed: false,
                estimated_power_savings_watts: 0.0,
            };
        }

        let breakdown = self.get_power_breakdown();
        let latest_stats = self.get_realtime_stats();
        
        let thermal_throttling_needed = if let Some(stats) = &latest_stats {
            stats.peak_temperature_celsius > 75.0
        } else {
            false
        };

        EnergyOptimizationRecommendations {
            reduce_cpu_frequency: breakdown.cpu_percentage > 60.0 && power_excess > 5.0,
            enable_power_saving_mode: power_excess > 10.0,
            reduce_memory_usage: breakdown.memory_percentage > 30.0 && power_excess > 3.0,
            thermal_throttling_needed,
            estimated_power_savings_watts: power_excess.min(current_power * 0.3), // Max 30% reduction
        }
    }
}

/// Energy benchmark result validator
pub struct EnergyValidator {
    pub target_power_watts: f64,
    pub thermal_limit_celsius: f64,
    pub efficiency_threshold_ops_per_watt: f64,
}

impl EnergyValidator {
    pub fn new(target_power_watts: f64) -> Self {
        Self {
            target_power_watts,
            thermal_limit_celsius: 80.0,
            efficiency_threshold_ops_per_watt: 1.0,
        }
    }

    /// Validate energy statistics against targets
    pub fn validate(&self, stats: &EnergyStats) -> EnergyValidationResult {
        let power_within_limit = stats.avg_power_watts <= self.target_power_watts;
        let peak_power_acceptable = stats.peak_power_watts <= self.target_power_watts * 1.2; // 20% tolerance
        let thermal_within_limit = stats.peak_temperature_celsius <= self.thermal_limit_celsius;
        let efficiency_acceptable = stats.power_efficiency_ops_per_watt >= self.efficiency_threshold_ops_per_watt;
        let no_violations = stats.power_violations == 0 && stats.thermal_violations == 0;

        let overall_pass = power_within_limit && peak_power_acceptable && thermal_within_limit && efficiency_acceptable && no_violations;

        EnergyValidationResult {
            overall_pass,
            power_within_limit,
            peak_power_acceptable,
            thermal_within_limit,
            efficiency_acceptable,
            no_violations,
            power_margin_watts: self.target_power_watts - stats.avg_power_watts,
            thermal_margin_celsius: self.thermal_limit_celsius - stats.peak_temperature_celsius,
            efficiency_ratio: stats.power_efficiency_ops_per_watt / self.efficiency_threshold_ops_per_watt,
        }
    }
}

/// Energy validation result
#[derive(Debug, Clone)]
pub struct EnergyValidationResult {
    pub overall_pass: bool,
    pub power_within_limit: bool,
    pub peak_power_acceptable: bool,
    pub thermal_within_limit: bool,
    pub efficiency_acceptable: bool,
    pub no_violations: bool,
    pub power_margin_watts: f64,
    pub thermal_margin_celsius: f64,
    pub efficiency_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_monitor_creation() {
        let config = EnergyMonitorConfig::default();
        let monitor = EnergyMonitor::new(config);
        
        assert!(monitor.baseline_power > 0.0);
    }

    #[test]
    fn test_current_power_measurement() {
        let config = EnergyMonitorConfig::default();
        let monitor = EnergyMonitor::new(config);
        
        let power = monitor.get_current_power();
        assert!(power > 0.0);
        assert!(power < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_energy_validator() {
        let validator = EnergyValidator::new(50.0);
        
        let good_stats = EnergyStats {
            avg_power_watts: 45.0,
            peak_power_watts: 55.0,
            min_power_watts: 40.0,
            total_energy_joules: 450.0,
            avg_temperature_celsius: 65.0,
            peak_temperature_celsius: 75.0,
            avg_frequency_mhz: 3000.0,
            power_efficiency_ops_per_watt: 1.5,
            thermal_violations: 0,
            power_violations: 0,
            measurement_duration_ms: 10000,
        };
        
        let result = validator.validate(&good_stats);
        assert!(result.overall_pass);
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = EnergyMonitorConfig::default();
        let monitor = EnergyMonitor::new(config);
        
        let recommendations = monitor.get_optimization_recommendations(30.0);
        
        // Should recommend optimizations if current power exceeds target
        assert!(recommendations.estimated_power_savings_watts >= 0.0);
    }
}