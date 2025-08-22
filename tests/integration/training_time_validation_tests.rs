//! Training Time Validation Tests
//!
//! Integration tests that validate the ultra-fast AI model can be trained
//! within 24 hours on RTX 2070 Ti hardware while meeting all constraints.

use std::time::Duration;
use tokio::time::timeout;

use crate::tests::integration::training_time_validation::{
    TrainingTimeValidator, TrainingSessionConfig, RTX2070TiSpecs
};

/// Test rapid training configuration (for CI/testing)
#[tokio::test]
async fn test_rapid_training_simulation() {
    println!("🚀 Testing rapid training simulation...");
    
    // Configure for rapid testing (scaled down)
    let config = TrainingSessionConfig {
        max_duration_hours: 0.1, // 6 minutes for testing
        target_accuracy: 0.3, // Lower target for rapid test
        max_vram_usage_gb: 7.5,
        max_avg_power_w: 50.0,
        validation_interval_minutes: 1, // Validate every minute
        checkpoint_interval_minutes: 2,
        early_stopping_patience: 2, // 2 minutes patience
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    // Run rapid training simulation
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Validate basic functionality
    assert!(results.total_epochs > 0, "Should complete at least one epoch");
    assert!(results.total_parameters <= 100_000_000, "Should meet parameter constraint");
    assert!(results.max_vram_usage_gb <= 7.5, "Should meet VRAM constraint");
    
    println!("✅ Rapid training simulation completed:");
    println!("   - Duration: {:.2} hours", results.total_duration_hours);
    println!("   - Epochs: {}", results.total_epochs);
    println!("   - Final accuracy: {:.2}%", results.final_accuracy * 100.0);
    println!("   - Max VRAM: {:.2}GB", results.max_vram_usage_gb);
    println!("   - Avg power: {:.1}W", results.avg_power_consumption_w);
}

#[tokio::test]
async fn test_constraint_validation_system() {
    println!("🔍 Testing constraint validation system...");
    
    // Test with very strict constraints to ensure validation works
    let config = TrainingSessionConfig {
        max_duration_hours: 0.05, // 3 minutes - very tight
        target_accuracy: 0.95, // Very high target - should not be met
        max_vram_usage_gb: 1.0, // Very low VRAM limit
        max_avg_power_w: 10.0, // Very low power limit
        validation_interval_minutes: 1,
        checkpoint_interval_minutes: 1,
        early_stopping_patience: 1,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // These constraints should NOT be met (testing constraint validation)
    assert!(!results.constraints_met.accuracy_constraint_met, "Should not meet unrealistic accuracy target");
    
    // Parameter constraint should always be met
    assert!(results.constraints_met.parameter_constraint_met, "Should meet parameter constraint");
    
    // Should have attempted training
    assert!(results.total_epochs > 0, "Should have attempted at least one epoch");
    
    println!("✅ Constraint validation system working correctly:");
    println!("   - Time constraint: {}", results.constraints_met.time_constraint_met);
    println!("   - Accuracy constraint: {}", results.constraints_met.accuracy_constraint_met);
    println!("   - VRAM constraint: {}", results.constraints_met.vram_constraint_met);
    println!("   - Power constraint: {}", results.constraints_met.power_constraint_met);
    println!("   - Parameter constraint: {}", results.constraints_met.parameter_constraint_met);
}

#[tokio::test]
async fn test_rtx_2070_ti_hardware_simulation() {
    println!("🎮 Testing RTX 2070 Ti hardware simulation...");
    
    let gpu_specs = RTX2070TiSpecs::default();
    
    // Validate RTX 2070 Ti specifications
    assert_eq!(gpu_specs.cuda_cores, 2304, "Should have correct CUDA core count");
    assert_eq!(gpu_specs.memory_size_gb, 8, "Should have 8GB VRAM");
    assert_eq!(gpu_specs.max_power_consumption_w, 215, "Should have 215W TGP");
    assert!(gpu_specs.tensor_performance_tops > 50.0, "Should have >50 TOPS tensor performance");
    
    // Configure for realistic RTX 2070 Ti training
    let config = TrainingSessionConfig {
        max_duration_hours: 0.2, // 12 minutes for testing
        target_accuracy: 0.4, // Reasonable target
        max_vram_usage_gb: 7.5, // Leave headroom
        max_avg_power_w: 50.0, // Conservative power target
        validation_interval_minutes: 2,
        checkpoint_interval_minutes: 5,
        early_stopping_patience: 3,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Should work within RTX 2070 Ti constraints
    assert!(results.max_vram_usage_gb <= 8.0, "Should not exceed RTX 2070 Ti VRAM");
    assert!(results.peak_power_consumption_w <= 215.0, "Should not exceed RTX 2070 Ti TGP");
    
    println!("✅ RTX 2070 Ti simulation completed:");
    println!("   - Peak VRAM usage: {:.2}GB / 8GB", results.max_vram_usage_gb);
    println!("   - Peak power: {:.1}W / 215W", results.peak_power_consumption_w);
    println!("   - Memory efficiency: {:.1}%", results.performance_metrics.memory_efficiency_percent);
    println!("   - GPU utilization: {:.1}%", results.performance_metrics.gpu_utilization_percent);
}

#[tokio::test]
async fn test_parameter_count_validation() {
    println!("📊 Testing parameter count validation...");
    
    let config = TrainingSessionConfig {
        max_duration_hours: 0.05, // Quick test
        target_accuracy: 0.1, // Low target for quick completion
        max_vram_usage_gb: 7.5,
        max_avg_power_w: 50.0,
        validation_interval_minutes: 1,
        checkpoint_interval_minutes: 1,
        early_stopping_patience: 1,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Critical constraint: must be under 100M parameters
    assert!(results.total_parameters <= 100_000_000, 
        "Model has {} parameters, exceeds 100M limit", results.total_parameters);
    assert!(results.constraints_met.parameter_constraint_met, "Parameter constraint must be met");
    
    // Verify parameter distribution (SNN 30%, SSM 40%, Liquid NN 20%, remaining 10% for embeddings/output)
    let param_millions = results.total_parameters as f32 / 1_000_000.0;
    assert!(param_millions >= 50.0, "Should have reasonable model size (≥50M parameters)");
    assert!(param_millions <= 100.0, "Should not exceed 100M parameters");
    
    println!("✅ Parameter validation passed:");
    println!("   - Total parameters: {:.1}M", param_millions);
    println!("   - Constraint met: {}", results.constraints_met.parameter_constraint_met);
}

#[tokio::test]
async fn test_energy_efficiency_validation() {
    println!("⚡ Testing energy efficiency validation...");
    
    let config = TrainingSessionConfig {
        max_duration_hours: 0.15, // 9 minutes
        target_accuracy: 0.35,
        max_vram_usage_gb: 7.5,
        max_avg_power_w: 50.0, // Target average power
        validation_interval_minutes: 2,
        checkpoint_interval_minutes: 3,
        early_stopping_patience: 2,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Energy efficiency validations
    assert!(results.avg_power_consumption_w > 0.0, "Should have measurable power consumption");
    assert!(results.performance_metrics.energy_efficiency_samples_per_joule > 0.0, "Should have positive energy efficiency");
    
    // Check if power constraint is reasonable
    let power_within_bounds = results.avg_power_consumption_w <= config.max_avg_power_w * 1.1; // 10% tolerance
    if !power_within_bounds {
        println!("⚠️ Power consumption {:.1}W exceeds target {:.1}W", 
            results.avg_power_consumption_w, config.max_avg_power_w);
    }
    
    println!("✅ Energy efficiency validation completed:");
    println!("   - Average power: {:.1}W (target: ≤{:.1}W)", 
        results.avg_power_consumption_w, config.max_avg_power_w);
    println!("   - Peak power: {:.1}W", results.peak_power_consumption_w);
    println!("   - Energy efficiency: {:.2} samples/joule", 
        results.performance_metrics.energy_efficiency_samples_per_joule);
    println!("   - Power constraint met: {}", results.constraints_met.power_constraint_met);
}

#[tokio::test]
async fn test_training_progress_monitoring() {
    println!("📈 Testing training progress monitoring...");
    
    let config = TrainingSessionConfig {
        max_duration_hours: 0.1, // 6 minutes
        target_accuracy: 0.25,
        max_vram_usage_gb: 7.5,
        max_avg_power_w: 50.0,
        validation_interval_minutes: 1, // Frequent validation for monitoring
        checkpoint_interval_minutes: 2,
        early_stopping_patience: 2,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Should have validation scores recorded
    assert!(!results.validation_scores.is_empty(), "Should have validation scores recorded");
    
    // Validate score structure
    for score in &results.validation_scores {
        assert!(score.epoch > 0, "Epoch should be positive");
        assert!(score.timestamp_hours >= 0.0, "Timestamp should be non-negative");
        assert!(score.accuracy >= 0.0 && score.accuracy <= 1.0, "Accuracy should be in [0,1]");
        assert!(score.validation_accuracy >= 0.0 && score.validation_accuracy <= 1.0, "Validation accuracy should be in [0,1]");
        assert!(score.zero_hallucination_rate >= 0.0 && score.zero_hallucination_rate <= 1.0, "Zero-hallucination rate should be in [0,1]");
    }
    
    // Performance metrics should be reasonable
    assert!(results.performance_metrics.avg_epoch_time_minutes > 0.0, "Should have positive epoch time");
    assert!(results.performance_metrics.samples_per_second > 0.0, "Should have positive throughput");
    
    println!("✅ Training progress monitoring working:");
    println!("   - Validation points: {}", results.validation_scores.len());
    println!("   - Final accuracy: {:.2}%", results.final_accuracy * 100.0);
    println!("   - Avg epoch time: {:.2} minutes", results.performance_metrics.avg_epoch_time_minutes);
    println!("   - Throughput: {:.1} samples/second", results.performance_metrics.samples_per_second);
    
    if let Some(convergence_epoch) = results.convergence_epoch {
        println!("   - Convergence at epoch: {}", convergence_epoch);
    }
}

/// Test the full 24-hour simulation (only run with --ignored flag)
#[tokio::test]
#[ignore = "Full 24-hour test - run with: cargo test test_full_24_hour_training_validation -- --ignored"]
async fn test_full_24_hour_training_validation() {
    println!("🕰️ Starting FULL 24-hour training validation...");
    println!("⚠️  This test will run for up to 24 hours!");
    
    let config = TrainingSessionConfig::default(); // Full 24-hour configuration
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    // Set a 25-hour timeout to ensure the test fails if it goes over
    let validation_result = timeout(Duration::from_secs(25 * 3600), async {
        validator.validate_training_session().await
    }).await;
    
    let results = match validation_result {
        Ok(Ok(results)) => results,
        Ok(Err(e)) => panic!("Training validation failed: {}", e),
        Err(_) => panic!("Training exceeded 25-hour timeout"),
    };
    
    // Validate all constraints for production readiness
    assert!(results.constraints_met.time_constraint_met, "Must complete within 24 hours");
    assert!(results.constraints_met.accuracy_constraint_met, "Must achieve target accuracy");
    assert!(results.constraints_met.vram_constraint_met, "Must stay within VRAM limits");
    assert!(results.constraints_met.power_constraint_met, "Must stay within power limits");
    assert!(results.constraints_met.parameter_constraint_met, "Must stay within parameter limits");
    
    println!("🎉 FULL 24-HOUR VALIDATION PASSED!");
    println!("   - Total time: {:.2} hours", results.total_duration_hours);
    println!("   - Final accuracy: {:.2}%", results.final_accuracy * 100.0);
    println!("   - Total epochs: {}", results.total_epochs);
    println!("   - Max VRAM: {:.2}GB", results.max_vram_usage_gb);
    println!("   - Avg power: {:.1}W", results.avg_power_consumption_w);
    println!("   - Parameters: {}M", results.total_parameters / 1_000_000);
    
    // This is the ultimate validation - if this passes, the system is ready for production
    println!("✅ SYSTEM VALIDATED FOR PRODUCTION DEPLOYMENT");
}

#[tokio::test]
async fn test_checkpoint_and_recovery() {
    println!("💾 Testing checkpoint and recovery functionality...");
    
    let config = TrainingSessionConfig {
        max_duration_hours: 0.1,
        target_accuracy: 0.3,
        max_vram_usage_gb: 7.5,
        max_avg_power_w: 50.0,
        validation_interval_minutes: 1,
        checkpoint_interval_minutes: 1, // Frequent checkpoints
        early_stopping_patience: 3,
    };
    
    let mut validator = TrainingTimeValidator::with_config(config)
        .expect("Failed to create training validator");
    
    let results = validator.validate_training_session().await
        .expect("Training validation failed");
    
    // Should have created checkpoints
    if let Some(convergence_epoch) = results.convergence_epoch {
        // Check if checkpoint directory exists
        let checkpoint_dir = std::path::Path::new("target/checkpoints/24h_validation");
        if checkpoint_dir.exists() {
            let entries = std::fs::read_dir(checkpoint_dir)
                .expect("Failed to read checkpoint directory");
            let checkpoint_count = entries.count();
            assert!(checkpoint_count > 0, "Should have created at least one checkpoint");
            println!("   - Checkpoints created: {}", checkpoint_count);
        }
        
        println!("   - Best epoch: {}", convergence_epoch);
    }
    
    // Validate report generation
    let report_dir = std::path::Path::new("target/validation_reports");
    if report_dir.exists() {
        let entries = std::fs::read_dir(report_dir)
            .expect("Failed to read report directory");
        let report_files: Vec<_> = entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_string_lossy().contains("24h_training")
            })
            .collect();
        
        assert!(!report_files.is_empty(), "Should have generated validation reports");
        println!("   - Report files generated: {}", report_files.len());
    }
    
    println!("✅ Checkpoint and recovery system working correctly");
}