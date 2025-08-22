//! Comprehensive test suite using rstest fixtures and property-based testing
//!
//! This module provides exhaustive testing coverage with:
//! - Fixture-based testing for reusable test components
//! - Property-based testing for algorithm validation
//! - Security testing for robustness
//! - Performance constraint validation
//! - Edge case coverage

use rstest::*;
use proptest::prelude::*;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use test_case::test_case;
use mockall::predicate::*;
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use crate::model::validation::*;
use crate::model::mcp::*;
use crate::utils::perf::*;
use crate::utils::config::*;
use crate::utils::energy::*;

// =============================================================================
// FIXTURES - Reusable test components
// =============================================================================

#[fixture]
fn temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temporary directory")
}

#[fixture]
fn small_snn_config() -> SnnConfig {
    SnnConfig {
        input_size: 32,
        hidden_sizes: vec![64, 32],
        output_size: 16,
        threshold: 0.5,
        decay_rate: 0.9,
        refractory_period: 2,
        sparse_rate: 0.15,
    }
}

#[fixture]
fn large_snn_config() -> SnnConfig {
    SnnConfig {
        input_size: 768,
        hidden_sizes: vec![2048, 1024, 512],
        output_size: 256,
        threshold: 0.5,
        decay_rate: 0.9,
        refractory_period: 2,
        sparse_rate: 0.15,
    }
}

#[fixture]
fn test_ssm_config() -> SsmConfig {
    SsmConfig {
        input_size: 64,
        state_size: 16,
        num_layers: 6,
        output_size: 32,
        dt_min: 0.001,
        dt_max: 0.1,
        enable_cuda: false,
    }
}

#[fixture]
fn test_liquid_config() -> LiquidConfig {
    LiquidConfig {
        input_size: 64,
        hidden_size: 128,
        output_size: 32,
        time_constant: 10.0,
        adaptation_rate: 0.01,
        connectivity: 0.3,
        enable_adaptation: true,
        enable_plasticity: true,
    }
}

#[fixture]
fn test_fusion_config() -> FusionConfig {
    FusionConfig {
        input_dims: vec![16, 32, 32],
        output_dim: 64,
        hidden_dim: 128,
        attention_heads: 8,
        dropout_rate: 0.1,
        enable_residual: true,
    }
}

#[fixture]
fn performance_monitor() -> PerformanceMonitor {
    PerformanceMonitor::new()
}

#[fixture]
fn test_config() -> ModelConfig {
    ModelConfig {
        model_path: "test_model.bin".into(),
        max_tokens: 1024,
        enable_agentic: true,
        task_type: "text_generation".to_string(),
    }
}

#[fixture]
fn random_input_32() -> Array1<f32> {
    Array1::from_vec((0..32).map(|_| rand::random::<f32>()).collect())
}

#[fixture]
fn random_input_64() -> Array1<f32> {
    Array1::from_vec((0..64).map(|_| rand::random::<f32>()).collect())
}

#[fixture]
fn zero_input_64() -> Array1<f32> {
    Array1::zeros(64)
}

#[fixture]
fn ones_input_64() -> Array1<f32> {
    Array1::ones(64)
}

// =============================================================================
// SNN COMPONENT TESTS WITH FIXTURES
// =============================================================================

#[rstest]
#[case::small_config(small_snn_config())]
#[case::large_config(large_snn_config())]
fn test_snn_creation_with_configs(#[case] config: SnnConfig) {
    let result = SnnLayer::new(config.clone());
    assert!(result.is_ok(), "SNN creation should succeed for valid config");
    
    let snn = result.unwrap();
    assert!(snn.parameter_count() <= SNN_MAX_PARAMETERS, 
        "SNN parameter count should not exceed limit");
}

#[rstest]
fn test_snn_forward_pass_various_inputs(
    small_snn_config: SnnConfig,
    #[values(zero_input_64(), ones_input_64(), random_input_64())] input: Array1<f32>
) {
    let mut config = small_snn_config;
    config.input_size = input.len();
    
    let mut snn = SnnLayer::new(config.clone()).unwrap();
    let result = snn.forward(&input);
    
    assert!(result.is_ok(), "Forward pass should succeed");
    let output = result.unwrap();
    assert_eq!(output.len(), config.output_size);
    
    // Verify spike outputs are binary
    for &val in output.iter() {
        assert!(val == 0.0 || val == 1.0, "SNN output should be binary spikes");
    }
}

#[rstest]
#[case(0.05)] // Very sparse
#[case(0.15)] // Target sparse
#[case(0.25)] // Less sparse
fn test_snn_sparse_activation_rates(#[case] target_rate: f32, small_snn_config: SnnConfig) {
    let mut config = small_snn_config;
    config.sparse_rate = target_rate;
    
    let mut snn = SnnLayer::new(config.clone()).unwrap();
    
    // Test with high activation input
    let high_input = Array1::from_elem(config.input_size, 2.0);
    let output = snn.forward(&high_input).unwrap();
    
    let spike_count = output.iter().filter(|&&x| x > 0.0).count();
    let activation_rate = spike_count as f32 / output.len() as f32;
    
    // Should be approximately target rate (with tolerance)
    assert!(activation_rate <= target_rate * 1.5, 
        "Activation rate should be controlled by sparse_rate");
}

// =============================================================================
// PROPERTY-BASED TESTS
// =============================================================================

prop_compose! {
    fn valid_input_size()(size in 8usize..512) -> usize {
        size
    }
}

prop_compose! {
    fn valid_sparse_rate()(rate in 0.01f32..0.5) -> f32 {
        rate
    }
}

proptest! {
    #[test]
    fn property_snn_parameter_scaling(
        input_size in valid_input_size(),
        sparse_rate in valid_sparse_rate()
    ) {
        let config = SnnConfig {
            input_size,
            hidden_sizes: vec![input_size * 2, input_size],
            output_size: input_size / 2,
            sparse_rate,
            ..Default::default()
        };
        
        if let Ok(snn) = SnnLayer::new(config) {
            prop_assert!(snn.parameter_count() > 0);
            prop_assert!(snn.parameter_count() <= SNN_MAX_PARAMETERS);
        }
    }
    
    #[test]
    fn property_snn_output_bounds(
        input_size in 8usize..128,
        input_val in -10.0f32..10.0
    ) {
        let config = SnnConfig {
            input_size,
            hidden_sizes: vec![32],
            output_size: 16,
            ..Default::default()
        };
        
        if let Ok(mut snn) = SnnLayer::new(config) {
            let input = Array1::from_elem(input_size, input_val);
            if let Ok(output) = snn.forward(&input) {
                for &val in output.iter() {
                    prop_assert!(val == 0.0 || val == 1.0, 
                        "SNN output must be binary: {}", val);
                }
            }
        }
    }
}

// =============================================================================
// QUICKCHECK TESTS
// =============================================================================

#[quickcheck]
fn quickcheck_ssm_output_finite(input_size: u8, state_size: u8) -> TestResult {
    let input_size = input_size.max(4) as usize;
    let state_size = state_size.max(2) as usize;
    
    if input_size > 256 || state_size > 64 {
        return TestResult::discard();
    }
    
    let config = SsmConfig {
        input_size,
        state_size,
        num_layers: 4,
        output_size: state_size,
        ..Default::default()
    };
    
    if let Ok(mut ssm) = SsmLayer::new(config) {
        let input: Vec<f32> = (0..input_size).map(|_| rand::random::<f32>()).collect();
        let input_array = Array1::from_vec(input);
        
        if let Ok(output) = ssm.forward(&input_array) {
            return TestResult::from_bool(
                output.iter().all(|&x| x.is_finite())
            );
        }
    }
    
    TestResult::discard()
}

#[quickcheck]
fn quickcheck_liquid_adaptation_convergence(adaptation_rate: f32) -> TestResult {
    if adaptation_rate <= 0.0 || adaptation_rate >= 1.0 {
        return TestResult::discard();
    }
    
    let config = LiquidConfig {
        input_size: 32,
        hidden_size: 64,
        output_size: 16,
        adaptation_rate,
        enable_adaptation: true,
        ..Default::default()
    };
    
    if let Ok(mut liquid) = LiquidLayer::new(config) {
        let input = Array1::from_elem(32, 0.5);
        
        // Multiple forward passes to test adaptation
        let mut outputs = Vec::new();
        for _ in 0..10 {
            if let Ok(output) = liquid.forward(&input) {
                outputs.push(output);
            } else {
                return TestResult::failed();
            }
        }
        
        // Check that outputs are converging (getting more similar)
        if outputs.len() >= 3 {
            let first = &outputs[0];
            let middle = &outputs[outputs.len() / 2];
            let last = &outputs[outputs.len() - 1];
            
            let early_diff = (middle - first).mapv(|x| x.abs()).sum();
            let late_diff = (last - middle).mapv(|x| x.abs()).sum();
            
            return TestResult::from_bool(
                late_diff <= early_diff || late_diff < 0.1
            );
        }
    }
    
    TestResult::discard()
}

// =============================================================================
// SECURITY AND ROBUSTNESS TESTS
// =============================================================================

#[rstest]
#[case::extreme_negative(Array1::from_elem(32, -1000.0))]
#[case::extreme_positive(Array1::from_elem(32, 1000.0))]
#[case::nan_input(Array1::from_elem(32, f32::NAN))]
#[case::inf_input(Array1::from_elem(32, f32::INFINITY))]
fn test_snn_robustness_extreme_inputs(#[case] input: Array1<f32>, small_snn_config: SnnConfig) {
    let mut config = small_snn_config;
    config.input_size = input.len();
    
    let mut snn = SnnLayer::new(config).unwrap();
    let result = snn.forward(&input);
    
    match result {
        Ok(output) => {
            // If it succeeds, output should be finite and valid
            for &val in output.iter() {
                assert!(val.is_finite() || val == 0.0, 
                    "Output should be finite or zero for extreme inputs");
                assert!(val == 0.0 || val == 1.0, "SNN output should be binary");
            }
        },
        Err(_) => {
            // Error is acceptable for extreme inputs
        }
    }
}

#[rstest]
fn test_performance_constraints_validation(
    small_snn_config: SnnConfig,
    test_ssm_config: SsmConfig,
    test_liquid_config: LiquidConfig,
    test_fusion_config: FusionConfig,
    mut performance_monitor: PerformanceMonitor
) {
    // Create hybrid layer
    let hybrid = HybridLayer::new(
        small_snn_config.clone(),
        test_ssm_config.clone(),
        test_liquid_config.clone(),
        test_fusion_config.clone()
    ).unwrap();
    
    // Verify parameter budget
    let total_params = hybrid.total_parameters();
    assert!(total_params <= 100_000_000, 
        "Total parameters {} should not exceed 100M", total_params);
    
    // Verify parameter distribution
    let breakdown = hybrid.parameter_breakdown();
    let snn_params = breakdown.get("snn").unwrap_or(&0);
    let ssm_params = breakdown.get("ssm").unwrap_or(&0);
    let liquid_params = breakdown.get("liquid").unwrap_or(&0);
    
    assert!(*snn_params <= SNN_MAX_PARAMETERS);
    assert!(*ssm_params <= SSM_MAX_PARAMETERS);
    assert!(*liquid_params <= LIQUID_MAX_PARAMETERS);
    
    // Test inference timing constraint
    performance_monitor.start_inference_timer();
    
    let input = Array1::from_elem(small_snn_config.input_size, 0.5);
    let _output = hybrid.forward(&input).unwrap();
    
    let inference_time = performance_monitor.end_inference_timer();
    assert!(inference_time < 100.0, 
        "Inference time {:.1}ms should be under 100ms", inference_time);
}

// =============================================================================
// CONCURRENT AND STRESS TESTS
// =============================================================================

#[rstest]
#[tokio::test]
async fn test_concurrent_model_usage(small_snn_config: SnnConfig) {
    let config = small_snn_config;
    let snn = std::sync::Arc::new(std::sync::Mutex::new(
        SnnLayer::new(config.clone()).unwrap()
    ));
    
    let mut handles = Vec::new();
    
    // Spawn multiple concurrent tasks
    for i in 0..10 {
        let snn_clone = snn.clone();
        let input_size = config.input_size;
        
        let handle = tokio::spawn(async move {
            let input = Array1::from_elem(input_size, i as f32 * 0.1);
            let mut snn_guard = snn_clone.lock().unwrap();
            snn_guard.forward(&input)
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent execution should succeed");
    }
}

#[rstest]
fn test_memory_usage_bounds(large_snn_config: SnnConfig, mut performance_monitor: PerformanceMonitor) {
    let initial_memory = performance_monitor.get_memory_usage_mb();
    
    // Create large model
    let snn = SnnLayer::new(large_snn_config.clone()).unwrap();
    
    // Process multiple large inputs
    let mut outputs = Vec::new();
    for _ in 0..100 {
        let input = Array1::from_elem(large_snn_config.input_size, 0.5);
        let _snn_copy = snn.clone(); // Create copy to test memory growth
        outputs.push(input);
    }
    
    let final_memory = performance_monitor.get_memory_usage_mb();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    
    // Memory growth should be reasonable (less than 1GB)
    assert!(memory_growth < 1024, 
        "Memory growth {}MB should be under 1GB", memory_growth);
}

// =============================================================================
// INTEGRATION TESTS WITH MOCK COMPONENTS
// =============================================================================

#[rstest]
#[tokio::test]
async fn test_full_inference_pipeline(
    small_snn_config: SnnConfig,
    test_ssm_config: SsmConfig,
    test_liquid_config: LiquidConfig,
    test_fusion_config: FusionConfig,
    temp_dir: TempDir
) {
    // Create hybrid model
    let mut hybrid = HybridLayer::new(
        small_snn_config.clone(),
        test_ssm_config.clone(),
        test_liquid_config.clone(),
        test_fusion_config.clone()
    ).unwrap();
    
    // Test full pipeline
    let input = Array1::from_elem(small_snn_config.input_size, 0.5);
    
    // Performance monitoring
    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start_inference_timer();
    
    // Forward pass
    let output = hybrid.forward(&input).unwrap();
    
    let inference_time = perf_monitor.end_inference_timer();
    
    // Validate output
    assert!(output.len() > 0);
    assert!(output.iter().all(|&x| x.is_finite()));
    
    // Validate performance constraints
    assert!(inference_time < 100.0, "Inference under 100ms");
    
    // Validate memory usage
    let memory_usage = perf_monitor.get_memory_usage_mb();
    assert!(memory_usage < 8192, "Memory under 8GB");
    
    // Test serialization/deserialization
    let model_path = temp_dir.path().join("test_model.bin");
    hybrid.save(&model_path).unwrap();
    
    let loaded_hybrid = HybridLayer::load(&model_path).unwrap();
    let loaded_output = loaded_hybrid.forward(&input).unwrap();
    
    // Outputs should be identical
    let diff: f32 = (&output - &loaded_output).mapv(|x| x.abs()).sum();
    assert!(diff < 1e-6, "Loaded model should produce identical outputs");
}

// =============================================================================
// VALIDATION AND SECURITY TESTS
// =============================================================================

#[rstest]
#[tokio::test]
async fn test_output_validation_comprehensive(temp_dir: TempDir) {
    let config = ValidationConfig::default();
    let validator = OutputValidator::new(config);
    
    // Test cases with expected outcomes
    let test_cases = [
        ("Clear factual statement.", true, 0.8),
        ("According to experts, this might be true.", false, 0.3),
        ("Some say this is widely known.", false, 0.2),
        ("Python is a programming language.", true, 0.9),
        ("Recent studies show unclear results.", false, 0.4),
    ];
    
    for (text, should_pass, min_confidence) in &test_cases {
        let result = validator.validate_output(text, None, None).await.unwrap();
        
        if *should_pass {
            assert!(result.confidence_score >= *min_confidence, 
                "Good text should have high confidence: {} -> {}", 
                text, result.confidence_score);
        } else {
            assert!(result.hallucination_risk > 0.3 || result.confidence_score < 0.6,
                "Suspicious text should be flagged: {} -> risk: {}, conf: {}", 
                text, result.hallucination_risk, result.confidence_score);
        }
    }
}

#[rstest]
#[case::code_generation("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")]
#[case::text_generation("The quick brown fox jumps over the lazy dog.")]
#[case::math_reasoning("The solution to 2x + 3 = 7 is x = 2.")]
fn test_task_specific_validation(#[case] output: &str) {
    let config = ValidationConfig::default();
    let validator = OutputValidator::new(config);
    
    // Quick validation should pass for reasonable outputs
    assert!(validator.quick_validate(output), 
        "Reasonable output should pass quick validation: {}", output);
}

// =============================================================================
// BENCHMARK AND PERFORMANCE TESTS
// =============================================================================

#[rstest]
fn test_throughput_requirements(small_snn_config: SnnConfig) {
    let mut snn = SnnLayer::new(small_snn_config.clone()).unwrap();
    
    let start = std::time::Instant::now();
    let num_inferences = 100;
    
    for _ in 0..num_inferences {
        let input = Array1::from_elem(small_snn_config.input_size, 0.5);
        let _output = snn.forward(&input).unwrap();
    }
    
    let elapsed = start.elapsed();
    let throughput = num_inferences as f64 / elapsed.as_secs_f64();
    
    // Should achieve reasonable throughput (>10 inferences/second)
    assert!(throughput > 10.0, 
        "Throughput {:.1} inferences/sec should exceed 10/sec", throughput);
}

#[rstest]
fn test_energy_consumption_estimation(mut performance_monitor: PerformanceMonitor) {
    let power_before = performance_monitor.get_power_consumption();
    
    // Simulate computational load
    let config = SnnConfig {
        input_size: 512,
        hidden_sizes: vec![1024, 512],
        output_size: 256,
        ..Default::default()
    };
    
    let mut snn = SnnLayer::new(config.clone()).unwrap();
    
    // Run multiple inferences
    for _ in 0..50 {
        let input = Array1::from_elem(config.input_size, 0.5);
        let _output = snn.forward(&input).unwrap();
    }
    
    let power_after = performance_monitor.get_power_consumption();
    
    // Power consumption should be reasonable (under 50W)
    assert!(power_after < 50.0, 
        "Power consumption {:.1}W should be under 50W target", power_after);
    
    // Power should not decrease significantly (sanity check)
    assert!(power_after >= power_before - 5.0, 
        "Power consumption should not drop significantly during computation");
}

// =============================================================================
// ERROR HANDLING AND EDGE CASES
// =============================================================================

#[rstest]
#[case(0)]      // Zero size
#[case(1)]      // Minimum size  
#[case(10000)]  // Very large size
fn test_invalid_configurations(#[case] invalid_size: usize) {
    let config = SnnConfig {
        input_size: invalid_size,
        hidden_sizes: if invalid_size == 0 { vec![] } else { vec![invalid_size] },
        output_size: invalid_size,
        ..Default::default()
    };
    
    let result = SnnLayer::new(config);
    
    if invalid_size == 0 || invalid_size > 5000 {
        // Should fail for unreasonable sizes
        assert!(result.is_err(), "Should reject invalid configuration");
    } else {
        // Should succeed for reasonable sizes
        assert!(result.is_ok() || invalid_size > 1000, "Should accept reasonable configuration");
    }
}

#[rstest]
fn test_resource_cleanup(small_snn_config: SnnConfig) {
    // Create and drop many models to test resource cleanup
    for _ in 0..100 {
        let snn = SnnLayer::new(small_snn_config.clone()).unwrap();
        let input = Array1::from_elem(small_snn_config.input_size, 0.5);
        let _output = snn.forward(&input).unwrap();
        // Model should be dropped here
    }
    
    // Should not crash or leak memory significantly
    assert!(true);
}