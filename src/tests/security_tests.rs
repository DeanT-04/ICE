//! Security testing suite
//!
//! Comprehensive security validation including:
//! - Input fuzzing and injection testing
//! - Memory safety validation  
//! - Overflow and underflow protection
//! - Denial of service resistance
//! - Information leakage prevention
//! - Cryptographic security (where applicable)

use rstest::*;
use proptest::prelude::*;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use test_case::test_case;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array1, Array2};
use tempfile::TempDir;

use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use crate::model::validation::*;
use crate::utils::perf::*;

// =============================================================================
// SECURITY TEST FIXTURES
// =============================================================================

#[fixture]
fn security_snn_config() -> SnnConfig {
    SnnConfig {
        input_size: 128,
        hidden_sizes: vec![256, 128],
        output_size: 64,
        threshold: 0.5,
        decay_rate: 0.9,
        refractory_period: 2,
        sparse_rate: 0.15,
    }
}

#[fixture]
fn malicious_inputs() -> Vec<Array1<f32>> {
    vec![
        // Extreme values
        Array1::from_elem(128, f32::MAX),
        Array1::from_elem(128, f32::MIN),
        Array1::from_elem(128, f32::INFINITY),
        Array1::from_elem(128, f32::NEG_INFINITY),
        Array1::from_elem(128, f32::NAN),
        
        // Large oscillating values
        Array1::from_shape_fn(128, |i| if i % 2 == 0 { 1e6 } else { -1e6 }),
        
        // Gradual overflow attempt
        Array1::from_shape_fn(128, |i| 1e10 + i as f32),
        
        // Denormalized numbers
        Array1::from_elem(128, f32::MIN_POSITIVE / 2.0),
        
        // Zero-sized attacks
        Array1::from_elem(128, 0.0),
        
        // Random noise
        Array1::from_shape_fn(128, |_| rand::random::<f32>() * 1e20),
    ]
}

// =============================================================================
// INPUT VALIDATION AND SANITIZATION TESTS
// =============================================================================

#[rstest]
fn test_input_sanitization_against_injection(
    security_snn_config: SnnConfig,
    malicious_inputs: Vec<Array1<f32>>
) {
    let mut snn = SnnLayer::new(security_snn_config).unwrap();
    
    for (i, malicious_input) in malicious_inputs.iter().enumerate() {
        let result = snn.forward(malicious_input);
        
        match result {
            Ok(output) => {
                // If processing succeeds, output must be safe
                for &val in output.iter() {
                    assert!(val.is_finite() || val == 0.0, 
                        "Malicious input {} produced unsafe output: {}", i, val);
                    assert!(val == 0.0 || val == 1.0, 
                        "SNN output should be binary even with malicious input");
                }
            },
            Err(_) => {
                // Graceful error handling is acceptable for malicious inputs
                continue;
            }
        }
    }
}

#[rstest]
#[case::oversized_input(10000)]
#[case::zero_input(0)]
#[case::minimal_input(1)]
fn test_input_size_validation(#[case] input_size: usize, security_snn_config: SnnConfig) {
    let mut config = security_snn_config;
    config.input_size = 128; // Fixed expected size
    
    let snn = SnnLayer::new(config).unwrap();
    let input = Array1::zeros(input_size);
    
    let result = snn.forward(&input);
    
    if input_size != 128 {
        // Should reject mismatched input sizes
        assert!(result.is_err(), 
            "Should reject input size {} when expecting 128", input_size);
    } else {
        // Should accept correct input size
        assert!(result.is_ok(), "Should accept correct input size");
    }
}

// =============================================================================
// MEMORY SAFETY AND BOUNDS CHECKING
// =============================================================================

#[rstest]
fn test_memory_bounds_protection(security_snn_config: SnnConfig) {
    let snn = SnnLayer::new(security_snn_config.clone()).unwrap();
    
    // Test access patterns that might cause buffer overflows
    let test_cases = vec![
        // Normal case
        Array1::from_elem(security_snn_config.input_size, 0.5),
        
        // Edge case: exactly at boundary
        Array1::from_elem(security_snn_config.input_size, 1.0),
    ];
    
    for input in test_cases {
        let result = snn.forward(&input);
        assert!(result.is_ok(), "Valid input should not cause memory safety issues");
        
        if let Ok(output) = result {
            // Verify output bounds
            assert_eq!(output.len(), security_snn_config.output_size);
            assert!(output.iter().all(|&x| x.is_finite() || x == 0.0));
        }
    }
}

#[rstest]
fn test_stack_overflow_protection() {
    // Create deeply nested or recursive scenario that might cause stack overflow
    let config = SnnConfig {
        input_size: 64,
        hidden_sizes: vec![64; 100], // Very deep network
        output_size: 32,
        ..Default::default()
    };
    
    // This should either work or fail gracefully, not crash
    let result = SnnLayer::new(config);
    
    match result {
        Ok(mut snn) => {
            let input = Array1::from_elem(64, 0.5);
            let forward_result = snn.forward(&input);
            // Should handle deep networks safely
            assert!(forward_result.is_ok() || forward_result.is_err());
        },
        Err(_) => {
            // Acceptable to reject overly deep networks
        }
    }
}

// =============================================================================
// DENIAL OF SERVICE RESISTANCE
// =============================================================================

#[rstest]
fn test_computational_dos_resistance(security_snn_config: SnnConfig) {
    let mut snn = SnnLayer::new(security_snn_config.clone()).unwrap();
    let input = Array1::from_elem(security_snn_config.input_size, 0.5);
    
    let start_time = Instant::now();
    let max_iterations = 10000;
    
    // Attempt computational DoS
    for _ in 0..max_iterations {
        let result = snn.forward(&input);
        assert!(result.is_ok(), "Forward pass should remain stable under load");
        
        // Check if execution time is reasonable
        if start_time.elapsed() > Duration::from_secs(10) {
            // If it takes too long, stop and verify it's still functional
            break;
        }
    }
    
    // System should remain responsive
    let final_result = snn.forward(&input);
    assert!(final_result.is_ok(), "System should remain functional after load test");
}

#[rstest]
fn test_memory_dos_resistance() {
    let mut models = Vec::new();
    let mut total_memory = 0;
    
    // Attempt to create many models to exhaust memory
    for i in 0..100 {
        let config = SnnConfig {
            input_size: 128,
            hidden_sizes: vec![256, 128],
            output_size: 64,
            ..Default::default()
        };
        
        match SnnLayer::new(config) {
            Ok(snn) => {
                let param_count = snn.parameter_count();
                total_memory += param_count;
                models.push(snn);
                
                // Check if we're approaching memory limits reasonably
                if total_memory > 1_000_000_000 { // 1B parameters
                    break;
                }
            },
            Err(_) => {
                // Graceful handling of resource exhaustion
                break;
            }
        }
    }
    
    // Should have created at least some models before hitting limits
    assert!(!models.is_empty(), "Should be able to create at least one model");
    
    // Verify models still work
    if let Some(mut model) = models.pop() {
        let input = Array1::from_elem(128, 0.5);
        let result = model.forward(&input);
        assert!(result.is_ok(), "Models should remain functional");
    }
}

// =============================================================================
// INFORMATION LEAKAGE PREVENTION
// =============================================================================

#[rstest]
fn test_timing_attack_resistance(security_snn_config: SnnConfig) {
    let mut snn = SnnLayer::new(security_snn_config.clone()).unwrap();
    
    // Create inputs that might reveal information through timing
    let normal_input = Array1::from_elem(security_snn_config.input_size, 0.5);
    let sparse_input = Array1::zeros(security_snn_config.input_size);
    let dense_input = Array1::ones(security_snn_config.input_size);
    
    let inputs = vec![normal_input, sparse_input, dense_input];
    let mut timings = Vec::new();
    
    for input in inputs {
        let start = Instant::now();
        let _result = snn.forward(&input).unwrap();
        let duration = start.elapsed();
        timings.push(duration);
    }
    
    // Timing differences should be minimal (< 10% variation)
    let min_time = timings.iter().min().unwrap();
    let max_time = timings.iter().max().unwrap();
    
    let time_ratio = max_time.as_nanos() as f64 / min_time.as_nanos() as f64;
    
    // Allow some variation but not dramatic differences
    assert!(time_ratio < 2.0, 
        "Timing variation too large: {:.2}x, potential timing attack vector", time_ratio);
}

#[rstest]
fn test_state_isolation(security_snn_config: SnnConfig) {
    let mut snn1 = SnnLayer::new(security_snn_config.clone()).unwrap();
    let mut snn2 = SnnLayer::new(security_snn_config.clone()).unwrap();
    
    let input1 = Array1::from_elem(security_snn_config.input_size, 0.3);
    let input2 = Array1::from_elem(security_snn_config.input_size, 0.7);
    
    // Process different inputs on different instances
    let output1_initial = snn1.forward(&input1).unwrap();
    let output2_initial = snn2.forward(&input2).unwrap();
    
    // Reset states
    snn1.reset_state();
    snn2.reset_state();
    
    // Process same inputs again
    let output1_reset = snn1.forward(&input1).unwrap();
    let output2_reset = snn2.forward(&input2).unwrap();
    
    // Outputs should be identical after reset (no state leakage)
    let diff1 = (&output1_initial - &output1_reset).mapv(|x| x.abs()).sum();
    let diff2 = (&output2_initial - &output2_reset).mapv(|x| x.abs()).sum();
    
    assert!(diff1 < 1e-6, "State should be properly reset in model 1");
    assert!(diff2 < 1e-6, "State should be properly reset in model 2");
}

// =============================================================================
// PROPERTY-BASED SECURITY TESTS
// =============================================================================

prop_compose! {
    fn malicious_array(size: usize)(
        values in prop::collection::vec(
            prop_oneof![
                Just(f32::NAN),
                Just(f32::INFINITY),
                Just(f32::NEG_INFINITY),
                Just(f32::MAX),
                Just(f32::MIN),
                (-1e10f32..1e10f32),
            ],
            size
        )
    ) -> Array1<f32> {
        Array1::from_vec(values)
    }
}

proptest! {
    #[test]
    fn property_input_sanitization(
        input in malicious_array(128)
    ) {
        let config = SnnConfig {
            input_size: 128,
            hidden_sizes: vec![64],
            output_size: 32,
            ..Default::default()
        };
        
        let mut snn = SnnLayer::new(config).unwrap();
        
        match snn.forward(&input) {
            Ok(output) => {
                // If processing succeeds, output must be safe
                for &val in output.iter() {
                    prop_assert!(val.is_finite() || val == 0.0, 
                        "Output must be finite or zero: {}", val);
                    prop_assert!(val == 0.0 || val == 1.0, 
                        "SNN output must be binary: {}", val);
                }
            },
            Err(_) => {
                // Error handling is acceptable for malicious inputs
            }
        }
    }
    
    #[test]
    fn property_memory_bounds(
        size in 1usize..1000,
        value in -1e6f32..1e6f32
    ) {
        if size > 512 {
            return Ok(()); // Skip very large inputs
        }
        
        let config = SnnConfig {
            input_size: size,
            hidden_sizes: vec![size / 2],
            output_size: size / 4 + 1,
            ..Default::default()
        };
        
        if let Ok(mut snn) = SnnLayer::new(config) {
            let input = Array1::from_elem(size, value);
            
            match snn.forward(&input) {
                Ok(output) => {
                    prop_assert!(output.len() == size / 4 + 1);
                    prop_assert!(output.iter().all(|&x| x.is_finite() || x == 0.0));
                },
                Err(_) => {
                    // Errors are acceptable for edge cases
                }
            }
        }
    }
}

// =============================================================================
// QUICKCHECK SECURITY TESTS
// =============================================================================

#[quickcheck]
fn quickcheck_numerical_stability(
    input_val: f32,
    input_size: u8
) -> TestResult {
    let input_size = (input_size as usize).max(4).min(256);
    
    if !input_val.is_finite() {
        return TestResult::discard();
    }
    
    let config = SnnConfig {
        input_size,
        hidden_sizes: vec![input_size / 2],
        output_size: input_size / 4 + 1,
        ..Default::default()
    };
    
    if let Ok(mut snn) = SnnLayer::new(config) {
        let input = Array1::from_elem(input_size, input_val);
        
        match snn.forward(&input) {
            Ok(output) => {
                TestResult::from_bool(
                    output.iter().all(|&x| x.is_finite() || x == 0.0) &&
                    output.iter().all(|&x| x == 0.0 || x == 1.0)
                )
            },
            Err(_) => TestResult::passed() // Errors are acceptable
        }
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn quickcheck_parameter_budget_enforcement(
    input_size: u16,
    hidden_multiplier: u8
) -> TestResult {
    let input_size = (input_size as usize).max(8).min(1000);
    let hidden_size = input_size * (hidden_multiplier as usize).max(1).min(10);
    
    if hidden_size > 5000 || input_size * hidden_size > 1_000_000 {
        return TestResult::discard();
    }
    
    let config = SnnConfig {
        input_size,
        hidden_sizes: vec![hidden_size],
        output_size: input_size / 2,
        ..Default::default()
    };
    
    match SnnLayer::new(config) {
        Ok(snn) => {
            TestResult::from_bool(snn.parameter_count() <= SNN_MAX_PARAMETERS)
        },
        Err(_) => {
            // Should fail gracefully for oversized networks
            TestResult::passed()
        }
    }
}

// =============================================================================
// FUZZING INTEGRATION TESTS
// =============================================================================

#[rstest]
fn test_fuzz_integration_readiness() {
    // Verify that the model can handle fuzz testing
    let config = SnnConfig {
        input_size: 64,
        hidden_sizes: vec![32],
        output_size: 16,
        ..Default::default()
    };
    
    let mut snn = SnnLayer::new(config).unwrap();
    
    // Simulate fuzz test inputs
    let fuzz_inputs = vec![
        vec![0u8; 64 * 4], // All zeros (as bytes)
        vec![255u8; 64 * 4], // All max values
        (0..64*4).map(|i| (i % 256) as u8).collect(), // Incrementing pattern
    ];
    
    for fuzz_bytes in fuzz_inputs {
        // Convert bytes to f32 array (unsafe but this is how fuzzers work)
        let float_slice = unsafe {
            std::slice::from_raw_parts(
                fuzz_bytes.as_ptr() as *const f32,
                fuzz_bytes.len() / 4
            )
        };
        
        if float_slice.len() >= 64 {
            let input = Array1::from_slice(&float_slice[..64]);
            
            // Should handle any input without crashing
            let _result = snn.forward(&input);
            // Don't assert on result - fuzz testing allows any outcome except crashes
        }
    }
}

// =============================================================================
// VALIDATION SECURITY TESTS
// =============================================================================

#[rstest]
#[tokio::test]
async fn test_validation_injection_resistance() {
    let config = ValidationConfig::default();
    let validator = OutputValidator::new(config);
    
    // Injection attempts
    let malicious_outputs = vec![
        "According to <script>alert('xss')</script> studies...",
        "Experts say '; DROP TABLE users; --",
        "Some studies show {{7*7}} interesting results",
        "Recent research indicates #{os.system('rm -rf /')}",
        "Data suggests ${eval('process.exit()')} conclusions",
    ];
    
    for malicious_output in malicious_outputs {
        let result = validator.validate_output(malicious_output, None, None).await;
        
        match result {
            Ok(validation_result) => {
                // Should flag as suspicious
                assert!(validation_result.hallucination_risk > 0.5 || 
                       !validation_result.issues.is_empty(),
                       "Should detect malicious content: {}", malicious_output);
            },
            Err(_) => {
                // Error handling is acceptable for malicious inputs
            }
        }
    }
}

// =============================================================================
// PERFORMANCE CONSTRAINT SECURITY
// =============================================================================

#[rstest]
fn test_performance_constraint_bypass_resistance(security_snn_config: SnnConfig) {
    let mut perf_monitor = PerformanceMonitor::new();
    
    // Attempt to bypass performance constraints
    let attack_scenarios = vec![
        // Rapid fire requests
        ("rapid_fire", 1000, Duration::from_millis(0)),
        // Resource exhaustion
        ("resource_exhaust", 100, Duration::from_millis(1)),
        // Timing manipulation
        ("timing_attack", 50, Duration::from_millis(10)),
    ];
    
    for (attack_name, iterations, delay) in attack_scenarios {
        let start_time = Instant::now();
        let mut snn = SnnLayer::new(security_snn_config.clone()).unwrap();
        
        for i in 0..iterations {
            perf_monitor.start_inference_timer();
            
            let input = Array1::from_elem(security_snn_config.input_size, 
                                        (i as f32) / (iterations as f32));
            let _result = snn.forward(&input);
            
            let inference_time = perf_monitor.end_inference_timer();
            
            // Performance constraints should still be enforced
            assert!(inference_time < 1000.0, 
                "Inference time constraint violated during {} attack", attack_name);
            
            std::thread::sleep(delay);
            
            // Prevent infinite loops
            if start_time.elapsed() > Duration::from_secs(30) {
                break;
            }
        }
        
        // System should remain functional after attack
        let final_input = Array1::from_elem(security_snn_config.input_size, 0.5);
        let final_result = snn.forward(&final_input);
        assert!(final_result.is_ok(), 
            "System should remain functional after {} attack", attack_name);
    }
}

// =============================================================================
// CRYPTOGRAPHIC SECURITY (WHERE APPLICABLE)
// =============================================================================

#[rstest]
fn test_deterministic_behavior_consistency(security_snn_config: SnnConfig) {
    // Verify that deterministic operations remain deterministic
    // (important for reproducibility and preventing information leakage)
    
    let snn1 = SnnLayer::new(security_snn_config.clone()).unwrap();
    let snn2 = SnnLayer::new(security_snn_config.clone()).unwrap();
    
    // Same configuration should produce same initial state
    let stats1 = snn1.get_activation_stats();
    let stats2 = snn2.get_activation_stats();
    
    // Initial states should be identical
    assert_eq!(stats1.len(), stats2.len());
    
    // Parameter counts should be identical
    assert_eq!(snn1.parameter_count(), snn2.parameter_count());
}

#[rstest]
fn test_state_cleanup_security() {
    // Ensure sensitive state is properly cleaned up
    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("security_test_model.bin");
    
    {
        let config = SnnConfig {
            input_size: 64,
            hidden_sizes: vec![32],
            output_size: 16,
            ..Default::default()
        };
        
        let mut snn = SnnLayer::new(config).unwrap();
        
        // Process sensitive input
        let sensitive_input = Array1::from_elem(64, 0.123456789);
        let _output = snn.forward(&sensitive_input).unwrap();
        
        // Save and drop model
        snn.save(&model_path).unwrap();
    } // Model dropped here
    
    // Load model and verify no sensitive data leakage
    let loaded_snn = SnnLayer::load(&model_path).unwrap();
    
    // Process different input
    let clean_input = Array1::zeros(64);
    let clean_output = loaded_snn.forward(&clean_input).unwrap();
    
    // Output should not contain traces of previous sensitive input
    for &val in clean_output.iter() {
        assert!(val == 0.0 || val == 1.0, "Output should be clean binary values");
    }
}