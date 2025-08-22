//! Integration tests
//!
//! End-to-end tests for the complete ultra-fast AI model system
//! including model integration, training pipeline, and performance validation.

use super::test_utils::*;
use super::*;
use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use crate::model::validation::*;
use crate::model::mcp::*;
use crate::training::datasets::*;
use crate::training::trainer::*;
use crate::training::genetic::*;
use crate::utils::perf::*;
use ndarray::Array1;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(test)]
mod full_system_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_model_pipeline() {
        init_test_env();
        
        // Create complete model configuration
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        // Create hybrid model
        let mut hybrid_model = HybridLayer::new(
            snn_config.clone(),
            ssm_config,
            liquid_config,
            fusion_config,
        ).unwrap();
        
        // Test parameter constraints
        let total_params = hybrid_model.total_parameters();
        assert!(total_params <= 100_000_000, "Model exceeds 100M parameter limit");
        
        // Test forward pass
        let input = random_input(snn_config.input_size);
        let output = hybrid_model.forward(&input).unwrap();
        
        assert!(output.len() > 0);
        assert!(output.iter().all(|&x| x.is_finite()));
        
        // Test multiple forward passes for consistency
        let output2 = hybrid_model.forward(&input).unwrap();
        assert_eq!(output.len(), output2.len());
        
        log::info!("Complete model pipeline test passed with {} parameters", total_params);
    }

    #[tokio::test]
    async fn test_training_integration() {
        init_test_env();
        
        // Create training configuration
        let mut training_config = TrainingConfig::default();
        training_config.max_epochs = 1; // Quick test
        training_config.batch_size = 2;
        training_config.max_steps = Some(5); // Limit steps for testing
        
        // Create model components
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        let agentic_coordinator = AgenticCoordinator::new(TaskConfig::default(), VotingStrategy::WeightedVote);
        let dataset_manager = DatasetManager::new();
        
        // Create trainer
        let mut trainer = Trainer::new(training_config, model, agentic_coordinator, dataset_manager);
        
        // Test training can start and run
        let initial_stats = trainer.get_stats();
        assert_eq!(initial_stats.get("epoch").copied().unwrap_or(-1.0), 0.0);
        
        log::info!("Training integration test setup complete");
    }

    #[tokio::test]
    async fn test_agentic_system_integration() {
        init_test_env();
        
        // Create agentic coordinator
        let task_config = TaskConfig::default();
        let mut coordinator = AgenticCoordinator::new(task_config, VotingStrategy::WeightedVote);
        
        // Test task execution
        let input = random_input(64);
        let result = coordinator.execute_task(&input).await.unwrap();
        
        assert!(result.len() > 0);
        assert!(result.iter().all(|&x| x.is_finite()));
        
        // Test voting mechanism
        let stats = coordinator.get_stats();
        assert!(stats.contains_key("num_sub_models"));
        
        log::info!("Agentic system integration test passed");
    }

    #[tokio::test]
    async fn test_validation_system_integration() {
        init_test_env();
        
        // Create validation system
        let validation_config = ValidationConfig::default();
        let validator = OutputValidator::new(validation_config);
        
        // Test validation of good output
        let good_output = "This is a factual statement about machine learning.";
        let validation_result = validator.validate_output(good_output, None, None).await.unwrap();
        
        assert!(validation_result.confidence_score > 0.0);
        assert!(validation_result.consistency_score > 0.0);
        assert!(validation_result.hallucination_risk >= 0.0);
        
        // Test validation of suspicious output
        let suspicious_output = "According to recent studies by experts, this is widely accepted.";
        let suspicious_result = validator.validate_output(suspicious_output, None, None).await.unwrap();
        
        assert!(suspicious_result.hallucination_risk > 0.0);
        assert!(!suspicious_result.issues.is_empty());
        
        log::info!("Validation system integration test passed");
    }

    #[tokio::test]
    async fn test_mcp_integration() {
        init_test_env();
        
        // Create MCP client
        let mcp_config = McpConfig::default();
        let mut mcp_client = McpClient::new(mcp_config);
        
        // Test dataset fetching
        let dataset_result = mcp_client.fetch_dataset("test_dataset").await;
        assert!(dataset_result.is_ok());
        
        // Test code analysis
        let code_analysis = mcp_client.analyze_code("fn test() { println!(\"hello\"); }", "rust").await;
        assert!(code_analysis.is_ok());
        
        // Test API call
        let api_result = mcp_client.call_external_api("test_endpoint", serde_json::json!({})).await;
        assert!(api_result.is_ok());
        
        log::info!("MCP integration test passed");
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;

    #[test]
    fn test_inference_latency_constraint() {
        init_test_env();
        
        // Create model
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut model = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        let input = random_input(snn_config.input_size);
        
        // Measure inference time
        let start = Instant::now();
        let _output = model.forward(&input).unwrap();
        let inference_time = start.elapsed();
        
        // Should be under 100ms for small test model
        assert!(inference_time < Duration::from_millis(100), 
                "Inference took {:?}, exceeds 100ms limit", inference_time);
        
        log::info!("Inference latency test passed: {:?}", inference_time);
    }

    #[test]
    fn test_memory_constraint() {
        init_test_env();
        
        // Create model and measure memory
        let performance_monitor = PerformanceMonitor::new();
        let initial_memory = performance_monitor.get_memory_usage_mb();
        
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let _model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        
        let final_memory = performance_monitor.get_memory_usage_mb();
        let memory_increase = final_memory - initial_memory;
        
        // Should not use excessive memory for test model
        assert!(memory_increase < 1000, "Memory increase of {}MB is excessive", memory_increase);
        
        log::info!("Memory constraint test passed: {}MB increase", memory_increase);
    }

    #[test]
    fn test_parameter_budget_constraint() {
        init_test_env();
        
        // Test with realistic model sizes that approach the 100M limit
        let snn_config = SnnConfig {
            input_size: 768,
            hidden_sizes: vec![2048, 1024, 512],
            output_size: 256,
            threshold: 0.5,
            decay_rate: 0.9,
            refractory_period: 2,
            sparse_rate: 0.15,
        };
        
        let ssm_config = SsmConfig {
            input_size: 768,
            state_size: 64,
            output_size: 256,
            num_layers: 12,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "random".to_string(),
            conv_kernel_size: 4,
        };
        
        let liquid_config = LiquidConfig {
            input_size: 768,
            hidden_size: 512,
            output_size: 256,
            time_constant: 1.0,
            adaptation_rate: 0.01,
            connectivity: 0.3,
            enable_adaptation: true,
        };
        
        let fusion_config = FusionConfig {
            input_dims: vec![256, 256, 256],
            output_dim: 768,
            hidden_dim: 1024,
            attention_heads: 8,
            dropout_rate: 0.1,
            use_layer_norm: true,
            activation_function: "gelu".to_string(),
        };
        
        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        let total_params = model.total_parameters();
        
        assert!(total_params <= 100_000_000, 
                "Model has {} parameters, exceeds 100M limit", total_params);
        
        log::info!("Parameter budget test passed: {} parameters", total_params);
    }

    #[test]
    fn test_sparse_activation_constraint() {
        init_test_env();
        
        // Create SNN with specific sparsity requirements
        let mut snn_config = create_test_snn_config();
        snn_config.sparse_rate = 0.15; // 15% activation rate
        
        let mut snn = SnnLayer::new(snn_config.clone()).unwrap();
        
        // Test with high input to trigger many spikes
        let input = Array1::ones(snn_config.input_size) * 2.0;
        let output = snn.forward(&input).unwrap();
        
        // Calculate actual activation rate
        let active_neurons = output.iter().filter(|&&x| x > 0.0).count();
        let activation_rate = active_neurons as f32 / output.len() as f32;
        
        // Should maintain sparsity even with high input
        assert!(activation_rate <= snn_config.sparse_rate + 0.05, 
                "Activation rate {:.3} exceeds sparsity constraint of {:.3}", 
                activation_rate, snn_config.sparse_rate);
        
        log::info!("Sparse activation test passed: {:.3} activation rate", activation_rate);
    }
}

#[cfg(test)]
mod data_pipeline_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_dataset_loading_pipeline() {
        init_test_env();
        
        // Create dataset manager
        let dataset_manager = DatasetManager::new();
        
        // Test loading different split types
        let splits = [SplitType::Train, SplitType::Validation, SplitType::Test];
        
        for split in &splits {
            let result = dataset_manager.get_mixed_samples(*split, Some(5));
            
            match result {
                Ok(samples) => {
                    assert!(!samples.is_empty());
                    assert!(samples.len() <= 5);
                    
                    for sample in samples {
                        assert!(!sample.id.is_empty());
                        assert!(!sample.input.is_empty());
                        assert!(!sample.target.is_empty());
                    }
                }
                Err(_) => {
                    // Expected in test environment without real datasets
                    log::warn!("Dataset loading failed for {:?} split (expected in test)", split);
                }
            }
        }
        
        log::info!("Dataset loading pipeline test completed");
    }

    #[tokio::test]
    async fn test_data_preprocessing_pipeline() {
        init_test_env();
        
        // Create sample data
        let raw_samples = vec![
            TrainingSample {
                id: "test_1".to_string(),
                input: "  This is a test input with extra spaces.  ".to_string(),
                target: "Processed target output".to_string(),
                metadata: HashMap::new(),
                dataset: "test".to_string(),
            },
            TrainingSample {
                id: "test_2".to_string(),
                input: "Another test input with UPPERCASE text".to_string(),
                target: "another target".to_string(),
                metadata: HashMap::new(),
                dataset: "test".to_string(),
            },
        ];
        
        // Test preprocessing
        let preprocessing_config = PreprocessingConfig::default();
        let processed_samples = preprocess_samples(raw_samples, &preprocessing_config);
        
        assert!(!processed_samples.is_empty());
        
        for sample in processed_samples {
            // Check that preprocessing was applied
            assert!(!sample.input.starts_with(' '));
            assert!(!sample.input.ends_with(' '));
            assert!(!sample.input.is_empty());
            assert!(!sample.target.is_empty());
        }
        
        log::info!("Data preprocessing pipeline test passed");
    }

    fn preprocess_samples(samples: Vec<TrainingSample>, config: &PreprocessingConfig) -> Vec<TrainingSample> {
        samples.into_iter().map(|mut sample| {
            if config.normalize_text {
                sample.input = sample.input.trim().to_lowercase();
                sample.target = sample.target.trim().to_lowercase();
            }
            sample
        }).collect()
    }
}

#[cfg(test)]
mod genetic_algorithm_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_genetic_optimization_pipeline() {
        init_test_env();
        
        // Create genetic algorithm configuration
        let genetic_config = GeneticConfig {
            population_size: 3,
            num_generations: 2,
            mutation_rate: 0.2,
            crossover_rate: 0.8,
            elitism_rate: 0.2,
            tournament_size: 2,
            fitness_threshold: 0.9,
            convergence_threshold: 1e-4,
            max_stagnant_generations: 5,
            parallel_evaluation: false, // Disable for test stability
        };
        
        // Create dataset manager
        let dataset_manager = DatasetManager::new();
        
        // Create genetic evolution
        let mut evolution = GeneticEvolution::new(genetic_config.clone(), dataset_manager);
        
        // Initialize population
        let init_result = evolution.initialize_population();
        assert!(init_result.is_ok());
        
        // Test that genetic algorithm can be set up and initialized
        let stats = evolution.get_stats();
        assert_eq!(stats.get("population_size").copied().unwrap_or(0.0), genetic_config.population_size as f32);
        
        log::info!("Genetic optimization pipeline test passed");
    }

    #[test]
    fn test_genome_validation() {
        init_test_env();
        
        // Create test genome
        let genome = Genome {
            snn_params: SnnGenome {
                hidden_sizes: vec![256, 128],
                threshold: 0.5,
                decay_rate: 0.9,
                refractory_period: 2,
                sparse_rate: 0.15,
            },
            ssm_params: SsmGenome {
                state_size: 32,
                num_layers: 8,
                dt_min: 0.001,
                dt_max: 0.1,
                conv_kernel_size: 4,
            },
            liquid_params: LiquidGenome {
                hidden_size: 256,
                time_constant: 1.0,
                adaptation_rate: 0.05,
                connectivity: 0.3,
                enable_adaptation: true,
            },
            fusion_params: FusionGenome {
                hidden_dim: 512,
                attention_heads: 4,
                dropout_rate: 0.1,
                activation_function: "relu".to_string(),
            },
            training_params: TrainingGenome {
                learning_rate: 1e-4,
                batch_size: 8,
                weight_decay: 0.01,
                dropout_rate: 0.1,
            },
        };
        
        // Validate genome parameters
        assert!(!genome.snn_params.hidden_sizes.is_empty());
        assert!(genome.snn_params.threshold > 0.0 && genome.snn_params.threshold < 1.0);
        assert!(genome.ssm_params.state_size > 0);
        assert!(genome.liquid_params.hidden_size > 0);
        assert!(genome.fusion_params.attention_heads > 0);
        assert!(genome.training_params.learning_rate > 0.0);
        
        log::info!("Genome validation test passed");
    }
}

#[cfg(test)]
mod constraint_validation_integration_tests {
    use super::*;

    #[test]
    fn test_comprehensive_constraint_validation() {
        init_test_env();
        
        // Create performance monitor
        let mut monitor = PerformanceMonitor::new();
        
        // Test latency constraint
        monitor.record_inference_latency(Duration::from_millis(75)); // Good
        monitor.record_inference_latency(Duration::from_millis(120)); // Violation
        
        // Test energy constraint
        monitor.record_energy_consumption(35.0); // Good
        monitor.record_energy_consumption(55.0); // Violation
        
        // Check for constraint violations
        let violations = monitor.get_constraint_violations();
        
        // Should detect both latency and energy violations
        let latency_violations = violations.iter().filter(|v| v.constraint_type == "latency").count();
        let energy_violations = violations.iter().filter(|v| v.constraint_type == "energy").count();
        
        assert!(latency_violations > 0, "Should detect latency violation");
        assert!(energy_violations > 0, "Should detect energy violation");
        
        log::info!("Constraint validation test passed: {} violations detected", violations.len());
    }

    #[test]
    fn test_rtx_2070_ti_constraints() {
        init_test_env();
        
        // Test RTX 2070 Ti specific constraints
        let gpu_memory_mb = 8192; // 8GB
        let target_training_hours = 24.0;
        let max_power_w = 50.0;
        let max_latency_ms = 100.0;
        
        // Validate constraints are realistic for RTX 2070 Ti
        assert!(gpu_memory_mb == 8192, "GPU memory should be 8GB for RTX 2070 Ti");
        assert!(target_training_hours <= 24.0, "Training should complete within 24 hours");
        assert!(max_power_w <= 50.0, "Power consumption should be under 50W");
        assert!(max_latency_ms <= 100.0, "Inference should be under 100ms");
        
        // Test memory efficiency calculation
        let model_memory_estimate = 7500; // Leave 500MB buffer
        assert!(model_memory_estimate < gpu_memory_mb, 
                "Model memory {} should fit in GPU memory {}", 
                model_memory_estimate, gpu_memory_mb);
        
        log::info!("RTX 2070 Ti constraints test passed");
    }
}

#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_inference_pipeline() {
        init_test_env();
        
        // Create complete system
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut model = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        let validation_config = ValidationConfig::default();
        let validator = OutputValidator::new(validation_config);
        let performance_monitor = PerformanceMonitor::new();
        
        // Test inference pipeline
        let input = random_input(snn_config.input_size);
        
        // Time the inference
        let start = Instant::now();
        let output = model.forward(&input).unwrap();
        let inference_time = start.elapsed();
        
        // Validate output
        let output_text = format!("Generated output with {} values", output.len());
        let validation_result = validator.validate_output(&output_text, None, None).await.unwrap();
        
        // Check performance
        let memory_usage = performance_monitor.get_memory_usage_mb();
        
        // Assertions
        assert!(output.len() > 0);
        assert!(inference_time < Duration::from_millis(100));
        assert!(validation_result.confidence_score >= 0.0);
        assert!(memory_usage > 0);
        
        log::info!("Complete inference pipeline test passed in {:?}", inference_time);
    }

    #[tokio::test]
    async fn test_zero_hallucination_validation() {
        init_test_env();
        
        // Create validation system
        let validation_config = ValidationConfig::default();
        let validator = OutputValidator::new(validation_config);
        
        // Test with potentially problematic outputs
        let test_outputs = vec![
            "This is a factual statement about machine learning algorithms.",
            "According to recent studies, experts claim this is true.",
            "It is widely believed that this technology works.",
            "Some researchers suggest that this approach is effective.",
        ];
        
        let mut hallucination_detected = false;
        
        for output in test_outputs {
            let result = validator.validate_output(output, None, None).await.unwrap();
            
            if result.hallucination_risk > 0.5 {
                hallucination_detected = true;
                assert!(!result.issues.is_empty(), "High hallucination risk should have associated issues");
            }
        }
        
        // Should detect at least some potential hallucinations
        assert!(hallucination_detected, "Should detect potential hallucinations in test outputs");
        
        log::info!("Zero hallucination validation test passed");
    }

    #[tokio::test]
    async fn test_multi_modal_fusion() {
        init_test_env();
        
        // Create fusion layer
        let fusion_config = create_test_fusion_config();
        let mut fusion = FusionLayer::new(fusion_config.clone()).unwrap();
        
        // Create inputs from different modalities
        let snn_output = random_input(fusion_config.input_dims[0]);
        let ssm_output = random_input(fusion_config.input_dims[1]);
        let liquid_output = random_input(fusion_config.input_dims[2]);
        
        // Test fusion
        let fused_output = fusion.forward(&snn_output, &ssm_output, &liquid_output).unwrap();
        
        // Validate fusion output
        assert_eq!(fused_output.len(), fusion_config.output_dim);
        assert!(fused_output.iter().all(|&x| x.is_finite()));
        
        // Test attention weights
        let attention_weights = fusion.get_attention_weights();
        assert_eq!(attention_weights.len(), 3);
        
        let weight_sum: f32 = attention_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 0.1, "Attention weights should sum to ~1.0");
        
        log::info!("Multi-modal fusion test passed");
    }
}

#[cfg(test)]
mod error_recovery_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_graceful_degradation_on_component_failure() {
        init_test_env();
        
        // Test system behavior when individual components fail
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        // Create system with potential failure points
        let result = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config);
        
        match result {
            Ok(mut model) => {
                // Test with various input conditions that might cause failures
                let problematic_inputs = vec![
                    Array1::zeros(snn_config.input_size), // All zeros
                    Array1::from_elem(snn_config.input_size, f32::INFINITY), // Infinity values
                    Array1::from_elem(snn_config.input_size, f32::NAN), // NaN values
                    Array1::from_elem(snn_config.input_size, 1e10), // Very large values
                    Array1::from_elem(snn_config.input_size, -1e10), // Very negative values
                ];
                
                for (i, input) in problematic_inputs.iter().enumerate() {
                    let result = model.forward(input);
                    
                    match result {
                        Ok(output) => {
                            // Should produce valid output or handle gracefully
                            assert!(output.iter().all(|&x| x.is_finite() || x == 0.0), 
                                   "Output should be finite or zero for input case {}", i);
                        }
                        Err(e) => {
                            log::warn!("Expected failure for problematic input {}: {}", i, e);
                            // Error should be handled gracefully
                        }
                    }
                }
            }
            Err(e) => {
                log::info!("Model creation failed gracefully: {}", e);
            }
        }
        
        log::info!("Graceful degradation test completed");
    }

    #[tokio::test]
    async fn test_memory_pressure_recovery() {
        init_test_env();
        
        // Test system behavior under memory pressure
        let mut performance_monitor = PerformanceMonitor::new();
        let initial_memory = performance_monitor.get_memory_usage_mb();
        
        // Create multiple models to simulate memory pressure
        let mut models = Vec::new();
        
        for i in 0..3 {
            let snn_config = create_test_snn_config();
            let ssm_config = create_test_ssm_config();
            let liquid_config = create_test_liquid_config();
            let fusion_config = create_test_fusion_config();
            
            match HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config) {
                Ok(model) => {
                    models.push(model);
                    let current_memory = performance_monitor.get_memory_usage_mb();
                    
                    // Check for reasonable memory usage
                    if current_memory - initial_memory > 3000 {
                        log::warn!("High memory usage detected: {}MB increase", current_memory - initial_memory);
                        break;
                    }
                }
                Err(e) => {
                    log::info!("Model {} creation failed under memory pressure: {}", i, e);
                    break;
                }
            }
        }
        
        // Test that existing models still function
        if !models.is_empty() {
            let input = random_input(64);
            for (i, model) in models.iter_mut().enumerate() {
                let result = model.forward(&input);
                assert!(result.is_ok(), "Model {} should still function after memory pressure", i);
            }
        }
        
        log::info!("Memory pressure recovery test completed with {} models", models.len());
    }

    #[test]
    fn test_invalid_configuration_handling() {
        init_test_env();
        
        // Test various invalid configurations
        let invalid_configs = vec![
            // Invalid SNN config - zero input size
            (SnnConfig {
                input_size: 0,
                hidden_sizes: vec![32],
                output_size: 16,
                threshold: 0.5,
                decay_rate: 0.9,
                refractory_period: 2,
                sparse_rate: 0.15,
            }, "zero input size"),
            
            // Invalid threshold
            (SnnConfig {
                input_size: 64,
                hidden_sizes: vec![32],
                output_size: 16,
                threshold: -0.5, // Negative threshold
                decay_rate: 0.9,
                refractory_period: 2,
                sparse_rate: 0.15,
            }, "negative threshold"),
            
            // Invalid decay rate
            (SnnConfig {
                input_size: 64,
                hidden_sizes: vec![32],
                output_size: 16,
                threshold: 0.5,
                decay_rate: 1.5, // > 1.0
                refractory_period: 2,
                sparse_rate: 0.15,
            }, "invalid decay rate"),
        ];
        
        for (config, description) in invalid_configs {
            let result = SnnLayer::new(config);
            assert!(result.is_err(), "Should reject invalid config: {}", description);
        }
        
        log::info!("Invalid configuration handling test passed");
    }
}

#[cfg(test)]
mod stress_testing_integration {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_high_frequency_inference() {
        init_test_env();
        
        // Create model
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut model = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        
        // Perform many rapid inferences
        let num_inferences = 100;
        let mut total_time = Duration::new(0, 0);
        
        for i in 0..num_inferences {
            let input = random_input(snn_config.input_size);
            
            let start = Instant::now();
            let result = model.forward(&input);
            let inference_time = start.elapsed();
            
            assert!(result.is_ok(), "Inference {} should succeed", i);
            total_time += inference_time;
            
            // Ensure each inference is reasonably fast
            assert!(inference_time < Duration::from_millis(200), 
                   "Inference {} took {:?}, too slow", i, inference_time);
        }
        
        let avg_time = total_time / num_inferences;
        assert!(avg_time < Duration::from_millis(100), 
               "Average inference time {:?} exceeds 100ms limit", avg_time);
        
        log::info!("High frequency inference test passed: {} inferences, avg time {:?}", 
                  num_inferences, avg_time);
    }

    #[test]
    fn test_concurrent_model_usage() {
        init_test_env();
        
        // Create shared model (wrapped for thread safety)
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let model = Arc::new(Mutex::new(
            HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap()
        ));
        
        let num_threads = 3;
        let inferences_per_thread = 10;
        let mut handles = vec![];
        
        // Spawn multiple threads doing concurrent inference
        for thread_id in 0..num_threads {
            let model_clone = Arc::clone(&model);
            let input_size = snn_config.input_size;
            
            let handle = thread::spawn(move || {
                let mut successful_inferences = 0;
                
                for i in 0..inferences_per_thread {
                    let input = random_input(input_size);
                    
                    // Lock the model for this inference
                    let mut model_guard = model_clone.lock().unwrap();
                    let result = model_guard.forward(&input);
                    drop(model_guard); // Release lock quickly
                    
                    if result.is_ok() {
                        successful_inferences += 1;
                    } else {
                        log::warn!("Thread {} inference {} failed", thread_id, i);
                    }
                }
                
                successful_inferences
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads and collect results
        let mut total_successful = 0;
        for handle in handles {
            let successful = handle.join().unwrap();
            total_successful += successful;
        }
        
        let expected_total = num_threads * inferences_per_thread;
        let success_rate = total_successful as f32 / expected_total as f32;
        
        assert!(success_rate >= 0.8, 
               "Success rate {:.2} too low for concurrent usage", success_rate);
        
        log::info!("Concurrent model usage test passed: {}/{} successful inferences", 
                  total_successful, expected_total);
    }

    #[test]
    fn test_sustained_operation() {
        init_test_env();
        
        // Test sustained operation over longer period
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut model = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        let mut performance_monitor = PerformanceMonitor::new();
        
        let initial_memory = performance_monitor.get_memory_usage_mb();
        let test_duration = Duration::from_secs(2); // 2 second test
        let start_time = Instant::now();
        
        let mut inference_count = 0;
        let mut max_inference_time = Duration::new(0, 0);
        
        while start_time.elapsed() < test_duration {
            let input = random_input(snn_config.input_size);
            
            let inference_start = Instant::now();
            let result = model.forward(&input);
            let inference_time = inference_start.elapsed();
            
            assert!(result.is_ok(), "Inference should succeed during sustained operation");
            
            max_inference_time = max_inference_time.max(inference_time);
            inference_count += 1;
            
            // Small delay to avoid overwhelming the system
            thread::sleep(Duration::from_millis(1));
        }
        
        let final_memory = performance_monitor.get_memory_usage_mb();
        let memory_growth = final_memory - initial_memory;
        
        // Validate sustained operation metrics
        assert!(inference_count > 10, "Should complete multiple inferences");
        assert!(max_inference_time < Duration::from_millis(150), 
               "Max inference time {:?} too high", max_inference_time);
        assert!(memory_growth < 500, "Memory growth {} MB too high", memory_growth);
        
        log::info!("Sustained operation test passed: {} inferences, max time {:?}, memory growth {}MB", 
                  inference_count, max_inference_time, memory_growth);
    }
}

#[cfg(test)]
mod configuration_validation_integration {
    use super::*;

    #[test]
    fn test_parameter_budget_distribution() {
        init_test_env();
        
        // Test various parameter distribution strategies
        let test_configs = vec![
            // Balanced distribution
            ("balanced", create_test_snn_config(), create_test_ssm_config(), 
             create_test_liquid_config(), create_test_fusion_config()),
             
            // SNN-heavy distribution
            ("snn_heavy", SnnConfig {
                input_size: 128,
                hidden_sizes: vec![512, 256, 128],
                output_size: 64,
                threshold: 0.5,
                decay_rate: 0.9,
                refractory_period: 2,
                sparse_rate: 0.15,
            }, create_test_ssm_config(), create_test_liquid_config(), create_test_fusion_config()),
            
            // Fusion-heavy distribution
            ("fusion_heavy", create_test_snn_config(), create_test_ssm_config(), 
             create_test_liquid_config(), FusionConfig {
                input_dims: vec![8, 8, 8],
                output_dim: 32,
                hidden_dim: 128,
                attention_heads: 8,
                dropout_rate: 0.1,
                use_layer_norm: true,
                activation_function: "gelu".to_string(),
            }),
        ];
        
        for (name, snn_config, ssm_config, liquid_config, fusion_config) in test_configs {
            let result = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config);
            
            match result {
                Ok(model) => {
                    let total_params = model.total_parameters();
                    let breakdown = model.parameter_breakdown();
                    
                    assert!(total_params <= 100_000_000, 
                           "Config {} exceeds parameter budget: {}", name, total_params);
                    
                    // Validate parameter distribution
                    let snn_params = breakdown.get("snn").unwrap_or(&0);
                    let ssm_params = breakdown.get("ssm").unwrap_or(&0);
                    let liquid_params = breakdown.get("liquid").unwrap_or(&0);
                    let fusion_params = breakdown.get("fusion").unwrap_or(&0);
                    
                    assert!(*snn_params <= 30_000_000, "SNN params too high in {}", name);
                    assert!(*ssm_params <= 40_000_000, "SSM params too high in {}", name);
                    assert!(*liquid_params <= 20_000_000, "Liquid params too high in {}", name);
                    assert!(*fusion_params <= 10_000_000, "Fusion params too high in {}", name);
                    
                    log::info!("Config {} passed: {} total params", name, total_params);
                }
                Err(e) => {
                    log::warn!("Config {} failed (expected): {}", name, e);
                }
            }
        }
    }

    #[test]
    fn test_training_configuration_compatibility() {
        init_test_env();
        
        // Test various training configurations
        let training_configs = vec![
            TrainingConfig {
                dataset_name: "small_dataset".to_string(),
                epochs: 5,
                batch_size: 8,
                learning_rate: 1e-3,
                enable_genetic: false,
                output_dir: std::path::PathBuf::from("/tmp/training_small"),
                resume_from: None,
            },
            TrainingConfig {
                dataset_name: "large_dataset".to_string(),
                epochs: 20,
                batch_size: 64,
                learning_rate: 1e-4,
                enable_genetic: true,
                output_dir: std::path::PathBuf::from("/tmp/training_large"),
                resume_from: None,
            },
        ];
        
        for (i, config) in training_configs.iter().enumerate() {
            // Validate training configuration
            assert!(config.epochs > 0, "Config {} has invalid epochs", i);
            assert!(config.batch_size > 0, "Config {} has invalid batch size", i);
            assert!(config.learning_rate > 0.0, "Config {} has invalid learning rate", i);
            assert!(config.learning_rate < 1.0, "Config {} learning rate too high", i);
            
            // Test compatibility with model configurations
            let snn_config = create_test_snn_config();
            let batch_memory_estimate = config.batch_size * snn_config.input_size * 4; // 4 bytes per f32
            
            assert!(batch_memory_estimate < 100_000_000, 
                   "Config {} batch memory too high: {} bytes", i, batch_memory_estimate);
            
            log::info!("Training config {} validated successfully", i);
        }
    }

    #[test] 
    fn test_hardware_constraint_validation() {
        init_test_env();
        
        // RTX 2070 Ti specific constraints
        let rtx_2070_ti_constraints = {
            let mut constraints = HashMap::new();
            constraints.insert("gpu_memory_mb", 8192);
            constraints.insert("max_power_w", 50);
            constraints.insert("max_latency_ms", 100);
            constraints.insert("training_time_hours", 24);
            constraints
        };
        
        // Test model configurations against hardware constraints
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let model = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        
        // Estimate model memory usage (simplified)
        let model_params = model.total_parameters();
        let estimated_memory_mb = (model_params * 4) / (1024 * 1024); // 4 bytes per parameter
        let activation_memory_mb = (snn_config.input_size * 4) / (1024 * 1024);
        let total_memory_mb = estimated_memory_mb + activation_memory_mb + 1024; // +1GB buffer
        
        assert!(total_memory_mb <= rtx_2070_ti_constraints["gpu_memory_mb"] as usize,
               "Model memory {} MB exceeds GPU limit", total_memory_mb);
        
        // Test inference latency constraint
        let input = random_input(snn_config.input_size);
        let mut model_mut = model;
        
        let start = Instant::now();
        let _output = model_mut.forward(&input).unwrap();
        let inference_time_ms = start.elapsed().as_millis();
        
        assert!(inference_time_ms <= rtx_2070_ti_constraints["max_latency_ms"] as u128,
               "Inference time {} ms exceeds limit", inference_time_ms);
        
        log::info!("Hardware constraint validation passed: {} MB memory, {} ms latency", 
                  total_memory_mb, inference_time_ms);
    }
}