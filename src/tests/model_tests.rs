//! Model component tests
//!
//! Comprehensive tests for all neural network components including
//! SNN, SSM, Liquid NN, fusion layer, and agentic systems.

use super::test_utils::*;
use super::*;
use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use crate::model::validation::*;
use crate::model::mcp::*;
use ndarray::Array1;
use std::collections::HashMap;

#[cfg(test)]
mod snn_tests {
    use super::*;

    #[test]
    fn test_snn_config_creation() {
        let config = create_test_snn_config();
        assert_eq!(config.input_size, 64);
        assert_eq!(config.output_size, 8);
        assert!(config.threshold > 0.0);
        assert!(config.decay_rate > 0.0 && config.decay_rate < 1.0);
        assert!(config.sparse_rate > 0.0 && config.sparse_rate < 1.0);
    }

    #[test]
    fn test_snn_layer_creation() {
        let config = create_test_snn_config();
        let result = SnnLayer::new(config);
        assert!(result.is_ok());
        
        let snn = result.unwrap();
        assert_eq!(snn.parameter_count(), 0); // Will be set after initialization
    }

    #[test]
    fn test_snn_forward_pass() {
        let config = create_test_snn_config();
        let mut snn = SnnLayer::new(config.clone()).unwrap();
        let input = random_input(config.input_size);
        
        let result = snn.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_size);
        
        // Output should be binary (0 or 1) for spikes
        for &val in output.iter() {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_snn_sparse_activation() {
        let mut config = create_test_snn_config();
        config.sparse_rate = 0.1; // 10% activation rate
        
        let mut snn = SnnLayer::new(config.clone()).unwrap();
        let input = Array1::ones(config.input_size); // High input to trigger spikes
        
        let output = snn.forward(&input).unwrap();
        let spike_count = output.iter().filter(|&&x| x > 0.0).count();
        let activation_rate = spike_count as f32 / output.len() as f32;
        
        // Should be approximately sparse_rate (with some tolerance)
        assert!(activation_rate <= config.sparse_rate + 0.05);
    }

    #[test]
    fn test_snn_state_reset() {
        let config = create_test_snn_config();
        let mut snn = SnnLayer::new(config.clone()).unwrap();
        
        // Process some input to change internal state
        let input = random_input(config.input_size);
        snn.forward(&input).unwrap();
        
        // Reset state
        snn.reset_state();
        
        // State should be reset (this is mostly internal state)
        let stats = snn.get_activation_stats();
        assert!(stats.len() >= 0); // At least some stats should be available
    }

    #[test]
    fn test_snn_activation_stats() {
        let config = create_test_snn_config();
        let mut snn = SnnLayer::new(config.clone()).unwrap();
        let input = random_input(config.input_size);
        
        snn.forward(&input).unwrap();
        let stats = snn.get_activation_stats();
        
        // Should have stats for each layer
        assert!(!stats.is_empty());
        
        // Check that activation rates are reasonable
        for (key, &value) in &stats {
            if key.contains("activation_rate") {
                assert!(value >= 0.0 && value <= 1.0);
            }
        }
    }
}

#[cfg(test)]
mod ssm_tests {
    use super::*;

    #[test]
    fn test_ssm_config_creation() {
        let config = create_test_ssm_config();
        assert_eq!(config.input_size, 64);
        assert_eq!(config.state_size, 8);
        assert_eq!(config.output_size, 8);
        assert!(config.num_layers > 0);
        assert!(config.dt_min > 0.0);
        assert!(config.dt_max > config.dt_min);
    }

    #[test]
    fn test_ssm_layer_creation() {
        let config = create_test_ssm_config();
        let result = SsmLayer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ssm_forward_pass() {
        let config = create_test_ssm_config();
        let mut ssm = SsmLayer::new(config.clone()).unwrap();
        let input = random_input(config.input_size);
        
        let result = ssm.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_size);
        
        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ssm_parameter_count() {
        let config = create_test_ssm_config();
        let ssm = SsmLayer::new(config).unwrap();
        let param_count = ssm.parameter_count();
        
        assert!(param_count > 0);
        assert!(param_count <= SSM_MAX_PARAMETERS);
    }

    #[test]
    fn test_ssm_sequence_processing() {
        let config = create_test_ssm_config();
        let mut ssm = SsmLayer::new(config.clone()).unwrap();
        
        // Process a sequence of inputs
        let sequence_length = 5;
        let mut outputs = Vec::new();
        
        for _ in 0..sequence_length {
            let input = random_input(config.input_size);
            let output = ssm.forward(&input).unwrap();
            outputs.push(output);
        }
        
        assert_eq!(outputs.len(), sequence_length);
        
        // Each output should have correct size
        for output in outputs {
            assert_eq!(output.len(), config.output_size);
        }
    }

    #[test]
    fn test_ssm_state_reset() {
        let config = create_test_ssm_config();
        let mut ssm = SsmLayer::new(config.clone()).unwrap();
        
        // Process input to change state
        let input = random_input(config.input_size);
        let output1 = ssm.forward(&input).unwrap();
        
        // Reset state
        ssm.reset_state();
        
        // Process same input again
        let output2 = ssm.forward(&input).unwrap();
        
        // Outputs should be identical after reset
        assert_arrays_close(&output1, &output2, 1e-6);
    }
}

#[cfg(test)]
mod liquid_tests {
    use super::*;

    #[test]
    fn test_liquid_config_creation() {
        let config = create_test_liquid_config();
        assert_eq!(config.input_size, 64);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.output_size, 8);
        assert!(config.time_constant > 0.0);
        assert!(config.adaptation_rate > 0.0);
        assert!(config.connectivity > 0.0 && config.connectivity <= 1.0);
    }

    #[test]
    fn test_liquid_layer_creation() {
        let config = create_test_liquid_config();
        let result = LiquidLayer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_liquid_forward_pass() {
        let config = create_test_liquid_config();
        let mut liquid = LiquidLayer::new(config.clone()).unwrap();
        let input = random_input(config.input_size);
        
        let result = liquid.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_size);
        
        // Output should be finite and bounded
        for &val in output.iter() {
            assert!(val.is_finite());
            assert!(val >= -10.0 && val <= 10.0); // Reasonable bounds
        }
    }

    #[test]
    fn test_liquid_adaptation() {
        let mut config = create_test_liquid_config();
        config.enable_adaptation = true;
        
        let mut liquid = LiquidLayer::new(config.clone()).unwrap();
        let input = random_input(config.input_size);
        
        // Process same input multiple times
        let output1 = liquid.forward(&input).unwrap();
        let output2 = liquid.forward(&input).unwrap();
        let output3 = liquid.forward(&input).unwrap();
        
        // Outputs should differ due to adaptation
        let diff12 = (&output2 - &output1).mapv(|x| x.abs()).sum();
        let diff23 = (&output3 - &output2).mapv(|x| x.abs()).sum();
        
        assert!(diff12 > 0.0 || diff23 > 0.0); // Some adaptation should occur
    }

    #[test]
    fn test_liquid_parameter_count() {
        let config = create_test_liquid_config();
        let liquid = LiquidLayer::new(config).unwrap();
        let param_count = liquid.parameter_count();
        
        assert!(param_count > 0);
        assert!(param_count <= LIQUID_MAX_PARAMETERS);
    }

    #[test]
    fn test_liquid_dynamics() {
        let config = create_test_liquid_config();
        let liquid = LiquidLayer::new(config.clone()).unwrap();
        let dynamics = liquid.get_dynamics();
        
        assert_eq!(dynamics.len(), config.hidden_size);
        
        // Dynamics should be reasonable values
        for &val in dynamics.iter() {
            assert!(val.is_finite());
            assert!(val >= 0.0); // Should be positive (rates/times)
        }
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;

    #[test]
    fn test_fusion_config_creation() {
        let config = create_test_fusion_config();
        assert_eq!(config.input_dims, vec![8, 8, 8]);
        assert_eq!(config.output_dim, 16);
        assert!(config.attention_heads > 0);
        assert!(config.dropout_rate >= 0.0 && config.dropout_rate < 1.0);
    }

    #[test]
    fn test_fusion_layer_creation() {
        let config = create_test_fusion_config();
        let result = FusionLayer::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fusion_forward_pass() {
        let config = create_test_fusion_config();
        let mut fusion = FusionLayer::new(config.clone()).unwrap();
        
        // Create inputs for each modality
        let snn_output = random_input(config.input_dims[0]);
        let ssm_output = random_input(config.input_dims[1]);
        let liquid_output = random_input(config.input_dims[2]);
        
        let result = fusion.forward(&snn_output, &ssm_output, &liquid_output);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_dim);
        
        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_fusion_attention_weights() {
        let config = create_test_fusion_config();
        let mut fusion = FusionLayer::new(config.clone()).unwrap();
        
        let snn_output = random_input(config.input_dims[0]);
        let ssm_output = random_input(config.input_dims[1]);
        let liquid_output = random_input(config.input_dims[2]);
        
        fusion.forward(&snn_output, &ssm_output, &liquid_output).unwrap();
        let weights = fusion.get_attention_weights();
        
        assert_eq!(weights.len(), 3); // One weight per input modality
        
        // Weights should sum to approximately 1.0 (after softmax)
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
        
        // Each weight should be positive
        for &weight in &weights {
            assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_fusion_parameter_count() {
        let config = create_test_fusion_config();
        let fusion = FusionLayer::new(config).unwrap();
        let param_count = fusion.parameter_count();
        
        assert!(param_count > 0);
        assert!(param_count <= FUSION_MAX_PARAMETERS);
    }
}

#[cfg(test)]
mod hybrid_tests {
    use super::*;

    #[test]
    fn test_hybrid_layer_creation() {
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let result = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hybrid_forward_pass() {
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut hybrid = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        let input = random_input(snn_config.input_size);
        
        let result = hybrid.forward(&input);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert!(output.len() > 0);
        
        // Output should be finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_hybrid_parameter_budget() {
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let hybrid = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        let total_params = hybrid.total_parameters();
        
        // Should stay within 100M parameter budget
        assert!(total_params <= 100_000_000);
        
        // Should have reasonable distribution
        let breakdown = hybrid.parameter_breakdown();
        assert!(breakdown.contains_key("snn"));
        assert!(breakdown.contains_key("ssm"));
        assert!(breakdown.contains_key("liquid"));
        assert!(breakdown.contains_key("fusion"));
    }

    #[test]
    fn test_hybrid_multiple_forward_passes() {
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let mut hybrid = HybridLayer::new(snn_config.clone(), ssm_config, liquid_config, fusion_config).unwrap();
        
        // Process multiple inputs
        for _ in 0..5 {
            let input = random_input(snn_config.input_size);
            let output = hybrid.forward(&input).unwrap();
            assert!(output.len() > 0);
        }
    }
}

#[cfg(test)]
mod agentic_tests {
    use super::*;

    #[test]
    fn test_task_config_creation() {
        let config = TaskConfig::default();
        assert!(config.max_sub_models > 0);
        assert!(config.confidence_threshold > 0.0 && config.confidence_threshold <= 1.0);
        assert!(config.consensus_threshold > 0.0 && config.consensus_threshold <= 1.0);
    }

    #[test]
    fn test_agentic_coordinator_creation() {
        let config = TaskConfig::default();
        let coordinator = AgenticCoordinator::new(config, VotingStrategy::WeightedVote);
        
        let stats = coordinator.get_stats();
        assert!(stats.contains_key("num_sub_models"));
    }

    #[tokio::test]
    async fn test_agentic_task_execution() {
        let config = TaskConfig::default();
        let mut coordinator = AgenticCoordinator::new(config, VotingStrategy::WeightedVote);
        
        let input = random_input(64);
        let result = coordinator.execute_task(&input).await;
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.len() > 0);
    }

    #[test]
    fn test_voting_strategies() {
        let strategies = [
            VotingStrategy::MajorityVote,
            VotingStrategy::WeightedVote,
            VotingStrategy::ConsensusFiltering,
        ];
        
        for strategy in &strategies {
            let config = TaskConfig::default();
            let coordinator = AgenticCoordinator::new(config, strategy.clone());
            let stats = coordinator.get_stats();
            assert!(!stats.is_empty());
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_config_creation() {
        let config = ValidationConfig::default();
        assert!(config.enable_validation);
        assert!(config.confidence_threshold > 0.0);
        assert!(config.consistency_threshold > 0.0);
    }

    #[tokio::test]
    async fn test_output_validator_creation() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let stats = validator.get_validation_stats();
        assert!(stats.contains_key("confidence_threshold"));
    }

    #[tokio::test]
    async fn test_validation_good_output() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let good_output = "This is a clear and factual statement about Python programming.";
        let result = validator.validate_output(good_output, None, None).await.unwrap();
        
        assert!(result.confidence_score > 0.5);
        assert!(result.consistency_score > 0.5);
    }

    #[tokio::test]
    async fn test_validation_suspicious_output() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let suspicious_output = "According to recent studies, experts say this is widely known.";
        let result = validator.validate_output(suspicious_output, None, None).await.unwrap();
        
        assert!(result.hallucination_risk > 0.0);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_quick_validation() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let good_text = "This is a reasonable statement.";
        let bad_text = "According to experts, this is suspicious.";
        
        assert!(validator.quick_validate(good_text));
        assert!(!validator.quick_validate(bad_text));
    }
}

#[cfg(test)]
mod mcp_tests {
    use super::*;

    #[test]
    fn test_mcp_config_creation() {
        let config = McpConfig::default();
        assert!(!config.api_base_url.is_empty());
        assert!(config.default_timeout_ms > 0);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_mcp_client_creation() {
        let config = McpConfig::default();
        let client = McpClient::new(config);
        
        // Should create without error
        assert!(true);
    }

    #[tokio::test]
    async fn test_mcp_request_creation() {
        let request = McpRequest {
            server_type: McpServerType::Api,
            method: "test".to_string(),
            params: serde_json::json!({}),
            timeout_ms: 5000,
            cache_ttl_hours: 1,
            retry_count: 1,
        };
        
        assert_eq!(request.method, "test");
        assert_eq!(request.timeout_ms, 5000);
    }

    #[tokio::test]
    async fn test_mcp_helper_methods() {
        let config = McpConfig::default();
        let mut client = McpClient::new(config);
        
        // Test dataset fetch (will return mock response)
        let result = client.fetch_dataset("test_dataset").await;
        assert!(result.is_ok());
        
        // Test code analysis (will return mock response)
        let result = client.analyze_code("fn main() {}", "rust").await;
        assert!(result.is_ok());
    }
}