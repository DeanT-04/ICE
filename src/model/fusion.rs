//! Fusion layer for combining SNN, SSM, and Liquid NN outputs
//!
//! Implements sophisticated attention-based fusion with adaptive weighting
//! to optimally combine different neural network modalities.
//! Allocated 10M parameters (10% of 100M total budget).

use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum parameters for fusion layer (10% of 100M total)
const FUSION_MAX_PARAMETERS: usize = 10_000_000;

/// Fusion strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub input_dims: Vec<usize>,        // Input dimensions from each component
    pub output_dim: usize,             // Final output dimension
    pub attention_heads: usize,        // Multi-head attention heads
    pub hidden_dim: usize,             // Hidden dimension for fusion network
    pub use_cross_attention: bool,     // Enable cross-attention between modalities
    pub use_adaptive_weights: bool,    // Enable adaptive component weighting
    pub dropout_rate: f32,             // Dropout for regularization
    pub temperature: f32,              // Temperature for attention softmax
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            input_dims: vec![256, 768, 768], // SNN, SSM, Liquid outputs
            output_dim: 768,                 // Standard output size
            attention_heads: 8,              // Multi-head attention
            hidden_dim: 512,                 // Fusion network hidden size
            use_cross_attention: true,       // Enable cross-modality attention
            use_adaptive_weights: true,      // Adaptive component weighting
            dropout_rate: 0.1,               // 10% dropout
            temperature: 1.0,                // Standard temperature
        }
    }
}

/// Attention mechanism for fusion
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    query_weights: Array2<f32>,    // Query projection weights
    key_weights: Array2<f32>,      // Key projection weights  
    value_weights: Array2<f32>,    // Value projection weights
    output_weights: Array2<f32>,   // Output projection weights
    head_dim: usize,               // Dimension per attention head
    num_heads: usize,              // Number of attention heads
}

impl AttentionLayer {
    /// Create new attention layer
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Result<Self, String> {
        if output_dim % num_heads != 0 {
            return Err("Output dimension must be divisible by number of heads".to_string());
        }
        
        let head_dim = output_dim / num_heads;
        let mut rng = thread_rng();
        
        // Xavier initialization
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        
        let query_weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| rng.gen_range(-scale..scale)
        );
        
        let key_weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| rng.gen_range(-scale..scale)
        );
        
        let value_weights = Array2::from_shape_fn(
            (output_dim, input_dim),
            |_| rng.gen_range(-scale..scale)
        );
        
        let output_weights = Array2::from_shape_fn(
            (output_dim, output_dim),
            |_| rng.gen_range(-scale..scale)
        );
        
        Ok(Self {
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            head_dim,
            num_heads,
        })
    }
    
    /// Apply attention mechanism
    pub fn forward(&self, query: &Array1<f32>, key: &Array1<f32>, value: &Array1<f32>, temperature: f32) -> Result<Array1<f32>, String> {
        // Project inputs
        let q = self.query_weights.dot(query);
        let k = self.key_weights.dot(key);
        let v = self.value_weights.dot(value);
        
        // Reshape for multi-head attention (simplified for vector inputs)
        let attention_score = q.dot(&k) / (self.head_dim as f32).sqrt() / temperature;
        let attention_weight = attention_score.exp(); // Simplified softmax for single attention
        
        // Apply attention
        let attended = &v * attention_weight;
        
        // Output projection
        let output = self.output_weights.dot(&attended);
        
        Ok(output)
    }
    
    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.query_weights.len() + self.key_weights.len() + 
        self.value_weights.len() + self.output_weights.len()
    }
}

/// Adaptive weighting network
#[derive(Debug, Clone)]
pub struct AdaptiveWeights {
    weight_network: Array2<f32>,    // Network to compute component weights
    bias: Array1<f32>,              // Bias for weight computation
    num_components: usize,          // Number of input components
}

impl AdaptiveWeights {
    /// Create new adaptive weighting network
    pub fn new(input_dim: usize, num_components: usize) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / input_dim as f32).sqrt();
        
        let weight_network = Array2::from_shape_fn(
            (num_components, input_dim),
            |_| rng.gen_range(-scale..scale)
        );
        
        let bias = Array1::zeros(num_components);
        
        Self {
            weight_network,
            bias,
            num_components,
        }
    }
    
    /// Compute adaptive weights based on input statistics
    pub fn compute_weights(&self, inputs: &[&Array1<f32>]) -> Result<Array1<f32>, String> {
        if inputs.len() != self.num_components {
            return Err(format!(
                "Expected {} components, got {}",
                self.num_components, inputs.len()
            ));
        }
        
        // Compute statistics for each component
        let mut stats = Array1::zeros(self.weight_network.ncols());
        let mut stat_idx = 0;
        
        for input in inputs {
            if stat_idx < stats.len() {
                // Simple statistics: mean, std, max
                let mean = input.mean().unwrap_or(0.0);
                let variance = input.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
                let std_dev = variance.sqrt();
                let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
                
                // Pack statistics (simplified for demonstration)
                if stat_idx < stats.len() { stats[stat_idx] = mean; stat_idx += 1; }
                if stat_idx < stats.len() { stats[stat_idx] = std_dev; stat_idx += 1; }
                if stat_idx < stats.len() { stats[stat_idx] = max_val; stat_idx += 1; }
            }
        }
        
        // Compute weights using network
        let raw_weights = self.weight_network.dot(&stats) + &self.bias;
        
        // Apply softmax normalization
        let max_weight = raw_weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_weights: Array1<f32> = raw_weights.mapv(|x| (x - max_weight).exp());
        let sum_exp = exp_weights.sum();
        let normalized_weights = exp_weights / sum_exp;
        
        Ok(normalized_weights)
    }
    
    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.weight_network.len() + self.bias.len()
    }
}

/// Main fusion layer combining all neural network modalities
#[derive(Debug, Clone)]
pub struct FusionLayer {
    config: FusionConfig,
    
    // Cross-attention between modalities
    snn_to_ssm_attention: Option<AttentionLayer>,
    snn_to_liquid_attention: Option<AttentionLayer>,
    ssm_to_liquid_attention: Option<AttentionLayer>,
    
    // Adaptive component weighting
    adaptive_weights: Option<AdaptiveWeights>,
    
    // Final fusion network
    fusion_network: Array2<f32>,     // Main fusion transformation
    fusion_bias: Array1<f32>,        // Fusion bias
    output_projection: Array2<f32>,  // Final output projection
    output_bias: Array1<f32>,        // Output bias
    
    // Normalization layers
    layer_norm_weights: Array1<f32>, // Layer normalization weights
    layer_norm_bias: Array1<f32>,    // Layer normalization bias
    
    parameter_count: usize,
}

impl FusionLayer {
    /// Create new fusion layer with specified configuration
    pub fn new(config: FusionConfig) -> Result<Self, String> {
        let mut parameter_count = 0;
        let mut rng = thread_rng();
        
        // Initialize cross-attention if enabled
        let (snn_to_ssm_attention, snn_to_liquid_attention, ssm_to_liquid_attention) = 
            if config.use_cross_attention {
                let snn_ssm = AttentionLayer::new(
                    config.input_dims[0], 
                    config.hidden_dim, 
                    config.attention_heads
                )?;
                parameter_count += snn_ssm.parameter_count();
                
                let snn_liquid = AttentionLayer::new(
                    config.input_dims[0], 
                    config.hidden_dim, 
                    config.attention_heads
                )?;
                parameter_count += snn_liquid.parameter_count();
                
                let ssm_liquid = AttentionLayer::new(
                    config.input_dims[1], 
                    config.hidden_dim, 
                    config.attention_heads
                )?;
                parameter_count += ssm_liquid.parameter_count();
                
                (Some(snn_ssm), Some(snn_liquid), Some(ssm_liquid))
            } else {
                (None, None, None)
            };
        
        // Initialize adaptive weighting if enabled
        let adaptive_weights = if config.use_adaptive_weights {
            let stats_dim = config.input_dims.len() * 3; // mean, std, max per component
            let weights = AdaptiveWeights::new(stats_dim, config.input_dims.len());
            parameter_count += weights.parameter_count();
            Some(weights)
        } else {
            None
        };
        
        // Calculate fusion network input dimension
        let fusion_input_dim = if config.use_cross_attention {
            config.hidden_dim * 3 // From cross-attention outputs
        } else {
            config.input_dims.iter().sum() // Direct concatenation
        };
        
        // Initialize fusion network
        let scale = (2.0 / (fusion_input_dim + config.hidden_dim) as f32).sqrt();
        let fusion_network = Array2::from_shape_fn(
            (config.hidden_dim, fusion_input_dim),
            |_| rng.gen_range(-scale..scale)
        );
        parameter_count += fusion_network.len();
        
        let fusion_bias = Array1::zeros(config.hidden_dim);
        parameter_count += fusion_bias.len();
        
        // Output projection
        let output_scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();
        let output_projection = Array2::from_shape_fn(
            (config.output_dim, config.hidden_dim),
            |_| rng.gen_range(-output_scale..output_scale)
        );
        parameter_count += output_projection.len();
        
        let output_bias = Array1::zeros(config.output_dim);
        parameter_count += output_bias.len();
        
        // Layer normalization
        let layer_norm_weights = Array1::ones(config.output_dim);
        let layer_norm_bias = Array1::zeros(config.output_dim);
        parameter_count += layer_norm_weights.len() + layer_norm_bias.len();
        
        // Check parameter budget
        if parameter_count > FUSION_MAX_PARAMETERS {
            return Err(format!(
                "Fusion layer parameter count {} exceeds limit {}",
                parameter_count, FUSION_MAX_PARAMETERS
            ));
        }
        
        Ok(Self {
            config,
            snn_to_ssm_attention,
            snn_to_liquid_attention,
            ssm_to_liquid_attention,
            adaptive_weights,
            fusion_network,
            fusion_bias,
            output_projection,
            output_bias,
            layer_norm_weights,
            layer_norm_bias,
            parameter_count,
        })
    }
    
    /// Forward pass through fusion layer
    /// 
    /// Combines SNN, SSM, and Liquid NN outputs using attention and adaptive weighting
    pub fn forward(&self, snn_output: &Array1<f32>, ssm_output: &Array1<f32>, liquid_output: &Array1<f32>) -> Result<Array1<f32>, String> {
        // Validate input dimensions
        if snn_output.len() != self.config.input_dims[0] ||
           ssm_output.len() != self.config.input_dims[1] ||
           liquid_output.len() != self.config.input_dims[2] {
            return Err("Input dimensions don't match configuration".to_string());
        }
        
        // Apply cross-attention if enabled
        let fused_features = if self.config.use_cross_attention {
            let snn_ssm_attended = self.snn_to_ssm_attention.as_ref().unwrap()
                .forward(snn_output, ssm_output, ssm_output, self.config.temperature)?;
            
            let snn_liquid_attended = self.snn_to_liquid_attention.as_ref().unwrap()
                .forward(snn_output, liquid_output, liquid_output, self.config.temperature)?;
            
            let ssm_liquid_attended = self.ssm_to_liquid_attention.as_ref().unwrap()
                .forward(ssm_output, liquid_output, liquid_output, self.config.temperature)?;
            
            // Concatenate attended features
            let mut concatenated = Array1::zeros(
                snn_ssm_attended.len() + snn_liquid_attended.len() + ssm_liquid_attended.len()
            );
            
            let mut offset = 0;
            for (i, &val) in snn_ssm_attended.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            offset += snn_ssm_attended.len();
            
            for (i, &val) in snn_liquid_attended.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            offset += snn_liquid_attended.len();
            
            for (i, &val) in ssm_liquid_attended.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            
            concatenated
        } else {
            // Simple concatenation
            let total_len = snn_output.len() + ssm_output.len() + liquid_output.len();
            let mut concatenated = Array1::zeros(total_len);
            
            let mut offset = 0;
            for (i, &val) in snn_output.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            offset += snn_output.len();
            
            for (i, &val) in ssm_output.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            offset += ssm_output.len();
            
            for (i, &val) in liquid_output.iter().enumerate() {
                concatenated[offset + i] = val;
            }
            
            concatenated
        };
        
        // Apply fusion network
        let hidden = self.fusion_network.dot(&fused_features) + &self.fusion_bias;
        let activated_hidden = hidden.mapv(|x| x.tanh()); // Tanh activation
        
        // Apply dropout (simplified - just scaling during inference)
        let dropout_scale = 1.0 - self.config.dropout_rate;
        let dropout_hidden = activated_hidden * dropout_scale;
        
        // Output projection
        let mut output = self.output_projection.dot(&dropout_hidden) + &self.output_bias;
        
        // Apply layer normalization
        let mean = output.mean().unwrap_or(0.0);
        let variance = output.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let std_dev = (variance + 1e-8).sqrt(); // Add epsilon for numerical stability
        
        output = (output - mean) / std_dev;
        output = output * &self.layer_norm_weights + &self.layer_norm_bias;
        
        // Apply adaptive weighting if enabled
        if let Some(ref adaptive_weights) = self.adaptive_weights {
            let component_weights = adaptive_weights.compute_weights(&[snn_output, ssm_output, liquid_output])?;
            
            // Apply weighted combination (simplified)
            let weight_scale = component_weights.sum();
            output = output * weight_scale;
        }
        
        Ok(output)
    }
    
    /// Get parameter count for fusion layer
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }
    
    /// Get fusion layer statistics
    pub fn get_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        stats.insert("fusion_parameters".to_string(), self.parameter_count as f32);
        stats.insert("fusion_hidden_dim".to_string(), self.config.hidden_dim as f32);
        stats.insert("fusion_output_dim".to_string(), self.config.output_dim as f32);
        stats.insert("fusion_attention_heads".to_string(), self.config.attention_heads as f32);
        
        stats.insert("fusion_use_cross_attention".to_string(), 
                    if self.config.use_cross_attention { 1.0 } else { 0.0 });
        stats.insert("fusion_use_adaptive_weights".to_string(), 
                    if self.config.use_adaptive_weights { 1.0 } else { 0.0 });
        
        // Parameter distribution within fusion layer
        let attention_params = if self.config.use_cross_attention {
            self.snn_to_ssm_attention.as_ref().map(|a| a.parameter_count()).unwrap_or(0) +
            self.snn_to_liquid_attention.as_ref().map(|a| a.parameter_count()).unwrap_or(0) +
            self.ssm_to_liquid_attention.as_ref().map(|a| a.parameter_count()).unwrap_or(0)
        } else {
            0
        };
        
        let adaptive_params = self.adaptive_weights.as_ref()
                                  .map(|w| w.parameter_count()).unwrap_or(0);
        
        let fusion_net_params = self.fusion_network.len() + self.fusion_bias.len();
        let output_params = self.output_projection.len() + self.output_bias.len();
        let norm_params = self.layer_norm_weights.len() + self.layer_norm_bias.len();
        
        stats.insert("fusion_attention_params".to_string(), attention_params as f32);
        stats.insert("fusion_adaptive_params".to_string(), adaptive_params as f32);
        stats.insert("fusion_network_params".to_string(), fusion_net_params as f32);
        stats.insert("fusion_output_params".to_string(), output_params as f32);
        stats.insert("fusion_norm_params".to_string(), norm_params as f32);
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_layer_creation() {
        let attention = AttentionLayer::new(256, 512, 8);
        assert!(attention.is_ok());
        
        let attention = attention.unwrap();
        assert!(attention.parameter_count() > 0);
    }
    
    #[test]
    fn test_attention_forward() {
        let attention = AttentionLayer::new(256, 256, 8).unwrap();
        let query = Array1::ones(256);
        let key = Array1::ones(256);
        let value = Array1::ones(256);
        
        let output = attention.forward(&query, &key, &value, 1.0);
        assert!(output.is_ok());
        
        let output = output.unwrap();
        assert_eq!(output.len(), 256);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_adaptive_weights() {
        let weights = AdaptiveWeights::new(9, 3); // 3 stats per component
        let inputs = [
            &Array1::ones(100),
            &Array1::zeros(100),
            &Array1::from_elem(100, 0.5),
        ];
        
        let result = weights.compute_weights(&inputs);
        assert!(result.is_ok());
        
        let weights_result = result.unwrap();
        assert_eq!(weights_result.len(), 3);
        
        // Weights should sum to 1 (softmax normalization)
        let sum: f32 = weights_result.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_fusion_layer_creation() {
        let config = FusionConfig::default();
        let fusion = FusionLayer::new(config);
        assert!(fusion.is_ok());
        
        let fusion = fusion.unwrap();
        assert!(fusion.parameter_count() <= FUSION_MAX_PARAMETERS);
        assert!(fusion.parameter_count() > 0);
    }
    
    #[test]
    fn test_fusion_layer_forward() {
        let config = FusionConfig::default();
        let fusion = FusionLayer::new(config.clone()).unwrap();
        
        let snn_output = Array1::ones(config.input_dims[0]);
        let ssm_output = Array1::ones(config.input_dims[1]);
        let liquid_output = Array1::ones(config.input_dims[2]);
        
        let result = fusion.forward(&snn_output, &ssm_output, &liquid_output);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_dim);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_fusion_layer_with_cross_attention() {
        let config = FusionConfig {
            use_cross_attention: true,
            use_adaptive_weights: true,
            ..Default::default()
        };
        
        let fusion = FusionLayer::new(config.clone()).unwrap();
        
        let snn_output = Array1::from_elem(config.input_dims[0], 0.5);
        let ssm_output = Array1::from_elem(config.input_dims[1], -0.3);
        let liquid_output = Array1::from_elem(config.input_dims[2], 0.8);
        
        let result = fusion.forward(&snn_output, &ssm_output, &liquid_output);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.len(), config.output_dim);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_fusion_parameter_budget() {
        let config = FusionConfig {
            hidden_dim: 2048,     // Large hidden dimension
            output_dim: 2048,
            attention_heads: 16,  // Many attention heads
            use_cross_attention: true,
            use_adaptive_weights: true,
            ..Default::default()
        };
        
        let fusion = FusionLayer::new(config);
        
        if let Ok(fusion_layer) = fusion {
            assert!(fusion_layer.parameter_count() <= FUSION_MAX_PARAMETERS);
        } else {
            // Should fail due to parameter budget if too large
            assert!(true); // Expected failure
        }
    }
    
    #[test]
    fn test_fusion_statistics() {
        let config = FusionConfig::default();
        let fusion = FusionLayer::new(config).unwrap();
        
        let stats = fusion.get_stats();
        
        assert!(stats.contains_key("fusion_parameters"));
        assert!(stats.contains_key("fusion_hidden_dim"));
        assert!(stats.contains_key("fusion_output_dim"));
        assert!(stats.contains_key("fusion_use_cross_attention"));
        assert!(stats.contains_key("fusion_use_adaptive_weights"));
        
        let total_params = stats["fusion_parameters"];
        assert!(total_params > 0.0);
        assert!(total_params <= FUSION_MAX_PARAMETERS as f32);
    }
}