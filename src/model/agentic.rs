//! Agentic system for sub-model spawning and ensemble voting
//!
//! Implements intelligent task decomposition, parallel sub-model execution,
//! and ensemble voting for zero-hallucination performance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use ndarray::Array1;

use crate::model::core::{HybridLayer, SnnConfig, SsmConfig, LiquidConfig};
use crate::model::fusion::FusionConfig;

/// Task types for agentic decomposition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    TextGeneration,
    CodeGeneration,
    MathReasoning,
    Classification,
    QuestionAnswering,
}

/// Task complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskComplexity {
    Simple,      // 1 model
    Moderate,    // 3 models
    Complex,     // 5 models
}

/// Task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub task_type: TaskType,
    pub complexity: TaskComplexity,
    pub max_sub_models: usize,
    pub confidence_threshold: f32,
    pub consensus_threshold: f32,
    pub timeout_ms: u64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::TextGeneration,
            complexity: TaskComplexity::Moderate,
            max_sub_models: 5,
            confidence_threshold: 0.8,
            consensus_threshold: 0.6,
            timeout_ms: 5000,
        }
    }
}

/// Sub-model configuration
#[derive(Debug, Clone)]
pub struct SubModelConfig {
    pub id: String,
    pub specialization: TaskType,
    pub weight: f32,
    pub snn_config: SnnConfig,
    pub ssm_config: SsmConfig,
    pub liquid_config: LiquidConfig,
    pub fusion_config: FusionConfig,
}

/// Sub-model result
#[derive(Debug, Clone)]
pub struct SubModelResult {
    pub model_id: String,
    pub output: Array1<f32>,
    pub confidence: f32,
    pub execution_time_ms: u64,
}

/// Voting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    MajorityVote,
    WeightedVote,
    ConsensusFiltering,
}

/// Agentic coordinator
#[derive(Debug)]
pub struct AgenticCoordinator {
    config: TaskConfig,
    sub_models: Vec<SubModelConfig>,
    voting_strategy: VotingStrategy,
    model_performance: Arc<Mutex<HashMap<String, f32>>>,
}

impl AgenticCoordinator {
    /// Create new coordinator
    pub fn new(config: TaskConfig, voting_strategy: VotingStrategy) -> Self {
        let sub_models = Self::generate_sub_models(&config);
        
        Self {
            config,
            sub_models,
            voting_strategy,
            model_performance: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Generate specialized sub-models
    fn generate_sub_models(config: &TaskConfig) -> Vec<SubModelConfig> {
        let mut sub_models = Vec::new();
        let num_models = match config.complexity {
            TaskComplexity::Simple => 1,
            TaskComplexity::Moderate => 3,
            TaskComplexity::Complex => 5,
        };
        
        for i in 0..num_models {
            let specialization = match i {
                0 => config.task_type.clone(),
                1 => TaskType::Classification,
                2 => TaskType::QuestionAnswering,
                _ => config.task_type.clone(),
            };
            
            let (snn_config, ssm_config, liquid_config, fusion_config) = 
                Self::get_specialized_configs(&specialization);
            
            sub_models.push(SubModelConfig {
                id: format!("sub_model_{}", i),
                specialization,
                weight: 1.0 / num_models as f32,
                snn_config,
                ssm_config,
                liquid_config,
                fusion_config,
            });
        }
        
        sub_models
    }
    
    /// Get specialized configurations
    fn get_specialized_configs(
        specialization: &TaskType
    ) -> (SnnConfig, SsmConfig, LiquidConfig, FusionConfig) {
        match specialization {
            TaskType::CodeGeneration => {
                let snn_config = SnnConfig {
                    input_size: 512,
                    hidden_sizes: vec![512, 256],
                    output_size: 256,
                    sparse_rate: 0.2,
                    ..Default::default()
                };
                
                let ssm_config = SsmConfig {
                    input_size: 512,
                    state_size: 32,
                    num_layers: 10,
                    output_size: 256,
                    ..Default::default()
                };
                
                let liquid_config = LiquidConfig {
                    input_size: 512,
                    hidden_size: 128,
                    output_size: 256,
                    enable_adaptation: false,
                    ..Default::default()
                };
                
                let fusion_config = FusionConfig {
                    input_dims: vec![256, 256, 256],
                    output_dim: 256,
                    hidden_dim: 128,
                    attention_heads: 4,
                    ..Default::default()
                };
                
                (snn_config, ssm_config, liquid_config, fusion_config)
            },
            
            _ => {
                let snn_config = SnnConfig {
                    input_size: 512,
                    hidden_sizes: vec![384, 256],
                    output_size: 192,
                    ..Default::default()
                };
                
                let ssm_config = SsmConfig {
                    input_size: 512,
                    state_size: 24,
                    num_layers: 8,
                    output_size: 192,
                    ..Default::default()
                };
                
                let liquid_config = LiquidConfig {
                    input_size: 512,
                    hidden_size: 256,
                    output_size: 192,
                    ..Default::default()
                };
                
                let fusion_config = FusionConfig {
                    input_dims: vec![192, 192, 192],
                    output_dim: 192,
                    hidden_dim: 192,
                    attention_heads: 6,
                    ..Default::default()
                };
                
                (snn_config, ssm_config, liquid_config, fusion_config)
            }
        }
    }
    
    /// Execute task with ensemble
    pub async fn execute_task(&mut self, input: &Array1<f32>) -> Result<Array1<f32>, String> {
        let sub_results = self.spawn_sub_models(input).await?;
        let ensemble_result = self.ensemble_vote(&sub_results)?;
        self.update_performance(&sub_results).await;
        Ok(ensemble_result)
    }
    
    /// Spawn sub-models in parallel
    async fn spawn_sub_models(&self, input: &Array1<f32>) -> Result<Vec<SubModelResult>, String> {
        let mut handles = Vec::new();
        
        for sub_model_config in &self.sub_models {
            let input_clone = input.clone();
            let config_clone = sub_model_config.clone();
            
            let handle = tokio::spawn(async move {
                Self::execute_sub_model(config_clone, input_clone).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        let timeout_duration = std::time::Duration::from_millis(self.config.timeout_ms);
        
        for handle in handles {
            match tokio::time::timeout(timeout_duration, handle).await {
                Ok(Ok(Ok(result))) => results.push(result),
                _ => continue,
            }
        }
        
        if results.is_empty() {
            return Err("All sub-models failed".to_string());
        }
        
        Ok(results)
    }
    
    /// Execute single sub-model
    async fn execute_sub_model(
        config: SubModelConfig, 
        input: Array1<f32>
    ) -> Result<SubModelResult, String> {
        let start_time = Instant::now();
        
        let mut hybrid = HybridLayer::new(
            config.snn_config,
            config.ssm_config,
            config.liquid_config,
            config.fusion_config,
        )?;
        
        let output = hybrid.forward(&input)?;
        let confidence = Self::calculate_confidence(&output);
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        Ok(SubModelResult {
            model_id: config.id,
            output,
            confidence,
            execution_time_ms: execution_time,
        })
    }
    
    /// Calculate confidence score
    fn calculate_confidence(output: &Array1<f32>) -> f32 {
        let entropy = Self::calculate_entropy(output);
        let magnitude = output.mapv(|x| x * x).sum().sqrt() / output.len() as f32;
        
        let confidence = (1.0 - entropy) * 0.7 + magnitude.min(1.0) * 0.3;
        confidence.max(0.0).min(1.0)
    }
    
    /// Calculate normalized entropy
    fn calculate_entropy(output: &Array1<f32>) -> f32 {
        let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Array1<f32> = output.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_vals.sum();
        let softmax = exp_vals / sum_exp;
        
        let entropy = softmax.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum::<f32>();
        
        entropy / (output.len() as f32).ln()
    }
    
    /// Apply ensemble voting
    fn ensemble_vote(&self, results: &[SubModelResult]) -> Result<Array1<f32>, String> {
        match self.voting_strategy {
            VotingStrategy::MajorityVote => self.majority_vote(results),
            VotingStrategy::WeightedVote => self.weighted_vote(results),
            VotingStrategy::ConsensusFiltering => self.consensus_filtering(results),
        }
    }
    
    /// Simple majority vote
    fn majority_vote(&self, results: &[SubModelResult]) -> Result<Array1<f32>, String> {
        let output_len = results[0].output.len();
        let mut sum = Array1::zeros(output_len);
        let mut count = 0;
        
        for result in results {
            if result.output.len() == output_len {
                sum = sum + &result.output;
                count += 1;
            }
        }
        
        Ok(sum / count as f32)
    }
    
    /// Confidence-weighted vote
    fn weighted_vote(&self, results: &[SubModelResult]) -> Result<Array1<f32>, String> {
        let output_len = results[0].output.len();
        let mut weighted_sum = Array1::zeros(output_len);
        let mut total_weight = 0.0;
        
        for result in results {
            if result.output.len() == output_len {
                weighted_sum = weighted_sum + &result.output * result.confidence;
                total_weight += result.confidence;
            }
        }
        
        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            self.majority_vote(results)
        }
    }
    
    /// Consensus filtering
    fn consensus_filtering(&self, results: &[SubModelResult]) -> Result<Array1<f32>, String> {
        let mut consensus_results = Vec::new();
        
        for (i, result_i) in results.iter().enumerate() {
            let mut agreement_count = 0;
            
            for (j, result_j) in results.iter().enumerate() {
                if i != j {
                    let similarity = Self::calculate_similarity(&result_i.output, &result_j.output);
                    if similarity > self.config.consensus_threshold {
                        agreement_count += 1;
                    }
                }
            }
            
            let consensus_ratio = agreement_count as f32 / (results.len() - 1) as f32;
            if consensus_ratio >= self.config.consensus_threshold {
                consensus_results.push(result_i);
            }
        }
        
        if consensus_results.is_empty() {
            self.weighted_vote(results)
        } else {
            let filtered: Vec<SubModelResult> = consensus_results.iter().map(|&r| r.clone()).collect();
            self.weighted_vote(&filtered)
        }
    }
    
    /// Calculate cosine similarity
    fn calculate_similarity(output1: &Array1<f32>, output2: &Array1<f32>) -> f32 {
        if output1.len() != output2.len() {
            return 0.0;
        }
        
        let dot_product = output1.iter().zip(output2.iter()).map(|(a, b)| a * b).sum::<f32>();
        let norm1 = output1.mapv(|x| x * x).sum().sqrt();
        let norm2 = output2.mapv(|x| x * x).sum().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
    
    /// Update model performance metrics
    async fn update_performance(&self, results: &[SubModelResult]) {
        let mut performance = self.model_performance.lock().unwrap();
        for result in results {
            let current_perf = performance.get(&result.model_id).unwrap_or(&0.5);
            let new_perf = 0.9 * current_perf + 0.1 * result.confidence;
            performance.insert(result.model_id.clone(), new_perf);
        }
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        stats.insert("num_sub_models".to_string(), self.sub_models.len() as f32);
        stats.insert("confidence_threshold".to_string(), self.config.confidence_threshold);
        stats.insert("consensus_threshold".to_string(), self.config.consensus_threshold);
        
        if let Ok(perf_stats) = self.model_performance.lock() {
            let avg_performance = perf_stats.values().sum::<f32>() / perf_stats.len().max(1) as f32;
            stats.insert("avg_model_performance".to_string(), avg_performance);
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_task_config_creation() {
        let config = TaskConfig::default();
        assert_eq!(config.task_type, TaskType::TextGeneration);
        assert!(config.max_sub_models > 0);
    }
    
    #[test]
    fn test_confidence_calculation() {
        let output1 = Array1::from_vec(vec![0.9, 0.1, 0.0]);
        let output2 = Array1::from_vec(vec![0.33, 0.33, 0.34]);
        
        let conf1 = AgenticCoordinator::calculate_confidence(&output1);
        let conf2 = AgenticCoordinator::calculate_confidence(&output2);
        
        assert!(conf1 > conf2);
        assert!(conf1 >= 0.0 && conf1 <= 1.0);
    }
    
    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = TaskConfig::default();
        let coordinator = AgenticCoordinator::new(config, VotingStrategy::WeightedVote);
        
        assert!(!coordinator.sub_models.is_empty());
        assert!(matches!(coordinator.voting_strategy, VotingStrategy::WeightedVote));
    }
}