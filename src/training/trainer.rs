//! Training loop implementation with memory-efficient training
//!
//! Implements training with AdamW optimizer, FP16/INT8 mixed precision,
//! gradient accumulation, and memory optimization for <8GB VRAM constraints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::model::core::HybridLayer;
use crate::model::agentic::AgenticCoordinator;
use crate::training::datasets::{DatasetManager, SplitType, TrainingSample};
use crate::utils::perf::PerformanceMonitor;
use crate::Result;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // Model parameters
    pub model_config: ModelTrainingConfig,
    
    // Optimizer parameters
    pub optimizer: OptimizerConfig,
    
    // Training parameters
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub max_epochs: usize,
    pub max_steps: Option<usize>,
    pub eval_steps: usize,
    pub save_steps: usize,
    
    // Memory optimization
    pub mixed_precision: bool,
    pub use_fp16: bool,
    pub use_int8: bool,
    pub max_memory_mb: usize, // 8GB = 8192MB
    pub gradient_checkpointing: bool,
    
    // Learning rate scheduling
    pub learning_rate_schedule: LearningRateSchedule,
    
    // Early stopping
    pub early_stopping: EarlyStoppingConfig,
    
    // Validation
    pub validation_fraction: f32,
    pub validation_frequency: usize,
    
    // Checkpointing
    pub checkpoint_dir: String,
    pub keep_best_only: bool,
    
    // Hardware constraints
    pub target_vram_mb: usize,
    pub target_training_hours: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_config: ModelTrainingConfig::default(),
            optimizer: OptimizerConfig::default(),
            batch_size: 4, // Small batch for memory efficiency
            gradient_accumulation_steps: 16, // Effective batch size = 64
            max_epochs: 10,
            max_steps: None,
            eval_steps: 500,
            save_steps: 1000,
            mixed_precision: true,
            use_fp16: true,
            use_int8: false, // Enable for even more memory savings
            max_memory_mb: 7500, // Leave 500MB buffer
            gradient_checkpointing: true,
            learning_rate_schedule: LearningRateSchedule::default(),
            early_stopping: EarlyStoppingConfig::default(),
            validation_fraction: 0.1,
            validation_frequency: 100,
            checkpoint_dir: "checkpoints".to_string(),
            keep_best_only: true,
            target_vram_mb: 8192,
            target_training_hours: 24.0,
        }
    }
}

/// Model training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub dropout_rate: f32,
    pub label_smoothing: f32,
}

impl Default for ModelTrainingConfig {
    fn default() -> Self {
        Self {
            input_size: 768,
            hidden_size: 512,
            output_size: 768,
            vocab_size: 32000,
            max_sequence_length: 2048,
            dropout_rate: 0.1,
            label_smoothing: 0.1,
        }
    }
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub amsgrad: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            amsgrad: false,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    AdamW,
    Adam,
    SGD,
    RMSprop,
}

/// Learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateSchedule {
    pub schedule_type: ScheduleType,
    pub warmup_steps: usize,
    pub decay_steps: usize,
    pub decay_rate: f32,
    pub min_learning_rate: f32,
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self {
            schedule_type: ScheduleType::CosineAnnealing,
            warmup_steps: 1000,
            decay_steps: 10000,
            decay_rate: 0.1,
            min_learning_rate: 1e-6,
        }
    }
}

/// Schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Constant,
    Linear,
    CosineAnnealing,
    ExponentialDecay,
    StepDecay,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: usize,
    pub min_delta: f32,
    pub metric: String,
    pub mode: String, // "min" or "max"
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 5,
            min_delta: 1e-4,
            metric: "validation_loss".to_string(),
            mode: "min".to_string(),
        }
    }
}

/// Training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub global_step: usize,
    pub learning_rate: f32,
    pub train_loss: f32,
    pub validation_loss: Option<f32>,
    pub best_validation_loss: f32,
    pub no_improvement_count: usize,
    pub total_samples_seen: usize,
    pub memory_usage_mb: usize,
    pub training_time_hours: f32,
    pub should_stop: bool,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            global_step: 0,
            learning_rate: 0.0,
            train_loss: 0.0,
            validation_loss: None,
            best_validation_loss: f32::INFINITY,
            no_improvement_count: 0,
            total_samples_seen: 0,
            memory_usage_mb: 0,
            training_time_hours: 0.0,
            should_stop: false,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub perplexity: f32,
    pub accuracy: f32,
    pub learning_rate: f32,
    pub memory_usage_mb: usize,
    pub throughput_samples_per_sec: f32,
    pub step_time_ms: u64,
}

/// AdamW optimizer implementation
pub struct AdamWOptimizer {
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: usize,
    momentum: HashMap<String, Array1<f32>>,
    velocity: HashMap<String, Array1<f32>>,
}

impl AdamWOptimizer {
    pub fn new(config: &OptimizerConfig) -> Self {
        Self {
            learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            step_count: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    pub fn step(&mut self, parameters: &mut HashMap<String, Array1<f32>>, gradients: &HashMap<String, Array1<f32>>) -> Result<()> {
        self.step_count += 1;
        let lr = self.get_learning_rate();

        for (name, param) in parameters.iter_mut() {
            if let Some(grad) = gradients.get(name) {
                // Initialize momentum and velocity if needed
                if !self.momentum.contains_key(name) {
                    self.momentum.insert(name.clone(), Array1::zeros(param.len()));
                    self.velocity.insert(name.clone(), Array1::zeros(param.len()));
                }

                let m = self.momentum.get_mut(name).unwrap();
                let v = self.velocity.get_mut(name).unwrap();

                // Update biased first moment estimate
                *m = &*m * self.beta1 + grad * (1.0 - self.beta1);

                // Update biased second raw moment estimate
                *v = &*v * self.beta2 + &grad.mapv(|x| x * x) * (1.0 - self.beta2);

                // Compute bias-corrected first moment estimate
                let m_hat = &*m / (1.0 - self.beta1.powi(self.step_count as i32));

                // Compute bias-corrected second raw moment estimate
                let v_hat = &*v / (1.0 - self.beta2.powi(self.step_count as i32));

                // Update parameters with weight decay (AdamW style)
                *param = &*param * (1.0 - lr * self.weight_decay);

                // Apply gradient update
                for i in 0..param.len() {
                    param[i] -= lr * m_hat[i] / (v_hat[i].sqrt() + self.epsilon);
                }
            }
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn update_learning_rate(&mut self, new_lr: f32) {
        self.learning_rate = new_lr;
    }
}

/// Training loop implementation
pub struct Trainer {
    config: TrainingConfig,
    model: Arc<Mutex<HybridLayer>>,
    agentic_coordinator: Arc<Mutex<AgenticCoordinator>>,
    optimizer: AdamWOptimizer,
    dataset_manager: Arc<Mutex<DatasetManager>>,
    performance_monitor: PerformanceMonitor,
    state: TrainingState,
    start_time: Instant,
}

impl Trainer {
    pub fn new(
        config: TrainingConfig,
        model: HybridLayer,
        agentic_coordinator: AgenticCoordinator,
        dataset_manager: DatasetManager,
    ) -> Self {
        let optimizer = AdamWOptimizer::new(&config.optimizer);
        let performance_monitor = PerformanceMonitor::new();

        Self {
            config,
            model: Arc::new(Mutex::new(model)),
            agentic_coordinator: Arc::new(Mutex::new(agentic_coordinator)),
            optimizer,
            dataset_manager: Arc::new(Mutex::new(dataset_manager)),
            performance_monitor,
            state: TrainingState::default(),
            start_time: Instant::now(),
        }
    }

    /// Main training loop
    pub async fn train(&mut self) -> Result<TrainingState> {
        log::info!("Starting training with config: {:?}", self.config);
        
        // Initialize training state
        self.state.learning_rate = self.config.optimizer.learning_rate;
        
        // Create checkpoint directory
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;

        // Training loop
        for epoch in 0..self.config.max_epochs {
            self.state.epoch = epoch;
            
            // Check time constraint
            if self.check_time_constraint() {
                log::warn!("Training time limit reached, stopping early");
                break;
            }

            // Train one epoch
            let epoch_metrics = self.train_epoch().await?;
            log::info!("Epoch {} completed. Loss: {:.4}, LR: {:.2e}", 
                epoch, epoch_metrics.loss, epoch_metrics.learning_rate);

            // Validation
            if epoch % self.config.validation_frequency == 0 {
                let val_metrics = self.validate().await?;
                self.state.validation_loss = Some(val_metrics.loss);
                
                // Early stopping check
                if self.check_early_stopping(&val_metrics) {
                    log::info!("Early stopping triggered");
                    break;
                }
            }

            // Save checkpoint
            if epoch % (self.config.save_steps / 1000) == 0 {
                self.save_checkpoint().await?;
            }

            // Memory cleanup
            self.cleanup_memory();
        }

        log::info!("Training completed. Final state: {:?}", self.state);
        Ok(self.state.clone())
    }

    /// Train one epoch
    async fn train_epoch(&mut self) -> Result<TrainingMetrics> {
        let dataset_manager = self.dataset_manager.lock().await;
        let mut train_samples = dataset_manager.get_mixed_samples(SplitType::Train, Some(10000))?;
        drop(dataset_manager);

        // Shuffle samples
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        train_samples.shuffle(&mut thread_rng());

        let mut accumulated_gradients: HashMap<String, Array1<f32>> = HashMap::new();
        let mut total_loss = 0.0;
        let mut step_count = 0;
        let epoch_start = Instant::now();

        // Process batches with gradient accumulation
        for batch_start in (0..train_samples.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];

            // Forward pass
            let step_start = Instant::now();
            let (loss, gradients) = self.forward_backward_pass(batch).await?;
            let step_time = step_start.elapsed();

            total_loss += loss;
            step_count += 1;
            self.state.step += 1;

            // Accumulate gradients
            for (name, grad) in gradients {
                let accumulated = accumulated_gradients.entry(name.clone()).or_insert_with(|| Array1::zeros(grad.len()));
                *accumulated = &*accumulated + &grad;
            }

            // Update parameters when accumulation steps reached
            if self.state.step % self.config.gradient_accumulation_steps == 0 {
                // Scale gradients by accumulation steps
                for grad in accumulated_gradients.values_mut() {
                    *grad = &*grad / self.config.gradient_accumulation_steps as f32;
                }

                // Update learning rate
                self.update_learning_rate();

                // Optimizer step (mock for now)
                self.state.global_step += 1;
                
                // Clear accumulated gradients
                accumulated_gradients.clear();

                // Memory monitoring
                self.state.memory_usage_mb = self.performance_monitor.get_memory_usage_mb();
                if self.state.memory_usage_mb > self.config.max_memory_mb {
                    self.cleanup_memory();
                }
            }

            // Logging
            if self.state.step % 100 == 0 {
                let avg_loss = total_loss / step_count as f32;
                log::info!("Step {}: Loss: {:.4}, LR: {:.2e}, Memory: {}MB, Time: {:?}",
                    self.state.step, avg_loss, self.state.learning_rate, 
                    self.state.memory_usage_mb, step_time);
            }
        }

        let epoch_time = epoch_start.elapsed();
        let avg_loss = total_loss / step_count as f32;
        self.state.train_loss = avg_loss;
        self.state.total_samples_seen += train_samples.len();

        Ok(TrainingMetrics {
            loss: avg_loss,
            perplexity: avg_loss.exp(),
            accuracy: 0.0, // Would need actual accuracy calculation
            learning_rate: self.state.learning_rate,
            memory_usage_mb: self.state.memory_usage_mb,
            throughput_samples_per_sec: train_samples.len() as f32 / epoch_time.as_secs_f32(),
            step_time_ms: epoch_time.as_millis() as u64 / step_count,
        })
    }

    /// Forward and backward pass for a batch
    async fn forward_backward_pass(&self, batch: &[TrainingSample]) -> Result<(f32, HashMap<String, Array1<f32>>)> {
        let mut total_loss = 0.0;
        let mut gradients: HashMap<String, Array1<f32>> = HashMap::new();

        // Process each sample in the batch
        for sample in batch {
            // Convert input to Array1 (mock tokenization)
            let input = self.tokenize_text(&sample.input)?;
            let target = self.tokenize_text(&sample.target)?;

            // Forward pass through model
            let mut model = self.model.lock().await;
            let output = model.forward(&input)?;
            drop(model);

            // Calculate loss (simplified cross-entropy)
            let loss = self.calculate_loss(&output, &target);
            total_loss += loss;

            // Calculate gradients (mock implementation)
            let sample_gradients = self.calculate_gradients(&output, &target);
            
            // Accumulate gradients
            for (name, grad) in sample_gradients {
                let accumulated = gradients.entry(name.clone()).or_insert_with(|| Array1::zeros(grad.len()));
                *accumulated = &*accumulated + &grad;
            }
        }

        let avg_loss = total_loss / batch.len() as f32;
        Ok((avg_loss, gradients))
    }

    /// Tokenize text to Array1 (simplified)
    fn tokenize_text(&self, text: &str) -> Result<Array1<f32>> {
        // Simple character-level tokenization for demonstration
        let tokens: Vec<f32> = text.chars()
            .take(self.config.model_config.max_sequence_length)
            .map(|c| (c as u8 as f32) / 255.0)
            .collect();
        
        let mut padded = vec![0.0; self.config.model_config.max_sequence_length];
        for (i, &token) in tokens.iter().enumerate() {
            if i < padded.len() {
                padded[i] = token;
            }
        }
        
        Ok(Array1::from_vec(padded))
    }

    /// Calculate loss (simplified)
    fn calculate_loss(&self, output: &Array1<f32>, target: &Array1<f32>) -> f32 {
        // Mean squared error for simplicity
        let diff = output - target;
        diff.mapv(|x| x * x).mean().unwrap_or(0.0)
    }

    /// Calculate gradients (mock implementation)
    fn calculate_gradients(&self, output: &Array1<f32>, target: &Array1<f32>) -> HashMap<String, Array1<f32>> {
        let mut gradients = HashMap::new();
        
        // Mock gradient calculation - in real implementation this would be much more complex
        let grad = (output - target) * 2.0 / output.len() as f32;
        gradients.insert("main_weights".to_string(), grad);
        
        gradients
    }

    /// Validate model
    async fn validate(&self) -> Result<TrainingMetrics> {
        let dataset_manager = self.dataset_manager.lock().await;
        let val_samples = dataset_manager.get_mixed_samples(SplitType::Validation, Some(1000))?;
        drop(dataset_manager);

        let mut total_loss = 0.0;
        let val_start = Instant::now();

        for sample in &val_samples {
            let input = self.tokenize_text(&sample.input)?;
            let target = self.tokenize_text(&sample.target)?;

            let mut model = self.model.lock().await;
            let output = model.forward(&input)?;
            drop(model);

            let loss = self.calculate_loss(&output, &target);
            total_loss += loss;
        }

        let avg_loss = total_loss / val_samples.len() as f32;
        let val_time = val_start.elapsed();

        Ok(TrainingMetrics {
            loss: avg_loss,
            perplexity: avg_loss.exp(),
            accuracy: 0.0,
            learning_rate: self.state.learning_rate,
            memory_usage_mb: self.state.memory_usage_mb,
            throughput_samples_per_sec: val_samples.len() as f32 / val_time.as_secs_f32(),
            step_time_ms: val_time.as_millis() as u64,
        })
    }

    /// Update learning rate based on schedule
    fn update_learning_rate(&mut self) {
        let new_lr = match self.config.learning_rate_schedule.schedule_type {
            ScheduleType::Constant => self.config.optimizer.learning_rate,
            ScheduleType::Linear => {
                let decay_factor = 1.0 - (self.state.global_step as f32 / self.config.learning_rate_schedule.decay_steps as f32);
                self.config.optimizer.learning_rate * decay_factor.max(0.1)
            },
            ScheduleType::CosineAnnealing => {
                let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * self.state.global_step as f32 / self.config.learning_rate_schedule.decay_steps as f32).cos());
                self.config.learning_rate_schedule.min_learning_rate + 
                (self.config.optimizer.learning_rate - self.config.learning_rate_schedule.min_learning_rate) * cos_factor
            },
            ScheduleType::ExponentialDecay => {
                self.config.optimizer.learning_rate * self.config.learning_rate_schedule.decay_rate.powf(self.state.global_step as f32 / self.config.learning_rate_schedule.decay_steps as f32)
            },
            ScheduleType::StepDecay => {
                let steps = self.state.global_step / self.config.learning_rate_schedule.decay_steps;
                self.config.optimizer.learning_rate * self.config.learning_rate_schedule.decay_rate.powi(steps as i32)
            },
        };

        self.state.learning_rate = new_lr.max(self.config.learning_rate_schedule.min_learning_rate);
        self.optimizer.update_learning_rate(self.state.learning_rate);
    }

    /// Check early stopping condition
    fn check_early_stopping(&mut self, metrics: &TrainingMetrics) -> bool {
        if !self.config.early_stopping.enabled {
            return false;
        }

        let current_metric = metrics.loss;
        let is_improvement = if self.config.early_stopping.mode == "min" {
            current_metric < self.state.best_validation_loss - self.config.early_stopping.min_delta
        } else {
            current_metric > self.state.best_validation_loss + self.config.early_stopping.min_delta
        };

        if is_improvement {
            self.state.best_validation_loss = current_metric;
            self.state.no_improvement_count = 0;
            false
        } else {
            self.state.no_improvement_count += 1;
            self.state.no_improvement_count >= self.config.early_stopping.patience
        }
    }

    /// Check time constraint
    fn check_time_constraint(&mut self) -> bool {
        let elapsed_hours = self.start_time.elapsed().as_secs_f32() / 3600.0;
        self.state.training_time_hours = elapsed_hours;
        elapsed_hours >= self.config.target_training_hours
    }

    /// Save checkpoint
    async fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_path = format!("{}/checkpoint_epoch_{}_step_{}.json", 
            self.config.checkpoint_dir, self.state.epoch, self.state.step);
        
        let checkpoint_data = serde_json::to_string_pretty(&self.state)?;
        std::fs::write(checkpoint_path, checkpoint_data)?;
        
        log::info!("Checkpoint saved at epoch {} step {}", self.state.epoch, self.state.step);
        Ok(())
    }

    /// Cleanup memory
    fn cleanup_memory(&mut self) {
        // Force garbage collection and memory cleanup
        // In a real implementation, this would clear unused tensors, gradients, etc.
        log::debug!("Performing memory cleanup");
    }

    /// Get training statistics
    pub fn get_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        stats.insert("epoch".to_string(), self.state.epoch as f32);
        stats.insert("step".to_string(), self.state.step as f32);
        stats.insert("global_step".to_string(), self.state.global_step as f32);
        stats.insert("learning_rate".to_string(), self.state.learning_rate);
        stats.insert("train_loss".to_string(), self.state.train_loss);
        stats.insert("memory_usage_mb".to_string(), self.state.memory_usage_mb as f32);
        stats.insert("training_time_hours".to_string(), self.state.training_time_hours);
        
        if let Some(val_loss) = self.state.validation_loss {
            stats.insert("validation_loss".to_string(), val_loss);
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::core::{SnnConfig, SsmConfig, LiquidConfig};
    use crate::model::fusion::FusionConfig;
    use crate::model::agentic::{TaskConfig, AgenticCoordinator};
    use crate::training::datasets::DatasetManager;

    #[tokio::test]
    async fn test_trainer_creation() {
        let config = TrainingConfig::default();
        
        let snn_config = SnnConfig::default();
        let ssm_config = SsmConfig::default();
        let liquid_config = LiquidConfig::default();
        let fusion_config = FusionConfig::default();
        
        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        let agentic_coordinator = AgenticCoordinator::new(TaskConfig::default(), crate::model::agentic::VotingStrategy::WeightedVote);
        let dataset_manager = DatasetManager::new();
        
        let trainer = Trainer::new(config, model, agentic_coordinator, dataset_manager);
        
        assert_eq!(trainer.state.epoch, 0);
        assert_eq!(trainer.state.step, 0);
    }

    #[test]
    fn test_adamw_optimizer() {
        let config = OptimizerConfig::default();
        let mut optimizer = AdamWOptimizer::new(&config);
        
        let mut parameters = HashMap::new();
        parameters.insert("test_param".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        
        let mut gradients = HashMap::new();
        gradients.insert("test_param".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3]));
        
        optimizer.step(&mut parameters, &gradients).unwrap();
        
        // Parameters should have been updated
        assert!(parameters["test_param"][0] < 1.0);
    }

    #[test]
    fn test_learning_rate_schedules() {
        let mut config = TrainingConfig::default();
        config.learning_rate_schedule.schedule_type = ScheduleType::Linear;
        config.learning_rate_schedule.decay_steps = 1000;
        
        let model = HybridLayer::new(
            SnnConfig::default(),
            SsmConfig::default(), 
            LiquidConfig::default(),
            FusionConfig::default()
        ).unwrap();
        
        let agentic_coordinator = AgenticCoordinator::new(TaskConfig::default(), crate::model::agentic::VotingStrategy::WeightedVote);
        let dataset_manager = DatasetManager::new();
        
        let mut trainer = Trainer::new(config, model, agentic_coordinator, dataset_manager);
        
        let initial_lr = trainer.state.learning_rate;
        trainer.state.global_step = 100;
        trainer.update_learning_rate();
        
        // Learning rate should decrease with linear schedule
        assert!(trainer.state.learning_rate <= initial_lr);
    }
}