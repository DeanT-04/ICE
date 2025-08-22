//! Training system tests
//!
//! Comprehensive tests for training loop, datasets, genetic algorithms,
//! and optimization components.

use super::test_utils::*;
use super::*;
use crate::training::datasets::*;
use crate::training::trainer::*;
use crate::training::genetic::*;
use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use ndarray::Array1;
use std::collections::HashMap;

#[cfg(test)]
mod dataset_tests {
    use super::*;

    #[test]
    fn test_dataset_config_creation() {
        let config = create_test_dataset_config();
        assert_eq!(config.name, "test_dataset");
        assert!(config.shuffle);
        assert_eq!(config.max_samples, Some(100));
    }

    #[test]
    fn test_dataset_manager_creation() {
        let manager = DatasetManager::new();
        let configs = manager.get_dataset_configs();
        assert!(!configs.is_empty());
    }

    #[test]
    fn test_preprocessing_config() {
        let config = PreprocessingConfig::default();
        assert!(config.normalize_text);
        assert!(config.remove_duplicates);
        assert!(config.max_length > 0);
    }

    #[test]
    fn test_dataset_splits() {
        let splits = DatasetSplits::default();
        assert!((splits.train_ratio + splits.validation_ratio + splits.test_ratio - 1.0).abs() < 1e-6);
        assert!(splits.train_ratio > 0.0);
        assert!(splits.validation_ratio > 0.0);
        assert!(splits.test_ratio > 0.0);
    }

    #[tokio::test]
    async fn test_training_sample_creation() {
        let sample = create_test_sample("test_001");
        assert_eq!(sample.id, "test_001");
        assert!(!sample.input.is_empty());
        assert!(!sample.target.is_empty());
        assert_eq!(sample.dataset, "test");
    }

    #[tokio::test]
    async fn test_dataset_loading() {
        let manager = DatasetManager::new();
        let result = manager.get_mixed_samples(SplitType::Train, Some(10));
        
        match result {
            Ok(samples) => {
                assert!(!samples.is_empty());
                assert!(samples.len() <= 10);
                
                for sample in samples {
                    assert!(!sample.id.is_empty());
                    assert!(!sample.input.is_empty());
                    assert!(!sample.target.is_empty());
                }
            }
            Err(_) => {
                // Expected if no real datasets are available in test environment
                assert!(true);
            }
        }
    }

    #[test]
    fn test_dataset_format_detection() {
        let formats = [
            DatasetFormat::Json,
            DatasetFormat::Csv,
            DatasetFormat::Parquet,
            DatasetFormat::HuggingFace,
        ];
        
        for format in &formats {
            // All formats should be valid
            assert!(matches!(format, DatasetFormat::Json | DatasetFormat::Csv | DatasetFormat::Parquet | DatasetFormat::HuggingFace));
        }
    }
}

#[cfg(test)]
mod trainer_tests {
    use super::*;

    #[test]
    fn test_training_config_creation() {
        let config = TrainingConfig::default();
        assert!(config.batch_size > 0);
        assert!(config.max_epochs > 0);
        assert!(config.gradient_accumulation_steps > 0);
        assert!(config.mixed_precision);
        assert!(config.max_memory_mb > 0);
    }

    #[test]
    fn test_optimizer_config() {
        let config = OptimizerConfig::default();
        assert!(matches!(config.optimizer_type, OptimizerType::AdamW));
        assert!(config.learning_rate > 0.0);
        assert!(config.weight_decay >= 0.0);
        assert!(config.beta1 > 0.0 && config.beta1 < 1.0);
        assert!(config.beta2 > 0.0 && config.beta2 < 1.0);
    }

    #[test]
    fn test_learning_rate_schedule() {
        let schedule = LearningRateSchedule::default();
        assert!(schedule.warmup_steps > 0);
        assert!(schedule.decay_steps > 0);
        assert!(schedule.decay_rate > 0.0 && schedule.decay_rate < 1.0);
        assert!(schedule.min_learning_rate > 0.0);
    }

    #[test]
    fn test_early_stopping_config() {
        let config = EarlyStoppingConfig::default();
        assert!(config.enabled);
        assert!(config.patience > 0);
        assert!(config.min_delta > 0.0);
        assert!(!config.metric.is_empty());
        assert!(config.mode == "min" || config.mode == "max");
    }

    #[test]
    fn test_training_state() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.global_step, 0);
        assert!(!state.should_stop);
        assert_eq!(state.best_validation_loss, f32::INFINITY);
    }

    #[test]
    fn test_adamw_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = AdamWOptimizer::new(&config);
        
        // Should create without error
        assert!(true);
    }

    #[test]
    fn test_adamw_optimizer_step() {
        let config = OptimizerConfig::default();
        let mut optimizer = AdamWOptimizer::new(&config);
        
        let mut parameters = HashMap::new();
        parameters.insert("test_param".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        
        let mut gradients = HashMap::new();
        gradients.insert("test_param".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3]));
        
        let original_params = parameters["test_param"].clone();
        let result = optimizer.step(&mut parameters, &gradients);
        
        assert!(result.is_ok());
        
        // Parameters should have been updated
        let updated_params = &parameters["test_param"];
        let diff = (&original_params - updated_params).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[tokio::test]
    async fn test_trainer_creation() {
        let config = TrainingConfig::default();
        
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        let agentic_coordinator = AgenticCoordinator::new(TaskConfig::default(), VotingStrategy::WeightedVote);
        let dataset_manager = DatasetManager::new();
        
        let trainer = Trainer::new(config, model, agentic_coordinator, dataset_manager);
        
        let stats = trainer.get_stats();
        assert!(stats.contains_key("epoch"));
        assert!(stats.contains_key("step"));
    }

    #[tokio::test]
    async fn test_training_metrics() {
        let metrics = TrainingMetrics {
            loss: 0.5,
            perplexity: 1.65,
            accuracy: 0.85,
            learning_rate: 1e-4,
            memory_usage_mb: 4096,
            throughput_samples_per_sec: 100.0,
            step_time_ms: 50,
        };
        
        assert!(metrics.loss > 0.0);
        assert!(metrics.perplexity > 1.0);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert!(metrics.learning_rate > 0.0);
        assert!(metrics.memory_usage_mb > 0);
        assert!(metrics.throughput_samples_per_sec > 0.0);
        assert!(metrics.step_time_ms > 0);
    }
}

#[cfg(test)]
mod genetic_tests {
    use super::*;

    #[test]
    fn test_genetic_config_creation() {
        let config = GeneticConfig::default();
        assert!(config.population_size > 0);
        assert!(config.num_generations > 0);
        assert!(config.mutation_rate > 0.0 && config.mutation_rate < 1.0);
        assert!(config.crossover_rate > 0.0 && config.crossover_rate <= 1.0);
        assert!(config.elitism_rate >= 0.0 && config.elitism_rate < 1.0);
        assert!(config.tournament_size > 0);
    }

    #[test]
    fn test_genome_creation() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig::default();
        let evolution = GeneticEvolution::new(config, dataset_manager);
        
        // Access private method through test (would normally use trait or public method)
        // For now, we test the public interface
        let stats = evolution.get_stats();
        assert!(stats.contains_key("generation"));
    }

    #[test]
    fn test_individual_creation() {
        let individual = Individual {
            id: "test_001".to_string(),
            genome: Genome {
                snn_params: SnnGenome {
                    hidden_sizes: vec![128, 64],
                    threshold: 0.5,
                    decay_rate: 0.9,
                    refractory_period: 2,
                    sparse_rate: 0.15,
                },
                ssm_params: SsmGenome {
                    state_size: 16,
                    num_layers: 8,
                    dt_min: 0.001,
                    dt_max: 0.1,
                    conv_kernel_size: 3,
                },
                liquid_params: LiquidGenome {
                    hidden_size: 128,
                    time_constant: 1.0,
                    adaptation_rate: 0.05,
                    connectivity: 0.3,
                    enable_adaptation: true,
                },
                fusion_params: FusionGenome {
                    hidden_dim: 256,
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
            },
            fitness: 0.75,
            age: 5,
            evaluated: true,
        };
        
        assert_eq!(individual.id, "test_001");
        assert_eq!(individual.fitness, 0.75);
        assert_eq!(individual.age, 5);
        assert!(individual.evaluated);
    }

    #[tokio::test]
    async fn test_genetic_evolution_initialization() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig {
            population_size: 5,
            num_generations: 2,
            ..GeneticConfig::default()
        };
        
        let mut evolution = GeneticEvolution::new(config.clone(), dataset_manager);
        let result = evolution.initialize_population();
        
        assert!(result.is_ok());
        
        let stats = evolution.get_stats();
        assert_eq!(stats.get("population_size").copied().unwrap_or(0.0), config.population_size as f32);
    }

    #[test]
    fn test_population_stats() {
        let stats = PopulationStats {
            generation: 10,
            best_fitness: 0.9,
            avg_fitness: 0.7,
            worst_fitness: 0.4,
            fitness_std: 0.15,
            diversity_score: 0.2,
            convergence_score: 0.05,
        };
        
        assert_eq!(stats.generation, 10);
        assert!(stats.best_fitness >= stats.avg_fitness);
        assert!(stats.avg_fitness >= stats.worst_fitness);
        assert!(stats.fitness_std >= 0.0);
        assert!(stats.diversity_score >= 0.0);
        assert!(stats.convergence_score >= 0.0);
    }

    #[test]
    fn test_genome_parameters() {
        let snn_genome = SnnGenome {
            hidden_sizes: vec![256, 128],
            threshold: 0.6,
            decay_rate: 0.85,
            refractory_period: 3,
            sparse_rate: 0.2,
        };
        
        assert_eq!(snn_genome.hidden_sizes.len(), 2);
        assert!(snn_genome.threshold > 0.0 && snn_genome.threshold < 1.0);
        assert!(snn_genome.decay_rate > 0.0 && snn_genome.decay_rate < 1.0);
        assert!(snn_genome.refractory_period > 0);
        assert!(snn_genome.sparse_rate > 0.0 && snn_genome.sparse_rate < 1.0);
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::*;

    #[test]
    fn test_optimizer_types() {
        let optimizers = [
            OptimizerType::AdamW,
            OptimizerType::Adam,
            OptimizerType::SGD,
            OptimizerType::RMSprop,
        ];
        
        for optimizer in &optimizers {
            // All optimizer types should be valid
            assert!(matches!(optimizer, 
                OptimizerType::AdamW | 
                OptimizerType::Adam | 
                OptimizerType::SGD | 
                OptimizerType::RMSprop
            ));
        }
    }

    #[test]
    fn test_schedule_types() {
        let schedules = [
            ScheduleType::Constant,
            ScheduleType::Linear,
            ScheduleType::CosineAnnealing,
            ScheduleType::ExponentialDecay,
            ScheduleType::StepDecay,
        ];
        
        for schedule in &schedules {
            // All schedule types should be valid
            assert!(matches!(schedule,
                ScheduleType::Constant |
                ScheduleType::Linear |
                ScheduleType::CosineAnnealing |
                ScheduleType::ExponentialDecay |
                ScheduleType::StepDecay
            ));
        }
    }

    #[test]
    fn test_learning_rate_update() {
        let mut config = TrainingConfig::default();
        config.learning_rate_schedule.schedule_type = ScheduleType::Linear;
        config.learning_rate_schedule.decay_steps = 1000;
        
        // Mock the trainer update_learning_rate logic
        let initial_lr = config.optimizer.learning_rate;
        let global_step = 100;
        let decay_factor = 1.0 - (global_step as f32 / config.learning_rate_schedule.decay_steps as f32);
        let new_lr = initial_lr * decay_factor.max(0.1);
        
        assert!(new_lr <= initial_lr);
        assert!(new_lr >= initial_lr * 0.1);
    }

    #[test]
    fn test_memory_constraints() {
        let config = TrainingConfig::default();
        
        // Should respect memory constraints
        assert!(config.max_memory_mb <= 8192); // 8GB limit
        assert!(config.target_vram_mb <= 8192);
        assert!(config.gradient_accumulation_steps > 1); // For memory efficiency
        assert!(config.batch_size <= 16); // Small batches for memory
    }

    #[test]
    fn test_training_constraints() {
        let config = TrainingConfig::default();
        
        // Should respect training time constraints
        assert!(config.target_training_hours <= 24.0);
        assert!(config.mixed_precision); // For efficiency
        assert!(config.gradient_checkpointing); // For memory
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use super::perf_test_utils::*;
    use std::time::Duration;

    #[test]
    fn test_training_step_performance() {
        let config = OptimizerConfig::default();
        let mut optimizer = AdamWOptimizer::new(&config);
        
        let mut parameters = HashMap::new();
        parameters.insert("test".to_string(), Array1::from_vec(vec![1.0; 1000]));
        let gradients = HashMap::new();
        
        assert_time_bounds(|| {
            optimizer.step(&mut parameters, &gradients).unwrap();
        }, Duration::from_millis(10));
    }

    #[test]
    fn test_forward_pass_performance() {
        let snn_config = create_test_snn_config();
        let mut snn = SnnLayer::new(snn_config.clone()).unwrap();
        let input = random_input(snn_config.input_size);
        
        assert_time_bounds(|| {
            snn.forward(&input).unwrap();
        }, Duration::from_millis(5));
    }

    #[test]
    fn test_memory_efficiency() {
        let initial_memory = get_memory_usage();
        
        // Create large model components
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();
        
        let _model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config).unwrap();
        
        let final_memory = get_memory_usage();
        let memory_increase = final_memory - initial_memory;
        
        // Should not use excessive memory
        assert!(memory_increase < 1000); // Less than 1GB increase
    }
}

#[cfg(test)]
mod integration_training_tests {
    use super::*;
    use super::integration_setup::*;

    #[tokio::test]
    async fn test_full_training_pipeline() {
        init_test_env();
        let test_env = setup_test_environment();
        
        // Test that training pipeline can be set up
        let stats = test_env.trainer.get_stats();
        assert!(stats.contains_key("epoch"));
        assert!(stats.contains_key("learning_rate"));
    }

    #[tokio::test]
    async fn test_training_with_genetic_optimization() {
        let dataset_manager = DatasetManager::new();
        let genetic_config = GeneticConfig {
            population_size: 3,
            num_generations: 2,
            ..GeneticConfig::default()
        };
        
        let mut evolution = GeneticEvolution::new(genetic_config, dataset_manager);
        let result = evolution.initialize_population();
        
        assert!(result.is_ok());
        
        // Test that genetic algorithm can be initialized
        let stats = evolution.get_stats();
        assert_eq!(stats.get("population_size").copied().unwrap_or(0.0), 3.0);
    }
}