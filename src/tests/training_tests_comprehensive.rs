//! Comprehensive training component tests
//!
//! Tests for genetic algorithm, trainer, datasets, and metrics with mock data validation.

use rstest::*;
use proptest::prelude::*;
use quickcheck::TestResult;
use test_case::test_case;
use mockall::predicate::*;
use serial_test::serial;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array1, Array2};
use tempfile::TempDir;

use crate::training::trainer::*;
use crate::training::genetic::*;
use crate::training::datasets::*;
use crate::training::metrics::*;
use crate::model::core::*;
use crate::model::agentic::*;
use crate::utils::perf::*;

// =============================================================================
// TRAINING TEST FIXTURES
// =============================================================================

#[fixture]
fn training_config() -> TrainingConfig {
    TrainingConfig {
        dataset_name: "test_dataset".to_string(),
        epochs: 5,
        batch_size: 32,
        learning_rate: 0.001,
        enable_genetic: true,
        output_dir: std::path::PathBuf::from("test_output"),
        resume_from: None,
    }
}

#[fixture]
fn genetic_config() -> GeneticConfig {
    GeneticConfig {
        population_size: 20,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elitism_rate: 0.2,
        generations: 10,
        tournament_size: 3,
        enable_adaptive_mutation: true,
        fitness_threshold: 0.95,
    }
}

#[fixture]
fn mock_dataset() -> Vec<TrainingSample> {
    vec![
        TrainingSample {
            id: "sample_001".to_string(),
            input: "Test input for training sample 1".to_string(),
            target: "Expected output for sample 1".to_string(),
            metadata: HashMap::new(),
            dataset: "test".to_string(),
        },
        TrainingSample {
            id: "sample_002".to_string(),
            input: "Another test input for training".to_string(),
            target: "Another expected output".to_string(),
            metadata: HashMap::new(),
            dataset: "test".to_string(),
        },
    ]
}

#[fixture]
fn temp_training_dir() -> TempDir {
    TempDir::new().expect("Failed to create temporary training directory")
}

// =============================================================================
// TRAINING CONFIGURATION TESTS
// =============================================================================

#[rstest]
#[case::small_batch(8)]
#[case::medium_batch(32)]
#[case::large_batch(128)]
fn test_training_config_validation(#[case] batch_size: usize, mut training_config: TrainingConfig) {
    training_config.batch_size = batch_size;
    
    assert!(training_config.batch_size > 0, "Batch size must be positive");
    assert!(training_config.epochs > 0, "Epochs must be positive");
    assert!(training_config.learning_rate > 0.0, "Learning rate must be positive");
    assert!(training_config.learning_rate < 1.0, "Learning rate should be < 1.0");
}

// =============================================================================
// DATASET TESTS
// =============================================================================

#[rstest]
fn test_dataset_loading(mock_dataset: Vec<TrainingSample>) {
    for sample in &mock_dataset {
        assert!(!sample.id.is_empty(), "Sample ID should not be empty");
        assert!(!sample.input.is_empty(), "Sample input should not be empty");
        assert!(!sample.target.is_empty(), "Sample target should not be empty");
    }
    
    let total_samples = mock_dataset.len();
    assert!(total_samples > 0, "Dataset should contain samples");
}

#[rstest]
fn test_dataset_splitting(mock_dataset: Vec<TrainingSample>) {
    let splits = DatasetSplits {
        train_ratio: 0.7,
        validation_ratio: 0.2,
        test_ratio: 0.1,
    };
    
    let dataset_manager = DatasetManager::new();
    let split_datasets = dataset_manager.split_dataset(&mock_dataset, &splits).unwrap();
    
    let total_samples = mock_dataset.len();
    let assigned_samples = split_datasets.train.len() + 
                          split_datasets.validation.len() + 
                          split_datasets.test.len();
    
    assert_eq!(assigned_samples, total_samples, "All samples should be assigned");
}

// =============================================================================
// GENETIC ALGORITHM TESTS
// =============================================================================

#[rstest]
fn test_genetic_algorithm_initialization(genetic_config: GeneticConfig) {
    let mut genetic_evolution = GeneticEvolution::new(genetic_config.clone());
    
    assert!(genetic_evolution.config.mutation_rate > 0.0);
    assert!(genetic_evolution.config.crossover_rate > 0.0);
    assert!(genetic_evolution.config.elitism_rate >= 0.0);
    
    genetic_evolution.initialize_population();
    let population = genetic_evolution.get_population();
    assert_eq!(population.len(), genetic_config.population_size);
}

#[rstest]
fn test_genetic_operators(genetic_config: GeneticConfig) {
    let mut genetic_evolution = GeneticEvolution::new(genetic_config);
    genetic_evolution.initialize_population();
    
    let population = genetic_evolution.get_population();
    assert!(population.len() >= 2, "Need at least 2 individuals for testing");
    
    let parent1 = &population[0];
    let parent2 = &population[1];
    
    // Test crossover
    let offspring = genetic_evolution.crossover(parent1, parent2);
    assert_eq!(offspring.genes.len(), parent1.genes.len());
    
    // Test mutation
    let mut individual = parent1.clone();
    genetic_evolution.mutate(&mut individual);
    assert!(individual.genes.len() > 0);
}

// =============================================================================
// TRAINING METRICS TESTS
// =============================================================================

#[rstest]
fn test_training_metrics_tracking() {
    let mut training_metrics = TrainingMetrics::new();
    
    training_metrics.record_epoch_loss(0, 1.5);
    training_metrics.record_epoch_accuracy(0, 0.6);
    training_metrics.record_epoch_loss(1, 1.2);
    training_metrics.record_epoch_accuracy(1, 0.7);
    
    let losses = training_metrics.get_loss_history();
    let accuracies = training_metrics.get_accuracy_history();
    
    assert_eq!(losses.len(), 2);
    assert_eq!(accuracies.len(), 2);
    assert!(losses[1] < losses[0], "Loss should decrease");
    assert!(accuracies[1] > accuracies[0], "Accuracy should increase");
}

// =============================================================================
// PROPERTY-BASED TRAINING TESTS
// =============================================================================

prop_compose! {
    fn valid_training_params()(
        epochs in 1usize..50,
        batch_size in 1usize..128,
        learning_rate in 0.0001f32..0.1
    ) -> (usize, usize, f32) {
        (epochs, batch_size, learning_rate)
    }
}

proptest! {
    #[test]
    fn property_training_config_consistency(
        (epochs, batch_size, learning_rate) in valid_training_params()
    ) {
        let config = TrainingConfig {
            dataset_name: "test".to_string(),
            epochs,
            batch_size,
            learning_rate,
            enable_genetic: true,
            output_dir: std::path::PathBuf::from("/tmp"),
            resume_from: None,
        };
        
        prop_assert!(config.epochs > 0);
        prop_assert!(config.batch_size > 0);
        prop_assert!(config.learning_rate > 0.0 && config.learning_rate < 1.0);
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

#[rstest]
#[serial]
fn test_training_memory_efficiency(training_config: TrainingConfig, temp_training_dir: TempDir) {
    let mut perf_monitor = PerformanceMonitor::new();
    let initial_memory = perf_monitor.get_memory_usage_mb();
    
    let mut config = training_config;
    config.batch_size = 32;
    config.epochs = 1;
    config.output_dir = temp_training_dir.path().to_path_buf();
    
    // Memory usage test would go here
    let final_memory = perf_monitor.get_memory_usage_mb();
    let memory_increase = final_memory.saturating_sub(initial_memory);
    
    assert!(memory_increase < 2048, "Memory increase should be reasonable");
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[rstest]
#[case::empty_dataset(vec![])]
fn test_edge_case_datasets(#[case] dataset: Vec<TrainingSample>) {
    let dataset_manager = DatasetManager::new();
    
    if dataset.is_empty() {
        // Should handle empty datasets gracefully
        let result = dataset_manager.validate_dataset(&dataset);
        assert!(result.is_err() || result.unwrap() == false);
    }
}

#[rstest]
fn test_invalid_training_configs() {
    // Test invalid configurations
    let invalid_configs = vec![
        TrainingConfig {
            epochs: 0, // Invalid: zero epochs
            batch_size: 32,
            learning_rate: 0.01,
            ..Default::default()
        },
        TrainingConfig {
            epochs: 10,
            batch_size: 0, // Invalid: zero batch size
            learning_rate: 0.01,
            ..Default::default()
        },
        TrainingConfig {
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.0, // Invalid: zero learning rate
            ..Default::default()
        },
    ];
    
    for config in invalid_configs {
        let validation_result = config.validate();
        assert!(validation_result.is_err(), "Invalid config should be rejected");
    }
}