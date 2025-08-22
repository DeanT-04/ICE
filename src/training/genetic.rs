//! Genetic algorithms for sub-module evolution
//!
//! Implements genetic algorithms for evolving neural network architectures,
//! optimizing sub-module configurations, and automatic hyperparameter tuning.

use std::collections::HashMap;
use std::sync::Arc;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use ndarray::Array1;

use crate::model::core::{HybridLayer, SnnConfig, SsmConfig, LiquidConfig};
use crate::model::fusion::FusionConfig;
use crate::model::agentic::{AgenticCoordinator, TaskConfig};
use crate::training::trainer::{Trainer, TrainingConfig};
use crate::training::datasets::DatasetManager;
use crate::Result;

/// Genetic algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub elitism_rate: f32,
    pub tournament_size: usize,
    pub fitness_threshold: f32,
    pub convergence_threshold: f32,
    pub max_stagnant_generations: usize,
    pub parallel_evaluation: bool,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            num_generations: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_rate: 0.2,
            tournament_size: 3,
            fitness_threshold: 0.95,
            convergence_threshold: 1e-6,
            max_stagnant_generations: 10,
            parallel_evaluation: true,
        }
    }
}

/// Individual in the genetic algorithm population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub id: String,
    pub genome: Genome,
    pub fitness: f32,
    pub age: usize,
    pub evaluated: bool,
}

/// Genome encoding neural architecture parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub snn_params: SnnGenome,
    pub ssm_params: SsmGenome,
    pub liquid_params: LiquidGenome,
    pub fusion_params: FusionGenome,
    pub training_params: TrainingGenome,
}

/// SNN-specific genome parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnnGenome {
    pub hidden_sizes: Vec<usize>,
    pub threshold: f32,
    pub decay_rate: f32,
    pub refractory_period: u32,
    pub sparse_rate: f32,
}

/// SSM-specific genome parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmGenome {
    pub state_size: usize,
    pub num_layers: usize,
    pub dt_min: f32,
    pub dt_max: f32,
    pub conv_kernel_size: usize,
}

/// Liquid NN-specific genome parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidGenome {
    pub hidden_size: usize,
    pub adaptation_rate: f32,
    pub enable_adaptation: bool,
}

/// Fusion layer genome parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGenome {
    pub hidden_dim: usize,
    pub attention_heads: usize,
    pub dropout_rate: f32,
}

/// Training-specific genome parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingGenome {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub weight_decay: f32,
    pub dropout_rate: f32,
}

/// Population statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
    pub fitness_std: f32,
    pub diversity_score: f32,
    pub convergence_score: f32,
}

/// Genetic algorithm engine
pub struct GeneticEvolution {
    config: GeneticConfig,
    population: Vec<Individual>,
    dataset_manager: Arc<Mutex<DatasetManager>>,
    generation: usize,
    best_individual: Option<Individual>,
    fitness_history: Vec<f32>,
    stagnant_generations: usize,
}

impl GeneticEvolution {
    pub fn new(config: GeneticConfig, dataset_manager: DatasetManager) -> Self {
        Self {
            config,
            population: Vec::new(),
            dataset_manager: Arc::new(Mutex::new(dataset_manager)),
            generation: 0,
            best_individual: None,
            fitness_history: Vec::new(),
            stagnant_generations: 0,
        }
    }

    /// Initialize random population
    pub fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();
        
        for i in 0..self.config.population_size {
            let individual = Individual {
                id: format!("ind_{}", i),
                genome: self.generate_random_genome(),
                fitness: 0.0,
                age: 0,
                evaluated: false,
            };
            self.population.push(individual);
        }
        
        log::info!("Initialized population with {} individuals", self.population.len());
        Ok(())
    }

    /// Generate random genome
    fn generate_random_genome(&self) -> Genome {
        let mut rng = thread_rng();
        
        Genome {
            snn_params: SnnGenome {
                hidden_sizes: vec![
                    rng.gen_range(128..512),
                    rng.gen_range(64..256),
                ],
                threshold: rng.gen_range(0.3..0.8),
                decay_rate: rng.gen_range(0.8..0.95),
                refractory_period: rng.gen_range(1..5),
                sparse_rate: rng.gen_range(0.1..0.3),
            },
            ssm_params: SsmGenome {
                state_size: rng.gen_range(8..32),
                num_layers: rng.gen_range(4..12),
                dt_min: rng.gen_range(0.001..0.01),
                dt_max: rng.gen_range(0.05..0.2),
                conv_kernel_size: rng.gen_range(2..8),
            },
            liquid_params: LiquidGenome {
                hidden_size: rng.gen_range(64..256),
                adaptation_rate: rng.gen_range(0.01..0.1),
                enable_adaptation: rng.gen_bool(0.7),
            },
            fusion_params: FusionGenome {
                hidden_dim: rng.gen_range(128..512),
                attention_heads: [2, 4, 6, 8].choose(&mut rng).copied().unwrap_or(4),
                dropout_rate: rng.gen_range(0.0..0.3),
            },
            training_params: TrainingGenome {
                learning_rate: rng.gen_range(1e-5..1e-3),
                batch_size: [2, 4, 8, 16].choose(&mut rng).copied().unwrap_or(4),
                weight_decay: rng.gen_range(1e-4..1e-2),
                dropout_rate: rng.gen_range(0.0..0.2),
            },
        }
    }

    /// Main evolution loop
    pub async fn evolve(&mut self) -> Result<Individual> {
        if self.population.is_empty() {
            self.initialize_population()?;
        }

        for generation in 0..self.config.num_generations {
            self.generation = generation;
            log::info!("Starting generation {}", generation);

            // Evaluate population
            self.evaluate_population().await?;

            // Calculate statistics
            let stats = self.calculate_stats();
            log::info!("Generation {} stats: Best: {:.4}, Avg: {:.4}, Diversity: {:.4}",
                generation, stats.best_fitness, stats.avg_fitness, stats.diversity_score);

            // Check convergence
            if self.check_convergence(&stats) {
                log::info!("Convergence reached at generation {}", generation);
                break;
            }

            // Selection and reproduction
            let new_population = self.reproduce().await?;
            self.population = new_population;

            // Age individuals
            for individual in &mut self.population {
                individual.age += 1;
            }
        }

        // Return best individual
        self.get_best_individual()
    }

    /// Evaluate fitness of entire population
    async fn evaluate_population(&mut self) -> Result<()> {
        if self.config.parallel_evaluation {
            // Parallel evaluation using tokio tasks
            let mut tasks = Vec::new();
            
            for individual in &mut self.population {
                if !individual.evaluated {
                    let genome = individual.genome.clone();
                    let dataset_manager = self.dataset_manager.clone();
                    let task = tokio::spawn(async move {
                        Self::evaluate_individual_fitness(genome, dataset_manager).await
                    });
                    tasks.push(task);
                }
            }

            // Collect results
            let mut fitness_results = Vec::new();
            for task in tasks {
                let fitness = task.await??;
                fitness_results.push(fitness);
            }

            // Update fitness scores
            let mut result_idx = 0;
            for individual in &mut self.population {
                if !individual.evaluated {
                    individual.fitness = fitness_results[result_idx];
                    individual.evaluated = true;
                    result_idx += 1;
                }
            }
        } else {
            // Sequential evaluation
            for individual in &mut self.population {
                if !individual.evaluated {
                    individual.fitness = Self::evaluate_individual_fitness(
                        individual.genome.clone(),
                        self.dataset_manager.clone()
                    ).await?;
                    individual.evaluated = true;
                }
            }
        }

        Ok(())
    }

    /// Evaluate fitness of single individual
    async fn evaluate_individual_fitness(genome: Genome, dataset_manager: Arc<Mutex<DatasetManager>>) -> Result<f32> {
        // Convert genome to model configuration
        let snn_config = SnnConfig {
            input_size: 768,
            hidden_sizes: genome.snn_params.hidden_sizes,
            output_size: 256,
            threshold: genome.snn_params.threshold,
            decay_rate: genome.snn_params.decay_rate,
            refractory_period: genome.snn_params.refractory_period,
            sparse_rate: genome.snn_params.sparse_rate,
        };

        let ssm_config = SsmConfig {
            input_size: 768,
            state_size: genome.ssm_params.state_size,
            output_size: 256,
            num_layers: genome.ssm_params.num_layers,
            dt_min: genome.ssm_params.dt_min,
            dt_max: genome.ssm_params.dt_max,
            dt_init: "random".to_string(),
            conv_kernel_size: genome.ssm_params.conv_kernel_size,
        };

        let liquid_config = LiquidConfig {
            input_size: 768,
            hidden_size: genome.liquid_params.hidden_size,
            output_size: 256,
            time_constant_min: 0.5,
            time_constant_max: 2.0,
            sensory_tau: 1.0,
            inter_tau: 1.5,
            command_tau: 2.0,
            adaptation_rate: genome.liquid_params.adaptation_rate,
            enable_adaptation: genome.liquid_params.enable_adaptation,
        };

        let fusion_config = FusionConfig {
            input_dims: vec![256, 256, 256],
            output_dim: 256,
            hidden_dim: genome.fusion_params.hidden_dim,
            attention_heads: genome.fusion_params.attention_heads,
            dropout_rate: genome.fusion_params.dropout_rate,
            use_cross_attention: true,
            use_adaptive_weights: true,
            temperature: 1.0,
        };

        // Create model
        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config)?;

        // Create training configuration
        let mut training_config = TrainingConfig::default();
        training_config.optimizer.learning_rate = genome.training_params.learning_rate;
        training_config.batch_size = genome.training_params.batch_size;
        training_config.optimizer.weight_decay = genome.training_params.weight_decay;
        training_config.max_epochs = 3; // Quick evaluation
        training_config.max_steps = Some(100); // Limit training steps

        // Create agentic coordinator
        let task_config = TaskConfig::default();
        let agentic_coordinator = AgenticCoordinator::new(task_config, crate::model::agentic::VotingStrategy::WeightedVote);

        // Create trainer and run brief training
        let dataset_manager_guard = dataset_manager.lock().await;
        let dataset_manager_clone = DatasetManager::new(); // Create new instance for thread safety
        drop(dataset_manager_guard);

        let mut trainer = Trainer::new(training_config, model, agentic_coordinator, dataset_manager_clone);
        
        // Quick training evaluation
        let final_state = trainer.train().await?;
        
        // Calculate fitness based on multiple criteria
        let loss_fitness = 1.0 / (1.0 + final_state.train_loss); // Lower loss = higher fitness
        let memory_fitness = if final_state.memory_usage_mb <= 7500 { 1.0 } else { 0.5 }; // Memory constraint
        let time_fitness = if final_state.training_time_hours <= 0.1 { 1.0 } else { 0.8 }; // Quick training
        
        let fitness = (loss_fitness * 0.6 + memory_fitness * 0.3 + time_fitness * 0.1).min(1.0);
        
        Ok(fitness)
    }

    /// Selection and reproduction
    async fn reproduce(&mut self) -> Result<Vec<Individual>> {
        let mut new_population = Vec::new();

        // Elitism - keep best individuals
        let elite_count = (self.config.population_size as f32 * self.config.elitism_rate) as usize;
        let mut sorted_population = self.population.clone();
        sorted_population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        for i in 0..elite_count {
            new_population.push(sorted_population[i].clone());
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < self.config.population_size {
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            let mut offspring = if thread_rng().gen::<f32>() < self.config.crossover_rate {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };

            if thread_rng().gen::<f32>() < self.config.mutation_rate {
                self.mutate(&mut offspring);
            }

            offspring.id = format!("gen{}_ind{}", self.generation + 1, new_population.len());
            offspring.fitness = 0.0;
            offspring.age = 0;
            offspring.evaluated = false;

            new_population.push(offspring);
        }

        Ok(new_population)
    }

    /// Tournament selection
    fn tournament_selection(&self) -> Individual {
        let mut rng = thread_rng();
        let mut tournament: Vec<_> = self.population.choose_multiple(&mut rng, self.config.tournament_size).collect();
        tournament.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        tournament[0].clone()
    }

    /// Crossover operation
    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> Result<Individual> {
        let mut rng = thread_rng();
        let mut offspring_genome = parent1.genome.clone();

        // SNN crossover
        if rng.gen_bool(0.5) {
            offspring_genome.snn_params.threshold = parent2.genome.snn_params.threshold;
        }
        if rng.gen_bool(0.5) {
            offspring_genome.snn_params.sparse_rate = parent2.genome.snn_params.sparse_rate;
        }

        // SSM crossover
        if rng.gen_bool(0.5) {
            offspring_genome.ssm_params.state_size = parent2.genome.ssm_params.state_size;
        }
        if rng.gen_bool(0.5) {
            offspring_genome.ssm_params.num_layers = parent2.genome.ssm_params.num_layers;
        }

        // Liquid NN crossover
        if rng.gen_bool(0.5) {
            offspring_genome.liquid_params.hidden_size = parent2.genome.liquid_params.hidden_size;
        }
        if rng.gen_bool(0.5) {
            offspring_genome.liquid_params.adaptation_rate = parent2.genome.liquid_params.adaptation_rate;
        }

        // Fusion crossover
        if rng.gen_bool(0.5) {
            offspring_genome.fusion_params.attention_heads = parent2.genome.fusion_params.attention_heads;
        }

        // Training crossover
        if rng.gen_bool(0.5) {
            offspring_genome.training_params.learning_rate = parent2.genome.training_params.learning_rate;
        }

        Ok(Individual {
            id: String::new(), // Will be set later
            genome: offspring_genome,
            fitness: 0.0,
            age: 0,
            evaluated: false,
        })
    }

    /// Mutation operation
    fn mutate(&self, individual: &mut Individual) {
        let mut rng = thread_rng();

        // SNN mutations
        if rng.gen_bool(0.3) {
            individual.genome.snn_params.threshold += rng.gen_range(-0.1..0.1);
            individual.genome.snn_params.threshold = individual.genome.snn_params.threshold.clamp(0.1, 1.0);
        }
        if rng.gen_bool(0.3) {
            individual.genome.snn_params.sparse_rate += rng.gen_range(-0.05..0.05);
            individual.genome.snn_params.sparse_rate = individual.genome.snn_params.sparse_rate.clamp(0.05, 0.5);
        }

        // SSM mutations
        if rng.gen_bool(0.3) {
            individual.genome.ssm_params.state_size = (individual.genome.ssm_params.state_size as i32 + rng.gen_range(-4..4)).max(4) as usize;
        }
        if rng.gen_bool(0.3) {
            individual.genome.ssm_params.num_layers = (individual.genome.ssm_params.num_layers as i32 + rng.gen_range(-2..2)).clamp(2, 16) as usize;
        }

        // Liquid NN mutations
        if rng.gen_bool(0.3) {
            individual.genome.liquid_params.adaptation_rate += rng.gen_range(-0.02..0.02);
            individual.genome.liquid_params.adaptation_rate = individual.genome.liquid_params.adaptation_rate.clamp(0.001, 0.2);
        }

        // Training mutations
        if rng.gen_bool(0.3) {
            individual.genome.training_params.learning_rate *= rng.gen_range(0.5..2.0);
            individual.genome.training_params.learning_rate = individual.genome.training_params.learning_rate.clamp(1e-6, 1e-2);
        }
    }

    /// Calculate population statistics
    fn calculate_stats(&self) -> PopulationStats {
        let fitness_values: Vec<f32> = self.population.iter().map(|ind| ind.fitness).collect();
        
        let best_fitness = fitness_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let worst_fitness = fitness_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let avg_fitness = fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
        
        let variance = fitness_values.iter()
            .map(|&x| (x - avg_fitness).powi(2))
            .sum::<f32>() / fitness_values.len() as f32;
        let fitness_std = variance.sqrt();

        // Calculate diversity (simplified)
        let diversity_score = fitness_std / avg_fitness.max(1e-8);

        // Calculate convergence
        let convergence_score = if self.fitness_history.len() > 5 {
            let recent_avg = self.fitness_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let old_avg = self.fitness_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;
            (recent_avg - old_avg).abs()
        } else {
            1.0
        };

        PopulationStats {
            generation: self.generation,
            best_fitness,
            avg_fitness,
            worst_fitness,
            fitness_std,
            diversity_score,
            convergence_score,
        }
    }

    /// Check convergence criteria
    fn check_convergence(&mut self, stats: &PopulationStats) -> bool {
        self.fitness_history.push(stats.best_fitness);

        // Check fitness threshold
        if stats.best_fitness >= self.config.fitness_threshold {
            return true;
        }

        // Check convergence
        if stats.convergence_score < self.config.convergence_threshold {
            self.stagnant_generations += 1;
        } else {
            self.stagnant_generations = 0;
        }

        self.stagnant_generations >= self.config.max_stagnant_generations
    }

    /// Get best individual from population
    fn get_best_individual(&self) -> Result<Individual> {
        self.population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned()
            .ok_or_else(|| crate::UltraFastAiError::OptimizationError("No individuals in population".to_string()))
    }

    /// Get evolution statistics
    pub fn get_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        if !self.population.is_empty() {
            let fitness_values: Vec<f32> = self.population.iter().map(|ind| ind.fitness).collect();
            let best_fitness = fitness_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let avg_fitness = fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
            
            stats.insert("generation".to_string(), self.generation as f32);
            stats.insert("best_fitness".to_string(), best_fitness);
            stats.insert("avg_fitness".to_string(), avg_fitness);
            stats.insert("population_size".to_string(), self.population.len() as f32);
            stats.insert("stagnant_generations".to_string(), self.stagnant_generations as f32);
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::datasets::DatasetManager;

    #[test]
    fn test_genetic_config_default() {
        let config = GeneticConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.num_generations, 50);
        assert!(config.mutation_rate > 0.0);
        assert!(config.crossover_rate > 0.0);
    }

    #[test]
    fn test_genome_generation() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig::default();
        let evolution = GeneticEvolution::new(config, dataset_manager);
        
        let genome = evolution.generate_random_genome();
        
        assert!(!genome.snn_params.hidden_sizes.is_empty());
        assert!(genome.snn_params.threshold > 0.0);
        assert!(genome.ssm_params.state_size > 0);
        assert!(genome.liquid_params.hidden_size > 0);
    }

    #[tokio::test]
    async fn test_population_initialization() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig::default();
        let mut evolution = GeneticEvolution::new(config.clone(), dataset_manager);
        
        evolution.initialize_population().unwrap();
        
        assert_eq!(evolution.population.len(), config.population_size);
        assert!(evolution.population.iter().all(|ind| !ind.evaluated));
    }

    #[test]
    fn test_crossover() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig::default();
        let evolution = GeneticEvolution::new(config, dataset_manager);
        
        let parent1 = Individual {
            id: "p1".to_string(),
            genome: evolution.generate_random_genome(),
            fitness: 0.8,
            age: 0,
            evaluated: true,
        };
        
        let parent2 = Individual {
            id: "p2".to_string(),
            genome: evolution.generate_random_genome(),
            fitness: 0.7,
            age: 0,
            evaluated: true,
        };
        
        let offspring = evolution.crossover(&parent1, &parent2).unwrap();
        
        // Offspring should have mixed traits
        assert!(offspring.fitness == 0.0); // Not evaluated yet
        assert!(!offspring.evaluated);
    }

    #[test]
    fn test_mutation() {
        let dataset_manager = DatasetManager::new();
        let config = GeneticConfig::default();
        let evolution = GeneticEvolution::new(config, dataset_manager);
        
        let mut individual = Individual {
            id: "test".to_string(),
            genome: evolution.generate_random_genome(),
            fitness: 0.5,
            age: 0,
            evaluated: true,
        };
        
        let _original_threshold = individual.genome.snn_params.threshold;
        evolution.mutate(&mut individual);
        
        // Some parameters might have changed (probabilistic)
        assert!(!individual.evaluated); // Should be marked for re-evaluation
    }
}