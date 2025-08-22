//! Validation tests against HumanEval and GSM8K benchmarks
//!
//! This module implements comprehensive testing against standardized benchmarks:
//! - HumanEval: Code generation and programming problem solving
//! - GSM8K: Grade school mathematical reasoning

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::process::Command;

use crate::model::core::*;
use crate::model::fusion::*;
use crate::model::agentic::*;
use crate::model::validation::*;

/// HumanEval problem definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanEvalProblem {
    pub task_id: String,
    pub prompt: String,
    pub canonical_solution: String,
    pub test: String,
    pub entry_point: String,
}

/// GSM8K problem definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSM8KProblem {
    pub question: String,
    pub answer: String,
    pub final_answer: f64,
}

/// Benchmark result for individual problems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub problem_id: String,
    pub success: bool,
    pub generated_solution: String,
    pub execution_result: Option<String>,
    pub error_message: Option<String>,
    pub inference_time_ms: u64,
    pub confidence_score: f32,
    pub validation_passed: bool,
}

/// Aggregate benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub benchmark_name: String,
    pub total_problems: usize,
    pub solved_problems: usize,
    pub accuracy: f32,
    pub average_inference_time_ms: f64,
    pub average_confidence_score: f32,
    pub zero_hallucination_rate: f32,
    pub results: Vec<BenchmarkResult>,
}

/// Benchmark runner configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub max_problems: Option<usize>,
    pub timeout_seconds: u64,
    pub temperature: f32,
    pub enable_validation: bool,
    pub parallel_execution: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            max_problems: Some(10),
            timeout_seconds: 30,
            temperature: 0.1,
            enable_validation: true,
            parallel_execution: false,
        }
    }
}

/// HumanEval benchmark runner
pub struct HumanEvalBenchmark {
    coordinator: AgenticCoordinator,
    validator: OutputValidator,
    config: BenchmarkConfig,
    problems: Vec<HumanEvalProblem>,
}

impl HumanEvalBenchmark {
    pub async fn new(config: BenchmarkConfig) -> Result<Self, crate::UltraFastAiError> {
        let task_config = TaskConfig {
            max_sub_models: 3,
            confidence_threshold: 0.8,
            consensus_threshold: 0.75,
            timeout_duration_ms: config.timeout_seconds * 1000,
            enable_validation: config.enable_validation,
            parallel_execution: config.parallel_execution,
            ..TaskConfig::default()
        };

        let coordinator = AgenticCoordinator::new(task_config, VotingStrategy::ConsensusFiltering);

        let validation_config = ValidationConfig {
            enable_validation: config.enable_validation,
            confidence_threshold: 0.8,
            consistency_threshold: 0.8,
            hallucination_threshold: 0.1,
            ..ValidationConfig::default()
        };

        let validator = OutputValidator::new(validation_config);
        let problems = Self::load_humaneval_problems().await?;

        Ok(Self {
            coordinator,
            validator,
            config,
            problems,
        })
    }

    async fn load_humaneval_problems() -> Result<Vec<HumanEvalProblem>, crate::UltraFastAiError> {
        let sample_problems = vec![
            HumanEvalProblem {
                task_id: "HumanEval/0".to_string(),
                prompt: "def has_close_elements(numbers, threshold):\n    \"\"\" Check if any two numbers are closer than threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    \"\"\"\n".to_string(),
                canonical_solution: "    for i, elem in enumerate(numbers):\n        for j, elem2 in enumerate(numbers):\n            if i != j and abs(elem - elem2) < threshold:\n                return True\n    return False\n".to_string(),
                test: "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True".to_string(),
                entry_point: "has_close_elements".to_string(),
            },
            HumanEvalProblem {
                task_id: "HumanEval/1".to_string(),
                prompt: "def separate_paren_groups(paren_string):\n    \"\"\" Separate groups of balanced parentheses.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n".to_string(),
                canonical_solution: "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n".to_string(),
                test: "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']".to_string(),
                entry_point: "separate_paren_groups".to_string(),
            },
        ];

        Ok(sample_problems)
    }

    pub async fn run_benchmark(&mut self) -> Result<BenchmarkSummary, crate::UltraFastAiError> {
        let mut results = Vec::new();
        let total_problems = self.config.max_problems.unwrap_or(self.problems.len()).min(self.problems.len());
        
        println!("🧪 Running HumanEval benchmark on {} problems...", total_problems);

        for (idx, problem) in self.problems.iter().take(total_problems).enumerate() {
            println!("  Processing problem {}/{}: {}", idx + 1, total_problems, problem.task_id);
            
            let start_time = Instant::now();
            let result = self.solve_problem(problem).await;
            let inference_time = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(mut benchmark_result) => {
                    benchmark_result.inference_time_ms = inference_time;
                    results.push(benchmark_result);
                }
                Err(e) => {
                    results.push(BenchmarkResult {
                        problem_id: problem.task_id.clone(),
                        success: false,
                        generated_solution: String::new(),
                        execution_result: None,
                        error_message: Some(format!("Error: {}", e)),
                        inference_time_ms: inference_time,
                        confidence_score: 0.0,
                        validation_passed: false,
                    });
                }
            }
        }

        Ok(self.calculate_summary("HumanEval", results))
    }

    async fn solve_problem(&mut self, problem: &HumanEvalProblem) -> Result<BenchmarkResult, crate::UltraFastAiError> {
        // Tokenize the problem prompt
        let input = self.tokenize_prompt(&problem.prompt)?;
        
        // Generate solution using agentic coordination
        let generated_output = self.coordinator.execute_task(&input).await?;
        
        // Convert model output back to code
        let generated_solution = self.detokenize_output(&generated_output)?;
        
        // Validate output for hallucination detection
        let validation_result = self.validator.validate_output(
            &generated_solution,
            Some(&problem.prompt),
            Some(&problem.canonical_solution)
        ).await?;

        // Execute the generated code to test correctness
        let execution_result = self.execute_solution(problem, &generated_solution).await;
        
        let success = execution_result.is_ok() && validation_result.confidence_score > 0.7;

        Ok(BenchmarkResult {
            problem_id: problem.task_id.clone(),
            success,
            generated_solution,
            execution_result: execution_result.ok(),
            error_message: execution_result.err(),
            inference_time_ms: 0,
            confidence_score: validation_result.confidence_score,
            validation_passed: validation_result.is_valid,
        })
    }

    async fn execute_solution(&self, problem: &HumanEvalProblem, solution: &str) -> Result<String, String> {
        // Create complete Python code with solution and test
        let full_code = format!("{}\n{}\n{}", problem.prompt, solution, problem.test);
        
        // Write to temporary file
        let temp_file = format!("target/temp_humaneval_{}.py", problem.task_id.replace("/", "_"));
        if let Err(e) = fs::write(&temp_file, &full_code).await {
            return Err(format!("Failed to write temp file: {}", e));
        }

        // Execute the Python code
        let output = Command::new("python3")
            .arg(&temp_file)
            .output()
            .await
            .map_err(|e| format!("Execution failed: {}", e))?;

        // Clean up temp file
        let _ = fs::remove_file(&temp_file).await;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    fn tokenize_prompt(&self, prompt: &str) -> Result<ndarray::Array1<f32>, crate::UltraFastAiError> {
        // Simple character-based tokenization
        let tokens: Vec<f32> = prompt.chars()
            .take(1024)
            .map(|c| (c as u8 as f32) / 255.0)
            .collect();
        
        if tokens.is_empty() {
            Ok(ndarray::Array1::zeros(1))
        } else {
            Ok(ndarray::Array1::from_vec(tokens))
        }
    }

    fn detokenize_output(&self, output: &ndarray::Array1<f32>) -> Result<String, crate::UltraFastAiError> {
        // Convert model output back to text
        let chars: String = output.iter()
            .map(|&x| ((x * 255.0) as u8) as char)
            .filter(|c| c.is_ascii() && !c.is_control())
            .take(512)
            .collect();
        
        Ok(chars)
    }

    fn calculate_summary(&self, name: &str, results: Vec<BenchmarkResult>) -> BenchmarkSummary {
        let total_problems = results.len();
        let solved_problems = results.iter().filter(|r| r.success).count();
        let accuracy = if total_problems > 0 {
            solved_problems as f32 / total_problems as f32
        } else {
            0.0
        };

        let average_inference_time_ms = if total_problems > 0 {
            results.iter().map(|r| r.inference_time_ms as f64).sum::<f64>() / total_problems as f64
        } else {
            0.0
        };

        let average_confidence_score = if total_problems > 0 {
            results.iter().map(|r| r.confidence_score).sum::<f32>() / total_problems as f32
        } else {
            0.0
        };

        let validation_passed_count = results.iter().filter(|r| r.validation_passed).count();
        let zero_hallucination_rate = if total_problems > 0 {
            validation_passed_count as f32 / total_problems as f32
        } else {
            0.0
        };

        BenchmarkSummary {
            benchmark_name: name.to_string(),
            total_problems,
            solved_problems,
            accuracy,
            average_inference_time_ms,
            average_confidence_score,
            zero_hallucination_rate,
            results,
        }
    }
}