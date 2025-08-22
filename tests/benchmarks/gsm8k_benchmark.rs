//! GSM8K mathematical reasoning benchmark
//!
//! Tests the model's ability to solve grade school math problems
//! with step-by-step reasoning and numerical accuracy.

use std::time::Instant;
use serde::{Deserialize, Serialize};
use regex::Regex;

use crate::model::agentic::*;
use crate::model::validation::*;
use super::humaneval_benchmark::{BenchmarkResult, BenchmarkSummary, BenchmarkConfig};

/// GSM8K problem definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GSM8KProblem {
    pub question: String,
    pub answer: String,
    pub final_answer: f64,
}

/// GSM8K benchmark runner
pub struct GSM8KBenchmark {
    coordinator: AgenticCoordinator,
    validator: OutputValidator,
    config: BenchmarkConfig,
    problems: Vec<GSM8KProblem>,
}

impl GSM8KBenchmark {
    pub async fn new(config: BenchmarkConfig) -> Result<Self, crate::UltraFastAiError> {
        let task_config = TaskConfig {
            max_sub_models: 3,
            confidence_threshold: 0.85,
            consensus_threshold: 0.8,
            timeout_duration_ms: config.timeout_seconds * 1000,
            enable_validation: config.enable_validation,
            parallel_execution: config.parallel_execution,
            ..TaskConfig::default()
        };

        let coordinator = AgenticCoordinator::new(task_config, VotingStrategy::WeightedVote);

        let validation_config = ValidationConfig {
            enable_validation: config.enable_validation,
            confidence_threshold: 0.85,
            consistency_threshold: 0.8,
            hallucination_threshold: 0.05, // Stricter for math problems
            ..ValidationConfig::default()
        };

        let validator = OutputValidator::new(validation_config);
        let problems = Self::load_gsm8k_problems().await?;

        Ok(Self {
            coordinator,
            validator,
            config,
            problems,
        })
    }

    async fn load_gsm8k_problems() -> Result<Vec<GSM8KProblem>, crate::UltraFastAiError> {
        let sample_problems = vec![
            GSM8KProblem {
                question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?".to_string(),
                answer: "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.".to_string(),
                final_answer: 72.0,
            },
            GSM8KProblem {
                question: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?".to_string(),
                answer: "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10.".to_string(),
                final_answer: 10.0,
            },
            GSM8KProblem {
                question: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?".to_string(),
                answer: "Betty has 100/2 = $50. Her grandparents gave her 15*2 = $30. Total money: 50 + 15 + 30 = $95. Betty needs 100 - 95 = $5 more.".to_string(),
                final_answer: 5.0,
            },
            GSM8KProblem {
                question: "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as much as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?".to_string(),
                answer: "Today she read 12*2 = 24 pages. Total pages read: 12 + 24 = 36 pages. Remaining pages: 120 - 36 = 84 pages. Tomorrow she should read 84/2 = 42 pages.".to_string(),
                final_answer: 42.0,
            },
            GSM8KProblem {
                question: "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more purple flowers than yellow. If there are only 25 pink flowers, what's the total number of flowers in Mark's garden?".to_string(),
                answer: "Purple flowers: 10 + (80% of 10) = 10 + 8 = 18. Total flowers: 10 + 18 + 25 = 53.".to_string(),
                final_answer: 53.0,
            },
        ];

        Ok(sample_problems)
    }

    pub async fn run_benchmark(&mut self) -> Result<BenchmarkSummary, crate::UltraFastAiError> {
        let mut results = Vec::new();
        let total_problems = self.config.max_problems.unwrap_or(self.problems.len()).min(self.problems.len());
        
        println!("🧮 Running GSM8K benchmark on {} problems...", total_problems);

        for (idx, problem) in self.problems.iter().take(total_problems).enumerate() {
            println!("  Processing problem {}/{}", idx + 1, total_problems);
            
            let start_time = Instant::now();
            let result = self.solve_math_problem(problem, idx).await;
            let inference_time = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(mut benchmark_result) => {
                    benchmark_result.inference_time_ms = inference_time;
                    results.push(benchmark_result);
                }
                Err(e) => {
                    results.push(BenchmarkResult {
                        problem_id: format!("GSM8K_{}", idx),
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

        Ok(self.calculate_summary("GSM8K", results))
    }

    async fn solve_math_problem(&mut self, problem: &GSM8KProblem, idx: usize) -> Result<BenchmarkResult, crate::UltraFastAiError> {
        // Tokenize the math problem
        let input = self.tokenize_question(&problem.question)?;
        
        // Generate solution using agentic coordination
        let generated_output = self.coordinator.execute_task(&input).await?;
        
        // Convert output to mathematical reasoning
        let generated_solution = self.detokenize_math_output(&generated_output)?;
        
        // Validate for mathematical correctness
        let validation_result = self.validator.validate_output(
            &generated_solution,
            Some(&problem.question),
            Some(&problem.answer)
        ).await?;

        // Extract and verify numerical answer
        let extracted_answer = self.extract_numerical_answer(&generated_solution);
        let success = match extracted_answer {
            Some(answer) => (answer - problem.final_answer).abs() < 0.01,
            None => false,
        };

        Ok(BenchmarkResult {
            problem_id: format!("GSM8K_{}", idx),
            success,
            generated_solution,
            execution_result: extracted_answer.map(|a| format!("Extracted answer: {}", a)),
            error_message: if success { None } else { Some("Incorrect numerical answer".to_string()) },
            inference_time_ms: 0,
            confidence_score: validation_result.confidence_score,
            validation_passed: validation_result.is_valid,
        })
    }

    fn tokenize_question(&self, question: &str) -> Result<ndarray::Array1<f32>, crate::UltraFastAiError> {
        // Enhanced tokenization for mathematical text
        let tokens: Vec<f32> = question.chars()
            .take(512)
            .map(|c| match c {
                '0'..='9' => (c as u8 - b'0') as f32 / 10.0 + 0.5, // Special encoding for numbers
                '.' => 0.95,
                '+' => 0.9,
                '-' => 0.85,
                '*' => 0.8,
                '/' => 0.75,
                '=' => 0.7,
                '$' => 0.65,
                '%' => 0.6,
                _ => (c as u8 as f32) / 255.0,
            })
            .collect();
        
        if tokens.is_empty() {
            Ok(ndarray::Array1::zeros(1))
        } else {
            Ok(ndarray::Array1::from_vec(tokens))
        }
    }

    fn detokenize_math_output(&self, output: &ndarray::Array1<f32>) -> Result<String, crate::UltraFastAiError> {
        // Convert model output to mathematical reasoning text
        let mut result = String::new();
        
        for &value in output.iter().take(256) {
            if value > 0.95 {
                result.push('.');
            } else if value > 0.9 {
                result.push('+');
            } else if value > 0.85 {
                result.push('-');
            } else if value > 0.8 {
                result.push('*');
            } else if value > 0.75 {
                result.push('/');
            } else if value > 0.7 {
                result.push('=');
            } else if value > 0.65 {
                result.push('$');
            } else if value > 0.6 {
                result.push('%');
            } else if value > 0.5 {
                // Number encoding
                let digit = ((value - 0.5) * 10.0) as u8;
                if digit < 10 {
                    result.push((b'0' + digit) as char);
                }
            } else {
                let c = (value * 255.0) as u8 as char;
                if c.is_ascii() && !c.is_control() {
                    result.push(c);
                }
            }
        }
        
        Ok(result)
    }

    fn extract_numerical_answer(&self, solution: &str) -> Option<f64> {
        // Extract the final numerical answer from the solution
        let patterns = [
            r"=\s*\$?(\d+(?:\.\d+)?)",
            r"answer:\s*\$?(\d+(?:\.\d+)?)",
            r"Answer:\s*\$?(\d+(?:\.\d+)?)",
            r"total:\s*\$?(\d+(?:\.\d+)?)",
            r"Total:\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*dollars?",
            r"(\d+(?:\.\d+)?)\s*pages?",
            r"(\d+(?:\.\d+)?)\s*flowers?",
        ];

        for pattern in &patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(captures) = re.captures(solution) {
                    if let Some(num_str) = captures.get(1) {
                        if let Ok(num) = num_str.as_str().parse::<f64>() {
                            return Some(num);
                        }
                    }
                }
            }
        }

        // Fallback: look for the last number in the solution
        let numbers: Vec<f64> = solution
            .split_whitespace()
            .filter_map(|word| {
                word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                    .parse::<f64>().ok()
            })
            .collect();

        numbers.last().copied()
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

/// Run comprehensive benchmark suite
pub async fn run_all_benchmarks() -> Result<Vec<BenchmarkSummary>, crate::UltraFastAiError> {
    let mut summaries = Vec::new();
    
    // HumanEval benchmark
    let humaneval_config = BenchmarkConfig {
        max_problems: Some(5),
        timeout_seconds: 30,
        temperature: 0.1,
        enable_validation: true,
        parallel_execution: false,
    };
    
    let mut humaneval_benchmark = crate::tests::benchmarks::humaneval_benchmark::HumanEvalBenchmark::new(humaneval_config).await?;
    let humaneval_summary = humaneval_benchmark.run_benchmark().await?;
    summaries.push(humaneval_summary);
    
    // GSM8K benchmark
    let gsm8k_config = BenchmarkConfig {
        max_problems: Some(5),
        timeout_seconds: 30,
        temperature: 0.1,
        enable_validation: true,
        parallel_execution: false,
    };
    
    let mut gsm8k_benchmark = GSM8KBenchmark::new(gsm8k_config).await?;
    let gsm8k_summary = gsm8k_benchmark.run_benchmark().await?;
    summaries.push(gsm8k_summary);
    
    Ok(summaries)
}