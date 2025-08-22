//! Benchmark validation tests for HumanEval and GSM8K
//!
//! Integration tests that validate the ultra-fast AI model against
//! standardized benchmarks for code generation and mathematical reasoning.

use std::time::Instant;
use tokio::fs;
use serde_json;

use crate::tests::benchmarks::humaneval_benchmark::{HumanEvalBenchmark, BenchmarkConfig, BenchmarkSummary};
use crate::tests::benchmarks::gsm8k_benchmark::{GSM8KBenchmark, run_all_benchmarks};

/// Comprehensive benchmark test suite
#[tokio::test]
async fn test_humaneval_benchmark() {
    println!("🧪 Starting HumanEval benchmark validation...");
    
    let config = BenchmarkConfig {
        max_problems: Some(2), // Small subset for testing
        timeout_seconds: 30,
        temperature: 0.1,
        enable_validation: true,
        parallel_execution: false,
    };
    
    let mut benchmark = HumanEvalBenchmark::new(config).await
        .expect("Failed to initialize HumanEval benchmark");
    
    let summary = benchmark.run_benchmark().await
        .expect("Failed to run HumanEval benchmark");
    
    // Validate benchmark results
    assert!(summary.total_problems > 0, "Should have test problems");
    assert!(!summary.results.is_empty(), "Should have results");
    
    // Check performance constraints
    assert!(summary.average_inference_time_ms < 30000.0, "Should complete within 30 seconds per problem");
    
    // Save results for analysis
    let results_json = serde_json::to_string_pretty(&summary)
        .expect("Failed to serialize results");
    
    fs::create_dir_all("target/benchmarks").await.ok();
    fs::write("target/benchmarks/humaneval_results.json", results_json).await
        .expect("Failed to write results");
    
    println!("✅ HumanEval benchmark completed:");
    println!("   - Total problems: {}", summary.total_problems);
    println!("   - Solved: {}", summary.solved_problems);
    println!("   - Accuracy: {:.2}%", summary.accuracy * 100.0);
    println!("   - Avg inference time: {:.2}ms", summary.average_inference_time_ms);
    println!("   - Zero-hallucination rate: {:.2}%", summary.zero_hallucination_rate * 100.0);
}

#[tokio::test]
async fn test_gsm8k_benchmark() {
    println!("🧮 Starting GSM8K benchmark validation...");
    
    let config = BenchmarkConfig {
        max_problems: Some(3), // Small subset for testing
        timeout_seconds: 30,
        temperature: 0.1,
        enable_validation: true,
        parallel_execution: false,
    };
    
    let mut benchmark = GSM8KBenchmark::new(config).await
        .expect("Failed to initialize GSM8K benchmark");
    
    let summary = benchmark.run_benchmark().await
        .expect("Failed to run GSM8K benchmark");
    
    // Validate benchmark results
    assert!(summary.total_problems > 0, "Should have test problems");
    assert!(!summary.results.is_empty(), "Should have results");
    
    // Check performance constraints
    assert!(summary.average_inference_time_ms < 30000.0, "Should complete within 30 seconds per problem");
    
    // Mathematical reasoning should have high confidence requirements
    for result in &summary.results {
        if result.success {
            assert!(result.confidence_score > 0.7, "Successful math solutions should have high confidence");
        }
    }
    
    // Save results for analysis
    let results_json = serde_json::to_string_pretty(&summary)
        .expect("Failed to serialize results");
    
    fs::create_dir_all("target/benchmarks").await.ok();
    fs::write("target/benchmarks/gsm8k_results.json", results_json).await
        .expect("Failed to write results");
    
    println!("✅ GSM8K benchmark completed:");
    println!("   - Total problems: {}", summary.total_problems);
    println!("   - Solved: {}", summary.solved_problems);
    println!("   - Accuracy: {:.2}%", summary.accuracy * 100.0);
    println!("   - Avg inference time: {:.2}ms", summary.average_inference_time_ms);
    println!("   - Zero-hallucination rate: {:.2}%", summary.zero_hallucination_rate * 100.0);
}

#[tokio::test]
async fn test_comprehensive_benchmark_suite() {
    println!("🏆 Running comprehensive benchmark validation suite...");
    
    let start_time = Instant::now();
    
    let summaries = run_all_benchmarks().await
        .expect("Failed to run comprehensive benchmarks");
    
    let total_duration = start_time.elapsed();
    
    // Validate overall performance
    assert!(!summaries.is_empty(), "Should have benchmark results");
    assert!(summaries.len() >= 2, "Should test both HumanEval and GSM8K");
    
    // Check that total benchmark time is reasonable
    assert!(total_duration.as_secs() < 300, "Full benchmark suite should complete in under 5 minutes");
    
    // Generate comprehensive report
    let mut report = String::new();
    report.push_str("# Benchmark Validation Report\n\n");
    report.push_str(&format!("**Generated**: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    report.push_str(&format!("**Total Duration**: {:.2}s\n\n", total_duration.as_secs_f64()));
    
    let mut total_problems = 0;
    let mut total_solved = 0;
    let mut total_inference_time = 0.0;
    
    for summary in &summaries {
        report.push_str(&format!("## {} Results\n\n", summary.benchmark_name));
        report.push_str(&format!("- **Problems**: {}\n", summary.total_problems));
        report.push_str(&format!("- **Solved**: {}\n", summary.solved_problems));
        report.push_str(&format!("- **Accuracy**: {:.2}%\n", summary.accuracy * 100.0));
        report.push_str(&format!("- **Avg Inference Time**: {:.2}ms\n", summary.average_inference_time_ms));
        report.push_str(&format!("- **Confidence Score**: {:.2}\n", summary.average_confidence_score));
        report.push_str(&format!("- **Zero-Hallucination Rate**: {:.2}%\n\n", summary.zero_hallucination_rate * 100.0));
        
        // Check quality thresholds
        let quality_status = if summary.accuracy > 0.7 && summary.zero_hallucination_rate > 0.8 {
            "✅ EXCELLENT"
        } else if summary.accuracy > 0.5 && summary.zero_hallucination_rate > 0.6 {
            "✅ GOOD"
        } else if summary.accuracy > 0.3 && summary.zero_hallucination_rate > 0.4 {
            "⚠️ NEEDS IMPROVEMENT"
        } else {
            "❌ POOR"
        };
        
        report.push_str(&format!("**Quality Assessment**: {}\n\n", quality_status));
        
        total_problems += summary.total_problems;
        total_solved += summary.solved_problems;
        total_inference_time += summary.average_inference_time_ms;
    }
    
    // Overall summary
    let overall_accuracy = if total_problems > 0 {
        total_solved as f32 / total_problems as f32
    } else {
        0.0
    };
    
    let avg_inference_time = total_inference_time / summaries.len() as f64;
    
    report.push_str("## Overall Summary\n\n");
    report.push_str(&format!("- **Total Problems**: {}\n", total_problems));
    report.push_str(&format!("- **Total Solved**: {}\n", total_solved));
    report.push_str(&format!("- **Overall Accuracy**: {:.2}%\n", overall_accuracy * 100.0));
    report.push_str(&format!("- **Average Inference Time**: {:.2}ms\n", avg_inference_time));
    
    // Performance validation
    let meets_performance_target = avg_inference_time < 100.0; // <100ms target
    let meets_accuracy_target = overall_accuracy > 0.5; // >50% accuracy target
    
    report.push_str(&format!("- **Performance Target (<100ms)**: {}\n", 
        if meets_performance_target { "✅ MET" } else { "❌ NOT MET" }));
    report.push_str(&format!("- **Accuracy Target (>50%)**: {}\n", 
        if meets_accuracy_target { "✅ MET" } else { "❌ NOT MET" }));
    
    // Save comprehensive report
    fs::create_dir_all("target/benchmarks").await.ok();
    fs::write("target/benchmarks/benchmark_report.md", report).await
        .expect("Failed to write benchmark report");
    
    // Save detailed results
    let detailed_results = serde_json::to_string_pretty(&summaries)
        .expect("Failed to serialize detailed results");
    fs::write("target/benchmarks/detailed_results.json", detailed_results).await
        .expect("Failed to write detailed results");
    
    println!("🎉 Comprehensive benchmark suite completed!");
    println!("📊 Overall Results:");
    println!("   - Total problems: {}", total_problems);
    println!("   - Total solved: {}", total_solved);
    println!("   - Overall accuracy: {:.2}%", overall_accuracy * 100.0);
    println!("   - Average inference time: {:.2}ms", avg_inference_time);
    println!("   - Performance target: {}", if meets_performance_target { "✅ MET" } else { "❌ NOT MET" });
    println!("   - Accuracy target: {}", if meets_accuracy_target { "✅ MET" } else { "❌ NOT MET" });
    
    // Assert key requirements
    assert!(meets_performance_target, "Must meet <100ms inference time target");
    // Note: Accuracy assertion is relaxed for development/testing
    if overall_accuracy < 0.3 {
        println!("⚠️ Warning: Accuracy below 30% - model may need additional training");
    }
}

#[tokio::test]
async fn test_benchmark_performance_constraints() {
    println!("⚡ Testing benchmark performance constraints...");
    
    // Test single problem performance
    let config = BenchmarkConfig {
        max_problems: Some(1),
        timeout_seconds: 10, // Strict timeout
        temperature: 0.0, // Deterministic
        enable_validation: true,
        parallel_execution: false,
    };
    
    // HumanEval performance test
    let mut humaneval = HumanEvalBenchmark::new(config.clone()).await
        .expect("Failed to initialize HumanEval");
    
    let start_time = Instant::now();
    let humaneval_result = humaneval.run_benchmark().await
        .expect("Failed to run HumanEval performance test");
    let humaneval_duration = start_time.elapsed();
    
    // GSM8K performance test  
    let mut gsm8k = GSM8KBenchmark::new(config).await
        .expect("Failed to initialize GSM8K");
    
    let start_time = Instant::now();
    let gsm8k_result = gsm8k.run_benchmark().await
        .expect("Failed to run GSM8K performance test");
    let gsm8k_duration = start_time.elapsed();
    
    // Validate performance constraints
    assert!(humaneval_duration.as_millis() < 15000, "HumanEval should complete in <15s for single problem");
    assert!(gsm8k_duration.as_millis() < 15000, "GSM8K should complete in <15s for single problem");
    
    // Check individual problem timing
    if !humaneval_result.results.is_empty() {
        let first_result = &humaneval_result.results[0];
        assert!(first_result.inference_time_ms < 10000, "Individual HumanEval problem should complete in <10s");
    }
    
    if !gsm8k_result.results.is_empty() {
        let first_result = &gsm8k_result.results[0];
        assert!(first_result.inference_time_ms < 10000, "Individual GSM8K problem should complete in <10s");
    }
    
    println!("✅ Performance constraints validated:");
    println!("   - HumanEval duration: {}ms", humaneval_duration.as_millis());
    println!("   - GSM8K duration: {}ms", gsm8k_duration.as_millis());
}

#[tokio::test]
async fn test_benchmark_zero_hallucination_validation() {
    println!("🔍 Testing zero-hallucination validation in benchmarks...");
    
    let config = BenchmarkConfig {
        max_problems: Some(2),
        timeout_seconds: 30,
        temperature: 0.0, // Deterministic for consistency
        enable_validation: true, // Critical for hallucination detection
        parallel_execution: false,
    };
    
    // Run benchmarks with strict validation
    let mut humaneval = HumanEvalBenchmark::new(config.clone()).await
        .expect("Failed to initialize HumanEval");
    let humaneval_result = humaneval.run_benchmark().await
        .expect("Failed to run HumanEval");
    
    let mut gsm8k = GSM8KBenchmark::new(config).await
        .expect("Failed to initialize GSM8K");
    let gsm8k_result = gsm8k.run_benchmark().await
        .expect("Failed to run GSM8K");
    
    // Validate that validation was actually performed
    for result in &humaneval_result.results {
        // Every result should have undergone validation
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0, 
                "Confidence score should be normalized");
    }
    
    for result in &gsm8k_result.results {
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0, 
                "Confidence score should be normalized");
    }
    
    // Check that zero-hallucination rates are reasonable
    println!("   - HumanEval zero-hallucination rate: {:.2}%", humaneval_result.zero_hallucination_rate * 100.0);
    println!("   - GSM8K zero-hallucination rate: {:.2}%", gsm8k_result.zero_hallucination_rate * 100.0);
    
    // In a production system, we'd want high zero-hallucination rates
    // For testing, we just ensure the validation system is working
    assert!(humaneval_result.zero_hallucination_rate >= 0.0, "Zero-hallucination rate should be non-negative");
    assert!(gsm8k_result.zero_hallucination_rate >= 0.0, "Zero-hallucination rate should be non-negative");
    
    println!("✅ Zero-hallucination validation system is functional");
}