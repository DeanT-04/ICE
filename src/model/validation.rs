//! Output validation for zero-hallucination enforcement
//!
//! Implements validation layers to detect and prevent hallucinations,
//! ensuring reliable and factual model outputs.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use regex::Regex;
use crate::Result;

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enable_validation: bool,
    pub confidence_threshold: f32,
    pub consistency_threshold: f32,
    pub fact_check_threshold: f32,
    pub max_validation_time_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            confidence_threshold: 0.8,
            consistency_threshold: 0.9,
            fact_check_threshold: 0.85,
            max_validation_time_ms: 500,
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence_score: f32,
    pub consistency_score: f32,
    pub fact_check_score: f32,
    pub hallucination_risk: f32,
    pub validation_time_ms: u64,
    pub issues: Vec<ValidationIssue>,
    pub suggestions: Vec<String>,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub severity: Severity,
    pub description: String,
    pub suggested_fix: Option<String>,
}

/// Issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    FactualInconsistency,
    LogicalContradiction,
    HallucinationDetected,
    ConfidenceThresholdFailure,
}

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Output validator
pub struct OutputValidator {
    config: ValidationConfig,
    known_facts: HashMap<String, f32>,
    suspicious_patterns: Vec<Regex>,
}

impl OutputValidator {
    pub fn new(config: ValidationConfig) -> Self {
        let mut validator = Self {
            config,
            known_facts: HashMap::new(),
            suspicious_patterns: Vec::new(),
        };
        validator.initialize();
        validator
    }

    fn initialize(&mut self) {
        // Load known facts
        let facts = [
            ("Python was created by Guido van Rossum", 0.95),
            ("The capital of France is Paris", 0.99),
            ("Water boils at 100°C at sea level", 0.99),
            ("Shakespeare wrote Hamlet", 0.95),
        ];
        
        for (fact, confidence) in &facts {
            self.known_facts.insert(fact.to_string(), *confidence);
        }

        // Initialize suspicious patterns
        let patterns = [
            r"\baccording to recent studies\b",
            r"\bexperts say\b",
            r"\bit is widely known that\b",
        ];
        
        for pattern in &patterns {
            if let Ok(regex) = Regex::new(pattern) {
                self.suspicious_patterns.push(regex);
            }
        }
    }

    /// Validate model output comprehensively
    pub async fn validate_output(
        &self,
        output: &str,
        context: Option<&str>,
        expected: Option<&str>,
    ) -> Result<ValidationResult> {
        if !self.config.enable_validation {
            return Ok(ValidationResult {
                is_valid: true,
                confidence_score: 1.0,
                consistency_score: 1.0,
                fact_check_score: 1.0,
                hallucination_risk: 0.0,
                validation_time_ms: 0,
                issues: Vec::new(),
                suggestions: Vec::new(),
            });
        }

        let start_time = std::time::Instant::now();
        let mut issues = Vec::new();

        // Confidence scoring
        let confidence_score = self.calculate_confidence(output, context);

        // Consistency checking
        let consistency_score = self.check_consistency(output, context, expected);

        // Fact checking
        let (fact_check_score, fact_issues) = self.check_facts(output);
        issues.extend(fact_issues);

        // Hallucination detection
        let (hallucination_risk, hallucination_issues) = self.detect_hallucinations(output, context);
        issues.extend(hallucination_issues);

        // Generate suggestions
        let suggestions = self.generate_suggestions(&issues);

        // Determine overall validity
        let is_valid = confidence_score >= self.config.confidence_threshold
            && consistency_score >= self.config.consistency_threshold
            && fact_check_score >= self.config.fact_check_threshold
            && hallucination_risk < 0.3
            && !issues.iter().any(|issue| matches!(issue.severity, Severity::Critical));

        let validation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ValidationResult {
            is_valid,
            confidence_score,
            consistency_score,
            fact_check_score,
            hallucination_risk,
            validation_time_ms,
            issues,
            suggestions,
        })
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, output: &str, context: Option<&str>) -> f32 {
        let mut confidence = 1.0;

        // Length-based confidence
        let length = output.len();
        confidence *= match length {
            0..=10 => 0.3,
            11..=50 => 0.8,
            51..=500 => 1.0,
            501..=1000 => 0.9,
            _ => 0.6,
        };

        // Uncertainty marker detection
        let uncertainty_markers = ["might", "maybe", "possibly", "perhaps", "could be"];
        let marker_count = uncertainty_markers.iter()
            .map(|marker| output.matches(marker).count())
            .sum::<usize>();

        confidence *= match marker_count {
            0 => 0.9,
            1..=2 => 1.0,
            3..=5 => 0.8,
            _ => 0.5,
        };

        // Repetition penalty
        let words: Vec<&str> = output.split_whitespace().collect();
        let unique_words: HashSet<&str> = words.iter().copied().collect();
        
        if !words.is_empty() {
            let repetition_ratio = unique_words.len() as f32 / words.len() as f32;
            confidence *= if repetition_ratio < 0.5 { 0.5 } 
                         else if repetition_ratio < 0.7 { 0.8 } 
                         else { 1.0 };
        }

        // Context relevance
        if let Some(ctx) = context {
            let output_words: HashSet<&str> = output.split_whitespace().collect();
            let context_words: HashSet<&str> = ctx.split_whitespace().collect();
            
            if !output_words.is_empty() {
                let relevant_words = output_words.intersection(&context_words).count();
                let relevance_ratio = relevant_words as f32 / output_words.len() as f32;
                confidence *= (relevance_ratio * 2.0).min(1.0);
            }
        }

        confidence.max(0.0).min(1.0)
    }

    /// Check consistency
    fn check_consistency(&self, output: &str, context: Option<&str>, _expected: Option<&str>) -> f32 {
        let mut consistency = 1.0;

        // Internal consistency
        let sentences: Vec<&str> = output.split('.').collect();
        let mut contradictions = 0;
        let mut total_checks = 0;

        for i in 0..sentences.len() {
            for j in (i + 1)..sentences.len() {
                total_checks += 1;
                if self.are_contradictory(sentences[i], sentences[j]) {
                    contradictions += 1;
                }
            }
        }

        if total_checks > 0 {
            consistency *= 1.0 - (contradictions as f32 / total_checks as f32);
        }

        // Context consistency
        if let Some(ctx) = context {
            let output_words: HashSet<&str> = output.split_whitespace().collect();
            let context_words: HashSet<&str> = ctx.split_whitespace().collect();
            
            let intersection = output_words.intersection(&context_words).count();
            let union = output_words.union(&context_words).count();
            
            if union > 0 {
                let similarity = intersection as f32 / union as f32;
                consistency *= (similarity * 2.0).min(1.0);
            }
        }

        consistency.max(0.0)
    }

    /// Check for contradictions
    fn are_contradictory(&self, sentence1: &str, sentence2: &str) -> bool {
        let negation_patterns = ["not", "no", "never", "none"];
        let s1_words: Vec<&str> = sentence1.split_whitespace().collect();
        let s2_words: Vec<&str> = sentence2.split_whitespace().collect();

        for pattern in &negation_patterns {
            if (s1_words.contains(pattern) && !s2_words.contains(pattern))
                || (!s1_words.contains(pattern) && s2_words.contains(pattern))
            {
                let common_words = s1_words.iter()
                    .filter(|&&word| s2_words.contains(&word) && word.len() > 3)
                    .count();
                
                if common_words >= 2 {
                    return true;
                }
            }
        }
        false
    }

    /// Check facts against known knowledge
    fn check_facts(&self, text: &str) -> (f32, Vec<ValidationIssue>) {
        let mut issues = Vec::new();
        let mut fact_score = 1.0;

        for (known_fact, confidence) in &self.known_facts {
            if text.contains(known_fact) {
                fact_score *= confidence;
            }
        }

        // Look for unverified factual claims
        let factual_patterns = [r"\b\d{4}\b", r"\b\d+\.\d+%\b"];
        for pattern_str in &factual_patterns {
            if let Ok(pattern) = Regex::new(pattern_str) {
                if pattern.is_match(text) {
                    // Check if we can verify this fact
                    let verified = self.known_facts.iter()
                        .any(|(fact, _)| text.contains(fact));
                    
                    if !verified {
                        issues.push(ValidationIssue {
                            issue_type: IssueType::FactualInconsistency,
                            severity: Severity::Medium,
                            description: "Unverified factual claim detected".to_string(),
                            suggested_fix: Some("Verify this claim against reliable sources".to_string()),
                        });
                        fact_score *= 0.8;
                    }
                }
            }
        }

        (fact_score, issues)
    }

    /// Detect potential hallucinations
    fn detect_hallucinations(&self, output: &str, context: Option<&str>) -> (f32, Vec<ValidationIssue>) {
        let mut issues = Vec::new();
        let mut risk = 0.0;

        // Check for suspicious patterns
        for pattern in &self.suspicious_patterns {
            if pattern.is_match(output) {
                issues.push(ValidationIssue {
                    issue_type: IssueType::HallucinationDetected,
                    severity: Severity::Medium,
                    description: "Suspicious pattern that may indicate hallucination".to_string(),
                    suggested_fix: Some("Verify with specific sources".to_string()),
                });
                risk += 0.3;
            }
        }

        // Check for context divergence
        if let Some(ctx) = context {
            let output_words: HashSet<&str> = output.split_whitespace().collect();
            let context_words: HashSet<&str> = ctx.split_whitespace().collect();
            
            if !output_words.is_empty() {
                let output_unique = output_words.difference(&context_words).count();
                let divergence = output_unique as f32 / output_words.len() as f32;
                
                if divergence > 0.7 {
                    issues.push(ValidationIssue {
                        issue_type: IssueType::HallucinationDetected,
                        severity: Severity::High,
                        description: "Output diverges significantly from context".to_string(),
                        suggested_fix: Some("Stay relevant to the given context".to_string()),
                    });
                    risk += divergence * 0.5;
                }
            }
        }

        (risk.min(1.0), issues)
    }

    /// Generate improvement suggestions
    fn generate_suggestions(&self, issues: &[ValidationIssue]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for issue in issues {
            match issue.issue_type {
                IssueType::FactualInconsistency => {
                    suggestions.push("Verify factual claims against reliable sources".to_string());
                },
                IssueType::LogicalContradiction => {
                    suggestions.push("Review logical flow and resolve contradictions".to_string());
                },
                IssueType::HallucinationDetected => {
                    suggestions.push("Remove or verify potentially fabricated information".to_string());
                },
                IssueType::ConfidenceThresholdFailure => {
                    suggestions.push("Consider rephrasing with uncertainty markers".to_string());
                },
            }
        }
        
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    /// Quick validation for performance-critical paths
    pub fn quick_validate(&self, output: &str) -> bool {
        if !self.config.enable_validation {
            return true;
        }

        // Basic checks only
        let confidence = self.calculate_confidence(output, None);
        let has_suspicious_patterns = self.suspicious_patterns.iter()
            .any(|pattern| pattern.is_match(output));

        confidence >= self.config.confidence_threshold && !has_suspicious_patterns
    }

    /// Get validation statistics
    pub fn get_validation_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        stats.insert("known_facts_count".to_string(), self.known_facts.len() as f32);
        stats.insert("suspicious_patterns_count".to_string(), self.suspicious_patterns.len() as f32);
        stats.insert("confidence_threshold".to_string(), self.config.confidence_threshold);
        stats.insert("consistency_threshold".to_string(), self.config.consistency_threshold);
        stats.insert("fact_check_threshold".to_string(), self.config.fact_check_threshold);
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_output_validator_creation() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        assert!(validator.config.enable_validation);
        assert_eq!(validator.config.confidence_threshold, 0.8);
    }

    #[tokio::test]
    async fn test_validation_with_good_output() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let output = "Python is a programming language created by Guido van Rossum.";
        let result = validator.validate_output(output, None, None).await.unwrap();
        
        assert!(result.confidence_score > 0.5);
        assert!(result.fact_check_score > 0.5);
    }

    #[tokio::test]
    async fn test_hallucination_detection() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let suspicious_output = "According to recent studies, aliens have visited Earth.";
        let result = validator.validate_output(suspicious_output, None, None).await.unwrap();
        
        assert!(result.hallucination_risk > 0.0);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_quick_validate() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let good_output = "This is a reasonable statement.";
        let bad_output = "According to experts, this is suspicious.";
        
        assert!(validator.quick_validate(good_output));
        assert!(!validator.quick_validate(bad_output));
    }

    #[test]
    fn test_consistency_checking() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let consistent = "The sky is blue. Blue is a color.";
        let inconsistent = "The sky is blue. The sky is not blue.";
        
        let consistent_score = validator.check_consistency(consistent, None, None);
        let inconsistent_score = validator.check_consistency(inconsistent, None, None);
        
        assert!(consistent_score > inconsistent_score);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = ValidationConfig::default();
        let validator = OutputValidator::new(config);
        
        let confident = "This is a clear and concise statement.";
        let uncertain = "Maybe this might possibly be true, perhaps.";
        
        let confident_score = validator.calculate_confidence(confident, None);
        let uncertain_score = validator.calculate_confidence(uncertain, None);
        
        assert!(confident_score > uncertain_score);
    }
}