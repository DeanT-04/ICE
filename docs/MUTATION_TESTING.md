# Mutation Testing Guide for Ultra-Fast AI Model

## Overview

This project uses **cargo-mutants** to achieve a mutation testing score of ≥80%, ensuring comprehensive test coverage and quality. Mutation testing works by introducing small changes (mutations) to the source code and verifying that the test suite can detect these changes.

## Quick Start

### Prerequisites

```bash
# Install required tools
cargo install --locked cargo-mutants@24.11.0
cargo install --locked cargo-tarpaulin
```

### Basic Usage

```bash
# Run quick mutation testing on core components
make mutants-quick

# Run comprehensive mutation testing
make mutants

# Run with custom score target
MUTATION_SCORE_TARGET=85 make mutants
```

## Configuration

### Main Configuration File: `mutants.toml`

The project uses a comprehensive mutation testing configuration that:

- **Target Score**: ≥80% overall, with component-specific targets
- **Timeout**: 300 seconds per mutant 
- **Parallel Jobs**: 4 for faster execution
- **Skip Patterns**: Excludes logging, panics, and debug code
- **Component Targets**:
  - Core Neural Networks: 90%
  - Training Components: 80%
  - Utility Components: 75%

### Component-Specific Configuration

```toml
[[mutants.files]]
path = "src/model/core.rs"
minimum_score = 90.0  # Higher requirement for critical components

[[mutants.files]]
path = "src/training/trainer.rs"
minimum_score = 80.0

[[mutants.files]]
path = "src/utils/perf.rs" 
minimum_score = 75.0
```

## Usage Scenarios

### 1. Development Workflow

```bash
# Quick check during development
make mutants-quick

# Watch mode (requires inotify-tools)
make watch-mutants
```

### 2. Pre-commit Testing

```bash
# Validate mutation testing setup
make validate-setup

# Run component-specific testing
make mutants-component
# Enter component path when prompted
```

### 3. CI/CD Integration

The project includes GitHub Actions workflows for:

- **Per-component testing** on each PR
- **Comprehensive testing** weekly or on-demand
- **Regression checking** for changed files only

### 4. Performance-Safe Testing

```bash
# Skip performance-critical mutations
make mutants-perf-safe
```

## Scripts and Tools

### PowerShell Script (Windows)

```powershell
# Run with custom parameters
.\scripts\run_mutation_tests.ps1 -MutationScoreTarget 85 -ParallelJobs 2
```

### Bash Script (Linux/macOS)

```bash
# Run with environment variables
MUTATION_SCORE_TARGET=85 PARALLEL_JOBS=2 ./scripts/run_mutation_tests.sh
```

### Analysis Tools

```bash
# Analyze results and generate insights
python scripts/analyze_mutations.py --generate-viz --analyze-survivors

# Generate trend reports
make mutants-trend
```

## Understanding Results

### Mutation Score Interpretation

- **90-100%**: Excellent test coverage
- **80-89%**: Good coverage, meets project standards
- **70-79%**: Acceptable but needs improvement
- **<70%**: Poor coverage, significant gaps

### Common Surviving Mutants

1. **Arithmetic Operators**: `+` → `-`, `*` → `/`
   - **Solution**: Add tests with specific expected values

2. **Boundary Conditions**: `<` → `<=`, `>` → `>=`
   - **Solution**: Test edge cases and boundary values

3. **Logical Operators**: `&&` → `||`, `!` → ` `
   - **Solution**: Test all boolean conditions

4. **Constants**: `0` → `1`, `true` → `false`
   - **Solution**: Verify specific constant usage

### Example Test Improvements

#### Before (Surviving Mutant)
```rust
// Mutant: threshold > 0.5 → threshold >= 0.5 (survives)
if neuron.potential > threshold {
    neuron.fire();
}

#[test]
fn test_neuron_firing() {
    let mut neuron = Neuron::new();
    neuron.potential = 0.6;
    neuron.update(0.5);
    assert!(neuron.fired); // Too general
}
```

#### After (Kills Mutant)
```rust
#[test]
fn test_neuron_firing_boundary() {
    let mut neuron = Neuron::new();
    
    // Test exact boundary
    neuron.potential = 0.5;
    neuron.update(0.5);
    assert!(!neuron.fired); // Should not fire at threshold
    
    // Test just above boundary
    neuron.potential = 0.5001;
    neuron.update(0.5);
    assert!(neuron.fired); // Should fire above threshold
}
```

## File Structure

```
ultra-fast-ai/
├── mutants.toml                    # Main configuration
├── Makefile                        # Build targets
├── scripts/
│   ├── run_mutation_tests.sh       # Bash execution script
│   ├── run_mutation_tests.ps1      # PowerShell execution script
│   └── analyze_mutations.py        # Results analysis
├── .github/workflows/
│   └── mutation-testing.yml        # CI/CD automation
└── target/mutants/
    ├── reports/                     # Generated reports
    ├── html/                        # HTML visualization
    └── *_results.json              # Raw results
```

## Troubleshooting

### Common Issues

#### 1. Timeouts

```bash
# Increase timeout for complex operations
cargo mutants --timeout 600
```

#### 2. Memory Issues

```bash
# Reduce parallel jobs
cargo mutants --jobs 1
```

#### 3. False Positives

Add to `mutants.toml`:
```toml
skip_calls = [
    "your_function_name",
    "debug_print",
]
```

#### 4. Compilation Errors

```bash
# Verify project compiles first
cargo check --all-targets
cargo test --all
```

### Performance Optimization

1. **Use `--in-place`** for faster execution
2. **Skip non-critical files** with patterns
3. **Use `--baseline skip`** to avoid baseline overhead
4. **Run component-specific tests** during development

## Best Practices

### 1. Test Strategy

- **Unit Tests**: Focus on individual functions
- **Integration Tests**: Test component interactions  
- **Property Tests**: Use `proptest` for algorithms
- **Edge Cases**: Test boundary conditions explicitly

### 2. Mutation Testing Workflow

1. **Start Small**: Test core components first
2. **Iterate**: Fix one component at a time
3. **Analyze**: Review surviving mutants carefully
4. **Improve**: Add targeted tests for gaps

### 3. CI Integration

- **PR Testing**: Run on changed files only
- **Scheduled**: Full testing weekly
- **Quality Gates**: Block merges below threshold

### 4. Monitoring

- **Track Trends**: Monitor score changes over time
- **Component Health**: Focus on critical components
- **Regression Detection**: Catch score decreases early

## Advanced Features

### Custom Mutation Operators

```toml
[mutants.strategies]
arithmetic = true      # +, -, *, /, %
logical = true        # &&, ||, !
relational = true     # <, >, <=, >=, ==, !=
constants = true      # 0→1, true→false
boundaries = true     # <→<=, >→>=
```

### AI Model Specific Configuration

```toml
[mutants.ai_model]
preserve_model_structure = true
preserve_activation_functions = true
preserve_gradient_computation = true
```

### Parallel Testing Strategies

```toml
# Component-based parallelization
[[mutants.test_variants]]
name = "neural_networks"
test_args = ["test", "--test", "model_tests"]

[[mutants.test_variants]]
name = "training" 
test_args = ["test", "--test", "training_tests"]
```

## Metrics and Reporting

### Key Metrics

- **Mutation Score**: Percentage of detected mutations
- **Test Coverage**: Lines/branches covered
- **Execution Time**: Performance impact
- **Surviving Mutants**: Undetected changes

### Report Formats

- **HTML**: Interactive browser reports
- **JSON**: Machine-readable results
- **Markdown**: Human-readable summaries
- **Visualization**: Charts and graphs

## Contributing

When contributing to the project:

1. **Run mutation tests** on changed components
2. **Maintain score** above component thresholds
3. **Add tests** for new functionality
4. **Review survivors** in your changes

### Commit Guidelines

```bash
# Trigger full mutation testing
git commit -m "feat: new neural layer [full-mutation]"

# Normal mutation testing
git commit -m "fix: improve gradient calculation"
```

---

## Support

For issues with mutation testing:

1. Check the [troubleshooting section](#troubleshooting)
2. Review generated reports in `target/mutants/reports/`
3. Analyze surviving mutants with `scripts/analyze_mutations.py`
4. Consult the [cargo-mutants documentation](https://mutants.rs/)

The mutation testing setup ensures the ultra-fast AI model maintains high code quality and comprehensive test coverage throughout development.