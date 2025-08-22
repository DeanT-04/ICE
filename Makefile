# Makefile for Ultra-Fast AI Model
# Mutation Testing and Quality Assurance

.PHONY: help test coverage mutants mutants-quick mutants-ci check-quality install-tools clean security security-quick security-ci install-security-tools

# Default target
help:
	@echo "Ultra-Fast AI Model - Mutation Testing Targets"
	@echo "=============================================="
	@echo "Available targets:"
	@echo "  test            - Run all unit tests"
	@echo "  coverage        - Generate test coverage report"
	@echo "  mutants         - Run full mutation testing (≥80% score target)"
	@echo "  mutants-quick   - Run quick mutation testing on core components"
	@echo "  mutants-ci      - Run mutation testing suitable for CI/CD"
	@echo "  check-quality   - Run all quality checks (tests + coverage + mutants)"
	@echo "  security        - Run full security scanning (audit + trivy + gosec)"
	@echo "  security-quick  - Run quick security scan on critical components"
	@echo "  security-ci     - Run security scanning suitable for CI/CD"
	@echo "  benchmarks      - Run benchmark validation tests (HumanEval & GSM8K)"
	@echo "  benchmarks-quick - Run quick benchmark validation"
	@echo "  benchmarks-full - Run comprehensive benchmark validation"
	@echo "  validate-all    - Run complete validation pipeline (all checks)"
	@echo "  install-tools   - Install required testing tools"
	@echo "  install-security-tools - Install security scanning tools"
	@echo "  clean           - Clean generated test artifacts"
	@echo ""

# Configuration
MUTATION_SCORE_TARGET = 80
COVERAGE_TARGET = 85
RUST_LOG = info
CARGO_MUTANTS_VERSION = 24.11.0

# Security Configuration
SECURITY_OUTPUT_DIR = target/security
TRIVY_VERSION = latest
GOSEC_VERSION = latest

# Install required tools
install-tools:
	@echo "🔧 Installing testing tools..."
	cargo install --locked cargo-mutants@$(CARGO_MUTANTS_VERSION) || true
	cargo install --locked cargo-tarpaulin || true
	cargo install --locked cargo-audit || true
	@echo "✅ Tools installed"

# Install security scanning tools
install-security-tools:
	@echo "🔒 Installing security scanning tools..."
	cargo install --locked cargo-audit || true
	cargo install --locked cargo-deny || true
	cargo install --locked cargo-outdated || true
	@if command -v go >/dev/null 2>&1; then \
		echo "Installing Go security tools..."; \
		go install github.com/securecodewarrior/gosec/v2/cmd/gosec@$(GOSEC_VERSION) || true; \
		go install golang.org/x/vuln/cmd/govulncheck@latest || true; \
	else \
		echo "⚠️ Go not found - skipping Go security tools"; \
	fi
	@if command -v docker >/dev/null 2>&1; then \
		echo "Installing Trivy..."; \
		docker pull aquasec/trivy:$(TRIVY_VERSION) || true; \
	else \
		echo "⚠️ Docker not found - please install Trivy manually"; \
	fi
	@echo "✅ Security tools installed"

# Run basic tests
test:
	@echo "🧪 Running unit tests..."
	cargo test --all --verbose

# Generate coverage report
coverage:
	@echo "📊 Generating test coverage report..."
	mkdir -p target/coverage
	cargo tarpaulin \
		--out Html \
		--output-dir target/coverage \
		--target-dir target/tarpaulin \
		--timeout 120 \
		--verbose \
		--fail-under $(COVERAGE_TARGET)
	@echo "📋 Coverage report generated: target/coverage/tarpaulin-report.html"

# Full mutation testing
mutants: test
	@echo "🧬 Running comprehensive mutation testing..."
	mkdir -p target/mutants/reports
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File scripts/run_mutation_tests.ps1; \
	elif command -v bash >/dev/null 2>&1; then \
		bash scripts/run_mutation_tests.sh; \
	else \
		echo "❌ Neither PowerShell nor Bash found"; \
		exit 1; \
	fi
	@echo "📋 Mutation testing report: target/mutants/reports/mutation_report.md"

# Quick mutation testing (core components only)
mutants-quick: test
	@echo "🧬 Running quick mutation testing on core components..."
	mkdir -p target/mutants
	cargo mutants \
		--config mutants.toml \
		--examine src/model/core.rs \
		--examine src/model/fusion.rs \
		--timeout 180 \
		--jobs 4 \
		--output target/mutants/quick_results.json

# CI-friendly mutation testing
mutants-ci: test
	@echo "🧬 Running CI mutation testing..."
	mkdir -p target/mutants
	cargo mutants \
		--config mutants.toml \
		--timeout 300 \
		--jobs 2 \
		--output target/mutants/ci_results.json \
		--in-place \
		--no-times

# Comprehensive quality check
check-quality: test coverage mutants
	@echo "✅ All quality checks completed!"
	@echo "📊 Results:"
	@echo "   - Unit Tests: ✅"
	@echo "   - Coverage: target/coverage/tarpaulin-report.html"
	@echo "   - Mutation Testing: target/mutants/reports/mutation_report.md"

# Full security scanning
security: install-security-tools
	@echo "🔒 Running comprehensive security scanning..."
	mkdir -p $(SECURITY_OUTPUT_DIR)
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File scripts/run_security_scan.ps1 -Mode full -OutputDir $(SECURITY_OUTPUT_DIR); \
	else \
		echo "❌ PowerShell not found - security scanning requires PowerShell"; \
		exit 1; \
	fi
	@echo "📋 Security report: $(SECURITY_OUTPUT_DIR)/security-report.html"

# Quick security scan (critical components only)
security-quick:
	@echo "🔒 Running quick security scan..."
	mkdir -p $(SECURITY_OUTPUT_DIR)
	@echo "Running cargo audit..."
	cargo audit --json --output $(SECURITY_OUTPUT_DIR)/quick-audit.json || true
	@echo "Running basic security checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check --format json > $(SECURITY_OUTPUT_DIR)/quick-deny.json || true; \
	fi
	@echo "✅ Quick security scan completed"

# CI-friendly security scanning
security-ci:
	@echo "🔒 Running CI security scanning..."
	mkdir -p $(SECURITY_OUTPUT_DIR)
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File scripts/run_security_scan.ps1 -Mode ci -OutputDir $(SECURITY_OUTPUT_DIR) -FailFast; \
	else \
		echo "Running basic security checks..."; \
		cargo audit --json --output $(SECURITY_OUTPUT_DIR)/ci-audit.json; \
	fi

# Comprehensive quality and security check
check-all: test coverage mutants security
	@echo "✅ All quality and security checks completed!"
	@echo "📊 Results:"
	@echo "   - Unit Tests: ✅"
	@echo "   - Coverage: target/coverage/tarpaulin-report.html"
	@echo "   - Mutation Testing: target/mutants/reports/mutation_report.md"
	@echo "   - Security Scanning: $(SECURITY_OUTPUT_DIR)/security-report.html"

# Benchmark validation testing
benchmarks:
	@echo "🏆 Running benchmark validation tests..."
	mkdir -p target/benchmarks
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File scripts/run_benchmark_tests.ps1 -GenerateReport -MaxProblems 3; \
	else \
		echo "Running basic benchmark tests..."; \
		cargo test --test benchmarks --release; \
	fi
	@echo "📋 Benchmark results: target/benchmarks/benchmark_report.md"

# Quick benchmark validation (minimal problems)
benchmarks-quick:
	@echo "🏆 Running quick benchmark validation..."
	mkdir -p target/benchmarks
	cargo test --test benchmarks --release -- --test-threads=1

# Full benchmark suite with comprehensive reporting
benchmarks-full:
	@echo "🏆 Running comprehensive benchmark validation..."
	mkdir -p target/benchmarks
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File scripts/run_benchmark_tests.ps1 -GenerateReport -MaxProblems 10 -Verbose; \
	else \
		echo "Running comprehensive benchmark tests..."; \
		cargo test --test benchmarks --release -- --nocapture; \
	fi

# Complete validation pipeline (all checks + benchmarks)
validate-all: test coverage mutants security benchmarks
	@echo "✅ Complete validation pipeline completed!"
	@echo "📋 Comprehensive Results:"
	@echo "   - Unit Tests: ✅"
	@echo "   - Coverage: target/coverage/tarpaulin-report.html"
	@echo "   - Mutation Testing: target/mutants/reports/mutation_report.md"
	@echo "   - Security Scanning: $(SECURITY_OUTPUT_DIR)/security-report.html"
	@echo "   - Benchmark Validation: target/benchmarks/benchmark_report.md"

# Advanced mutation testing with specific configurations
mutants-advanced:
	@echo "🧬 Running advanced mutation testing with all strategies..."
	cargo mutants \
		--config mutants.toml \
		--timeout 600 \
		--jobs 4 \
		--output target/mutants/advanced_results.json \
		--html-dir target/mutants/advanced_html \
		--examine src/model/ \
		--examine src/training/ \
		--examine src/utils/

# Mutation testing for specific component
mutants-component:
	@echo "🧬 Running mutation testing for specific component..."
	@read -p "Enter component path (e.g., src/model/core.rs): " component; \
	cargo mutants \
		--config mutants.toml \
		--examine $$component \
		--timeout 300 \
		--jobs 4 \
		--output target/mutants/component_results.json

# Performance-aware mutation testing (excludes performance-critical paths)
mutants-perf-safe:
	@echo "🧬 Running performance-safe mutation testing..."
	cargo mutants \
		--config mutants.toml \
		--timeout 300 \
		--jobs 4 \
		--output target/mutants/perf_safe_results.json \
		--skip-calls std::thread::sleep \
		--skip-calls tokio::time::sleep

# Generate mutation testing trend report
mutants-trend:
	@echo "📈 Generating mutation testing trend report..."
	mkdir -p target/mutants/trends
	@echo "Mutation Score Trend Report" > target/mutants/trends/trend_$(shell date +%Y%m%d).md
	@echo "=========================" >> target/mutants/trends/trend_$(shell date +%Y%m%d).md
	@echo "Date: $(shell date)" >> target/mutants/trends/trend_$(shell date +%Y%m%d).md
	@if [ -f target/mutants/complete_results.json ]; then \
		echo "Score: $(shell grep -o '"mutation_score":[0-9.]*' target/mutants/complete_results.json | cut -d: -f2)%" >> target/mutants/trends/trend_$(shell date +%Y%m%d).md; \
	fi

# Validate mutation testing setup
validate-setup:
	@echo "🔍 Validating mutation testing setup..."
	@echo "Checking configuration file..."
	@if [ -f mutants.toml ]; then \
		echo "✅ mutants.toml found"; \
	else \
		echo "❌ mutants.toml not found"; \
		exit 1; \
	fi
	@echo "Checking cargo-mutants installation..."
	@if command -v cargo-mutants >/dev/null 2>&1; then \
		echo "✅ cargo-mutants installed: $(shell cargo-mutants --version)"; \
	else \
		echo "❌ cargo-mutants not installed"; \
		exit 1; \
	fi
	@echo "Checking project compilation..."
	@if cargo check --all-targets >/dev/null 2>&1; then \
		echo "✅ Project compiles successfully"; \
	else \
		echo "❌ Project compilation failed"; \
		exit 1; \
	fi

# Clean generated artifacts
clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf target/mutants/
	rm -rf target/coverage/
	rm -rf target/tarpaulin/
	rm -rf $(SECURITY_OUTPUT_DIR)/
	rm -rf .cache/security/
	rm -rf .cache/trivy/
	cargo clean
	@echo "✅ Cleanup completed"

# Integration with GitHub Actions
github-mutants:
	@echo "🧬 Running mutation testing for GitHub Actions..."
	cargo mutants \
		--config mutants.toml \
		--timeout 300 \
		--jobs 2 \
		--output target/mutants/github_results.json \
		--baseline skip \
		--in-place

# Docker-based mutation testing (for consistent environment)
docker-mutants:
	@echo "🐳 Running mutation testing in Docker..."
	docker build -t ultra-fast-ai-mutants -f Dockerfile.mutants .
	docker run --rm -v $(PWD)/target:/app/target ultra-fast-ai-mutants

# Summary of all mutation testing results
summary:
	@echo "📋 Mutation Testing Summary"
	@echo "=========================="
	@if [ -f target/mutants/complete_results.json ]; then \
		echo "Latest Full Run:"; \
		grep -o '"mutation_score":[0-9.]*' target/mutants/complete_results.json | cut -d: -f2 | head -1 | xargs printf "  Score: %.2f%%\n"; \
	fi
	@if [ -f target/mutants/quick_results.json ]; then \
		echo "Latest Quick Run:"; \
		grep -o '"mutation_score":[0-9.]*' target/mutants/quick_results.json | cut -d: -f2 | head -1 | xargs printf "  Score: %.2f%%\n"; \
	fi
	@echo "Target: $(MUTATION_SCORE_TARGET)%"
	@echo ""
	@echo "Reports available:"
	@ls -la target/mutants/reports/ 2>/dev/null || echo "  No reports generated yet"

# Watch mode for continuous mutation testing during development
watch-mutants:
	@echo "👀 Starting mutation testing watch mode..."
	@echo "This will run quick mutation tests when files change..."
	while true; do \
		inotifywait -r -e modify src/; \
		echo "🔄 Files changed, running quick mutation tests..."; \
		$(MAKE) mutants-quick; \
		sleep 5; \
	done