#!/bin/bash
# Mutation Testing Script for Ultra-Fast AI Model
# Target: ≥80% mutation score

set -e

echo "🧬 Starting Mutation Testing for Ultra-Fast AI Model"
echo "Target: ≥80% mutation score"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MUTATION_SCORE_TARGET=80
COVERAGE_TARGET=85
MAX_TIMEOUT=300
PARALLEL_JOBS=4

# Create output directories
mkdir -p target/mutants
mkdir -p target/mutants/html
mkdir -p target/mutants/reports

echo -e "${BLUE}📋 Pre-flight checks...${NC}"

# Check if cargo-mutants is installed
if ! command -v cargo-mutants &> /dev/null; then
    echo -e "${YELLOW}⚠️  cargo-mutants not found. Installing...${NC}"
    cargo install --locked cargo-mutants
fi

# Verify the project compiles and tests pass
echo -e "${BLUE}🔧 Verifying project compilation...${NC}"
if ! cargo check --all-targets; then
    echo -e "${RED}❌ Project compilation failed. Please fix compilation errors first.${NC}"
    exit 1
fi

echo -e "${BLUE}🧪 Running baseline tests...${NC}"
if ! cargo test --all; then
    echo -e "${RED}❌ Baseline tests failed. Please fix failing tests first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"

# Run mutation testing with different phases
echo -e "${BLUE}🧬 Phase 1: Core Model Components${NC}"
cargo mutants --config mutants.toml \
    --examine src/model/core.rs \
    --examine src/model/fusion.rs \
    --timeout $MAX_TIMEOUT \
    --jobs $PARALLEL_JOBS \
    --output target/mutants/core_results.json || true

echo -e "${BLUE}🧬 Phase 2: Training Components${NC}"
cargo mutants --config mutants.toml \
    --examine src/training/ \
    --timeout $MAX_TIMEOUT \
    --jobs $PARALLEL_JOBS \
    --output target/mutants/training_results.json || true

echo -e "${BLUE}🧬 Phase 3: Utility Components${NC}"
cargo mutants --config mutants.toml \
    --examine src/utils/ \
    --timeout $MAX_TIMEOUT \
    --jobs $PARALLEL_JOBS \
    --output target/mutants/utils_results.json || true

echo -e "${BLUE}🧬 Phase 4: Complete Mutation Testing${NC}"
cargo mutants --config mutants.toml \
    --timeout $MAX_TIMEOUT \
    --jobs $PARALLEL_JOBS \
    --output target/mutants/complete_results.json \
    --html-dir target/mutants/html || true

# Analyze results
echo -e "${BLUE}📊 Analyzing mutation testing results...${NC}"

# Function to extract mutation score from JSON output
extract_score() {
    local file=$1
    if [ -f "$file" ]; then
        # Simple extraction - in real scenario would use jq or proper JSON parsing
        grep -o '"mutation_score":[0-9.]*' "$file" | cut -d: -f2 | head -1
    else
        echo "0"
    fi
}

# Extract scores
CORE_SCORE=$(extract_score "target/mutants/core_results.json")
TRAINING_SCORE=$(extract_score "target/mutants/training_results.json")
UTILS_SCORE=$(extract_score "target/mutants/utils_results.json")
OVERALL_SCORE=$(extract_score "target/mutants/complete_results.json")

# Generate comprehensive report
echo -e "${BLUE}📝 Generating comprehensive mutation testing report...${NC}"

cat > target/mutants/reports/mutation_report.md << EOF
# Mutation Testing Report - Ultra-Fast AI Model

## Summary
- **Overall Mutation Score**: ${OVERALL_SCORE}%
- **Target**: ≥${MUTATION_SCORE_TARGET}%
- **Status**: $([ "${OVERALL_SCORE%.*}" -ge "$MUTATION_SCORE_TARGET" ] && echo "✅ PASSED" || echo "❌ NEEDS IMPROVEMENT")

## Component Scores
- **Core Model Components**: ${CORE_SCORE}%
- **Training Components**: ${TRAINING_SCORE}%
- **Utility Components**: ${UTILS_SCORE}%

## Testing Configuration
- **Timeout**: ${MAX_TIMEOUT}s per mutant
- **Parallel Jobs**: ${PARALLEL_JOBS}
- **Test Command**: \`cargo test --all\`

## Files Analyzed
- \`src/model/core.rs\` - Neural network implementations
- \`src/model/fusion.rs\` - Model fusion logic  
- \`src/model/agentic.rs\` - Agentic coordination
- \`src/training/trainer.rs\` - Training loop
- \`src/training/genetic.rs\` - Genetic algorithms
- \`src/utils/perf.rs\` - Performance monitoring
- \`src/utils/energy.rs\` - Energy monitoring

## Mutation Types Tested
- ✅ Arithmetic operators (+, -, *, /, %)
- ✅ Logical operators (&&, ||, !)
- ✅ Relational operators (<, >, <=, >=, ==, !=)
- ✅ Constant values and boundaries
- ✅ Function call mutations
- ✅ Assignment operators

## Quality Gates
- **Minimum Mutation Score**: ${MUTATION_SCORE_TARGET}%
- **Core Components Target**: 90%
- **Training Components Target**: 80%
- **Utility Components Target**: 75%

## Recommendations
$([ "${OVERALL_SCORE%.*}" -ge "$MUTATION_SCORE_TARGET" ] && echo "✅ Mutation testing passed all quality gates." || echo "❌ Consider adding more comprehensive tests to improve mutation score.")

## Next Steps
1. Review surviving mutants in HTML report
2. Add tests for uncovered mutation cases
3. Improve test assertions to catch subtle bugs
4. Consider property-based testing for complex algorithms

## Generated Files
- **HTML Report**: \`target/mutants/html/index.html\`
- **JSON Results**: \`target/mutants/complete_results.json\`
- **Component Reports**: \`target/mutants/*_results.json\`

---
Generated: $(date)
EOF

# Display results
echo ""
echo -e "${BLUE}📊 MUTATION TESTING RESULTS${NC}"
echo "=========================================="
echo -e "Overall Score:    ${OVERALL_SCORE}%"
echo -e "Core Components:  ${CORE_SCORE}%"
echo -e "Training:         ${TRAINING_SCORE}%"
echo -e "Utils:           ${UTILS_SCORE}%"
echo ""

# Check if target was met
if [ "${OVERALL_SCORE%.*}" -ge "$MUTATION_SCORE_TARGET" ]; then
    echo -e "${GREEN}✅ SUCCESS: Mutation score target achieved!${NC}"
    echo -e "${GREEN}   Score: ${OVERALL_SCORE}% (Target: ${MUTATION_SCORE_TARGET}%)${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  IMPROVEMENT NEEDED: Mutation score below target${NC}"
    echo -e "${YELLOW}   Score: ${OVERALL_SCORE}% (Target: ${MUTATION_SCORE_TARGET}%)${NC}"
    echo ""
    echo -e "${BLUE}💡 Recommendations:${NC}"
    echo "   1. Review surviving mutants in target/mutants/html/index.html"
    echo "   2. Add more comprehensive test cases"
    echo "   3. Improve test assertions to catch edge cases"
    echo "   4. Consider property-based testing"
    exit 1
fi