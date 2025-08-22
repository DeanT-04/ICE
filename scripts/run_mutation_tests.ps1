# Mutation Testing Script for Ultra-Fast AI Model (Windows PowerShell)
# Target: ≥80% mutation score

param(
    [int]$MutationScoreTarget = 80,
    [int]$CoverageTarget = 85,
    [int]$MaxTimeout = 300,
    [int]$ParallelJobs = 4
)

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Blue" = [ConsoleColor]::Blue
        "White" = [ConsoleColor]::White
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

Write-ColorOutput "🧬 Starting Mutation Testing for Ultra-Fast AI Model" "Blue"
Write-ColorOutput "Target: ≥$MutationScoreTarget% mutation score" "Blue"
Write-ColorOutput "==========================================" "Blue"

# Create output directories
$directories = @(
    "target\mutants",
    "target\mutants\html", 
    "target\mutants\reports"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-ColorOutput "📋 Pre-flight checks..." "Blue"

# Check if cargo-mutants is installed
try {
    cargo mutants --version | Out-Null
    Write-ColorOutput "✅ cargo-mutants found" "Green"
} catch {
    Write-ColorOutput "⚠️  cargo-mutants not found. Installing..." "Yellow"
    cargo install --locked cargo-mutants
}

# Verify the project compiles and tests pass
Write-ColorOutput "🔧 Verifying project compilation..." "Blue"
$compileResult = cargo check --all-targets
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "❌ Project compilation failed. Please fix compilation errors first." "Red"
    exit 1
}

Write-ColorOutput "🧪 Running baseline tests..." "Blue"
$testResult = cargo test --all
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "❌ Baseline tests failed. Please fix failing tests first." "Red"
    exit 1
}

Write-ColorOutput "✅ Pre-flight checks passed" "Green"

# Function to run mutation testing phase
function Run-MutationPhase {
    param(
        [string]$PhaseName,
        [string]$ExaminePath,
        [string]$OutputFile
    )
    
    Write-ColorOutput "🧬 $PhaseName" "Blue"
    
    $args = @(
        "mutants",
        "--config", "mutants.toml",
        "--examine", $ExaminePath,
        "--timeout", $MaxTimeout,
        "--jobs", $ParallelJobs,
        "--output", $OutputFile
    )
    
    try {
        & cargo @args
    } catch {
        Write-ColorOutput "⚠️  Phase completed with some issues: $PhaseName" "Yellow"
    }
}

# Run mutation testing phases
Run-MutationPhase "Phase 1: Core Model Components" "src/model/core.rs" "target/mutants/core_results.json"
Run-MutationPhase "Phase 2: Fusion Components" "src/model/fusion.rs" "target/mutants/fusion_results.json"
Run-MutationPhase "Phase 3: Training Components" "src/training/" "target/mutants/training_results.json"
Run-MutationPhase "Phase 4: Utility Components" "src/utils/" "target/mutants/utils_results.json"

Write-ColorOutput "🧬 Phase 5: Complete Mutation Testing" "Blue"
try {
    cargo mutants --config mutants.toml --timeout $MaxTimeout --jobs $ParallelJobs --output target/mutants/complete_results.json --html-dir target/mutants/html
} catch {
    Write-ColorOutput "⚠️  Complete mutation testing completed with some issues" "Yellow"
}

# Analyze results
Write-ColorOutput "📊 Analyzing mutation testing results..." "Blue"

function Extract-MutationScore {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        try {
            $content = Get-Content $FilePath -Raw | ConvertFrom-Json
            if ($content.mutation_score) {
                return [math]::Round($content.mutation_score, 2)
            }
        } catch {
            # Fallback: simple text extraction
            $content = Get-Content $FilePath -Raw
            if ($content -match '"mutation_score":\s*([0-9.]+)') {
                return [math]::Round([double]$matches[1], 2)
            }
        }
    }
    return 0
}

# Extract scores
$coreScore = Extract-MutationScore "target/mutants/core_results.json"
$fusionScore = Extract-MutationScore "target/mutants/fusion_results.json"
$trainingScore = Extract-MutationScore "target/mutants/training_results.json"
$utilsScore = Extract-MutationScore "target/mutants/utils_results.json"
$overallScore = Extract-MutationScore "target/mutants/complete_results.json"

# Generate comprehensive report
Write-ColorOutput "📝 Generating comprehensive mutation testing report..." "Blue"

$reportContent = @"
# Mutation Testing Report - Ultra-Fast AI Model

## Summary
- **Overall Mutation Score**: $overallScore%
- **Target**: ≥$MutationScoreTarget%
- **Status**: $(if ($overallScore -ge $MutationScoreTarget) { "✅ PASSED" } else { "❌ NEEDS IMPROVEMENT" })

## Component Scores
- **Core Model Components**: $coreScore%
- **Fusion Components**: $fusionScore%
- **Training Components**: $trainingScore%
- **Utility Components**: $utilsScore%

## Testing Configuration
- **Timeout**: ${MaxTimeout}s per mutant
- **Parallel Jobs**: $ParallelJobs
- **Test Command**: ``cargo test --all``

## Files Analyzed
- ``src/model/core.rs`` - Neural network implementations
- ``src/model/fusion.rs`` - Model fusion logic  
- ``src/model/agentic.rs`` - Agentic coordination
- ``src/training/trainer.rs`` - Training loop
- ``src/training/genetic.rs`` - Genetic algorithms
- ``src/utils/perf.rs`` - Performance monitoring
- ``src/utils/energy.rs`` - Energy monitoring

## Mutation Types Tested
- ✅ Arithmetic operators (+, -, *, /, %)
- ✅ Logical operators (&&, ||, !)
- ✅ Relational operators (<, >, <=, >=, ==, !=)
- ✅ Constant values and boundaries
- ✅ Function call mutations
- ✅ Assignment operators

## Quality Gates
- **Minimum Mutation Score**: $MutationScoreTarget%
- **Core Components Target**: 90%
- **Training Components Target**: 80%
- **Utility Components Target**: 75%

## Recommendations
$(if ($overallScore -ge $MutationScoreTarget) { "✅ Mutation testing passed all quality gates." } else { "❌ Consider adding more comprehensive tests to improve mutation score." })

## Next Steps
1. Review surviving mutants in HTML report
2. Add tests for uncovered mutation cases
3. Improve test assertions to catch subtle bugs
4. Consider property-based testing for complex algorithms

## Generated Files
- **HTML Report**: ``target/mutants/html/index.html``
- **JSON Results**: ``target/mutants/complete_results.json``
- **Component Reports**: ``target/mutants/*_results.json``

---
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

$reportContent | Out-File -FilePath "target/mutants/reports/mutation_report.md" -Encoding UTF8

# Display results
Write-Host ""
Write-ColorOutput "📊 MUTATION TESTING RESULTS" "Blue"
Write-ColorOutput "==========================================" "Blue"
Write-Host "Overall Score:    $overallScore%"
Write-Host "Core Components:  $coreScore%"
Write-Host "Fusion:          $fusionScore%"
Write-Host "Training:         $trainingScore%"
Write-Host "Utils:           $utilsScore%"
Write-Host ""

# Check if target was met
if ($overallScore -ge $MutationScoreTarget) {
    Write-ColorOutput "✅ SUCCESS: Mutation score target achieved!" "Green"
    Write-ColorOutput "   Score: $overallScore% (Target: $MutationScoreTarget%)" "Green"
    exit 0
} else {
    Write-ColorOutput "⚠️  IMPROVEMENT NEEDED: Mutation score below target" "Yellow"
    Write-ColorOutput "   Score: $overallScore% (Target: $MutationScoreTarget%)" "Yellow"
    Write-Host ""
    Write-ColorOutput "💡 Recommendations:" "Blue"
    Write-Host "   1. Review surviving mutants in target/mutants/html/index.html"
    Write-Host "   2. Add more comprehensive test cases"
    Write-Host "   3. Improve test assertions to catch edge cases"
    Write-Host "   4. Consider property-based testing"
    exit 1
}