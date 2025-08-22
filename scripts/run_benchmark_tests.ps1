#!/usr/bin/env pwsh
# Benchmark Validation Test Runner for Ultra-Fast AI Model
# Runs HumanEval and GSM8K validation tests with comprehensive reporting

param(
    [string]$TestFilter = "",           # Filter specific tests
    [string]$OutputDir = "target/benchmarks",
    [switch]$Verbose,
    [switch]$GenerateReport,
    [int]$MaxProblems = 5
)

# Script configuration
$ErrorActionPreference = "Continue"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Write-SectionHeader {
    param([string]$Title)
    Write-Host ""
    Write-ColoredOutput "========================================" "Header"
    Write-ColoredOutput "  $Title" "Header"
    Write-ColoredOutput "========================================" "Header"
}

function Initialize-BenchmarkEnvironment {
    Write-SectionHeader "Initializing Benchmark Environment"
    
    # Create output directories
    $directories = @($OutputDir, "$OutputDir/reports", "$OutputDir/logs")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-ColoredOutput "Created directory: $dir" "Info"
        }
    }
    
    # Check dependencies
    $dependencies = @("python3", "cargo")
    foreach ($dep in $dependencies) {
        if (-not (Get-Command $dep -ErrorAction SilentlyContinue)) {
            Write-ColoredOutput "Warning: $dep not found - some tests may fail" "Warning"
        } else {
            Write-ColoredOutput "✅ $dep available" "Success"
        }
    }
    
    Write-ColoredOutput "Benchmark environment initialized" "Success"
}

function Invoke-BenchmarkTests {
    Write-SectionHeader "Running Benchmark Validation Tests"
    
    # Build test command
    $testArgs = @("test", "--test", "benchmarks")
    
    if ($TestFilter) {
        $testArgs += "--", $TestFilter
    }
    
    if ($Verbose) {
        $testArgs += "--nocapture"
    }
    
    # Set environment variables for configuration
    $env:BENCHMARK_MAX_PROBLEMS = $MaxProblems
    $env:BENCHMARK_OUTPUT_DIR = $OutputDir
    $env:RUST_LOG = "info"
    
    Write-ColoredOutput "Running: cargo $($testArgs -join ' ')" "Info"
    
    # Run the tests
    $testResult = & cargo @testArgs 2>&1
    $exitCode = $LASTEXITCODE
    
    # Save test output
    $logFile = "$OutputDir/logs/benchmark_test_$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
    $testResult | Out-File -FilePath $logFile -Encoding UTF8
    
    Write-ColoredOutput "Test output saved to: $logFile" "Info"
    
    if ($exitCode -eq 0) {
        Write-ColoredOutput "✅ Benchmark tests completed successfully" "Success"
    } else {
        Write-ColoredOutput "❌ Some benchmark tests failed (exit code: $exitCode)" "Error"
    }
    
    return @{
        ExitCode = $exitCode
        Output = $testResult
        LogFile = $logFile
    }
}

function New-BenchmarkReport {
    param(
        [hashtable]$TestResults
    )
    
    if (-not $GenerateReport) {
        return
    }
    
    Write-SectionHeader "Generating Benchmark Report"
    
    # Check for benchmark result files
    $humanEvalResults = "$OutputDir/humaneval_results.json"
    $gsm8kResults = "$OutputDir/gsm8k_results.json"
    $detailedResults = "$OutputDir/detailed_results.json"
    $markdownReport = "$OutputDir/benchmark_report.md"
    
    # Generate HTML report if results exist
    if ((Test-Path $humanEvalResults) -or (Test-Path $gsm8kResults)) {
        $htmlReport = "$OutputDir/reports/benchmark_dashboard.html"
        
        $htmlContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Validation Dashboard - Ultra-Fast AI Model</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 8px; }
        .section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }
        .success { background: #d4edda; border-left: 4px solid #28a745; }
        .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .error { background: #f8d7da; border-left: 4px solid #dc3545; }
        .chart { width: 100%; height: 300px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Benchmark Validation Dashboard</h1>
            <p class="timestamp">Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')</p>
            <p>Ultra-Fast AI Model - HumanEval & GSM8K Validation</p>
        </div>
        
        <div class="section">
            <h2>📊 Test Execution Summary</h2>
            <div class="metric">
                <strong>Exit Code:</strong> $($TestResults.ExitCode)
                $(if ($TestResults.ExitCode -eq 0) { "✅" } else { "❌" })
            </div>
            <div class="metric">
                <strong>Max Problems:</strong> $MaxProblems
            </div>
            <div class="metric">
                <strong>Output Directory:</strong> $OutputDir
            </div>
        </div>
        
        <div class="section">
            <h2>📁 Generated Files</h2>
            <table>
                <tr><th>File</th><th>Status</th><th>Purpose</th></tr>
                <tr>
                    <td>humaneval_results.json</td>
                    <td>$(if (Test-Path $humanEvalResults) { "✅ Generated" } else { "❌ Missing" })</td>
                    <td>HumanEval benchmark detailed results</td>
                </tr>
                <tr>
                    <td>gsm8k_results.json</td>
                    <td>$(if (Test-Path $gsm8kResults) { "✅ Generated" } else { "❌ Missing" })</td>
                    <td>GSM8K benchmark detailed results</td>
                </tr>
                <tr>
                    <td>benchmark_report.md</td>
                    <td>$(if (Test-Path $markdownReport) { "✅ Generated" } else { "❌ Missing" })</td>
                    <td>Comprehensive markdown report</td>
                </tr>
                <tr>
                    <td>detailed_results.json</td>
                    <td>$(if (Test-Path $detailedResults) { "✅ Generated" } else { "❌ Missing" })</td>
                    <td>Combined benchmark results</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="../benchmark_report.md">Markdown Report</a></li>
                <li><a href="../humaneval_results.json">HumanEval Results</a></li>
                <li><a href="../gsm8k_results.json">GSM8K Results</a></li>
                <li><a href="../detailed_results.json">Detailed Results</a></li>
                <li><a href="../logs/">Test Logs</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📝 Next Steps</h2>
            <ul>
                <li>Review benchmark accuracy and performance metrics</li>
                <li>Analyze failed test cases for improvement opportunities</li>
                <li>Compare results with previous benchmark runs</li>
                <li>Consider increasing max_problems for more comprehensive evaluation</li>
                <li>Monitor zero-hallucination rates for quality assurance</li>
            </ul>
        </div>
    </div>
</body>
</html>
"@
        
        $htmlContent | Out-File -FilePath $htmlReport -Encoding UTF8
        Write-ColoredOutput "HTML dashboard generated: $htmlReport" "Success"
    }
    
    # Generate summary report
    $summaryReport = "$OutputDir/reports/execution_summary.md"
    $summaryContent = @"
# Benchmark Test Execution Summary

**Date**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
**Max Problems**: $MaxProblems
**Output Directory**: $OutputDir
**Test Filter**: $(if ($TestFilter) { $TestFilter } else { "None" })

## Execution Results

- **Exit Code**: $($TestResults.ExitCode) $(if ($TestResults.ExitCode -eq 0) { "✅" } else { "❌" })
- **Test Status**: $(if ($TestResults.ExitCode -eq 0) { "PASSED" } else { "FAILED" })
- **Log File**: $($TestResults.LogFile)

## Generated Artifacts

| File | Status | Description |
|------|--------|-------------|
| humaneval_results.json | $(if (Test-Path $humanEvalResults) { "✅" } else { "❌" }) | HumanEval benchmark results |
| gsm8k_results.json | $(if (Test-Path $gsm8kResults) { "✅" } else { "❌" }) | GSM8K benchmark results |
| benchmark_report.md | $(if (Test-Path $markdownReport) { "✅" } else { "❌" }) | Comprehensive report |
| detailed_results.json | $(if (Test-Path $detailedResults) { "✅" } else { "❌" }) | Combined results |

## Recommendations

$(if ($TestResults.ExitCode -eq 0) {
    "✅ All tests passed successfully. Review detailed results for performance analysis."
} else {
    "❌ Some tests failed. Check log files for error details and retry with increased timeouts if needed."
})

For detailed analysis, review the generated JSON files and markdown report.
"@
    
    $summaryContent | Out-File -FilePath $summaryReport -Encoding UTF8
    Write-ColoredOutput "Summary report generated: $summaryReport" "Success"
}

function Show-BenchmarkSummary {
    param(
        [hashtable]$TestResults
    )
    
    Write-SectionHeader "Benchmark Validation Summary"
    
    Write-ColoredOutput "📊 Test Execution Results:" "Info"
    Write-ColoredOutput "   Exit Code: $($TestResults.ExitCode)" $(if ($TestResults.ExitCode -eq 0) { "Success" } else { "Error" })
    Write-ColoredOutput "   Max Problems: $MaxProblems" "Info"
    Write-ColoredOutput "   Output Directory: $OutputDir" "Info"
    Write-ColoredOutput "   Log File: $($TestResults.LogFile)" "Info"
    
    # Check for result files
    $humanEvalExists = Test-Path "$OutputDir/humaneval_results.json"
    $gsm8kExists = Test-Path "$OutputDir/gsm8k_results.json"
    $reportExists = Test-Path "$OutputDir/benchmark_report.md"
    
    Write-ColoredOutput "📁 Generated Files:" "Info"
    Write-ColoredOutput "   HumanEval Results: $(if ($humanEvalExists) { "✅ Available" } else { "❌ Missing" })" $(if ($humanEvalExists) { "Success" } else { "Warning" })
    Write-ColoredOutput "   GSM8K Results: $(if ($gsm8kExists) { "✅ Available" } else { "❌ Missing" })" $(if ($gsm8kExists) { "Success" } else { "Warning" })
    Write-ColoredOutput "   Benchmark Report: $(if ($reportExists) { "✅ Available" } else { "❌ Missing" })" $(if ($reportExists) { "Success" } else { "Warning" })
    
    if ($GenerateReport) {
        Write-ColoredOutput "📋 Reports generated in: $OutputDir/reports/" "Success"
    }
    
    # Overall status
    if ($TestResults.ExitCode -eq 0 -and $humanEvalExists -and $gsm8kExists) {
        Write-ColoredOutput "🎉 Benchmark validation completed successfully!" "Success"
        Write-ColoredOutput "Review the generated reports for detailed performance analysis." "Info"
    } elseif ($TestResults.ExitCode -eq 0) {
        Write-ColoredOutput "⚠️ Tests passed but some result files are missing." "Warning"
        Write-ColoredOutput "This may indicate partial test execution or configuration issues." "Warning"
    } else {
        Write-ColoredOutput "❌ Benchmark validation failed." "Error"
        Write-ColoredOutput "Check the log file for detailed error information." "Error"
    }
}

# Main execution
try {
    Write-SectionHeader "Ultra-Fast AI Model - Benchmark Validation Runner"
    Write-ColoredOutput "Max Problems: $MaxProblems" "Info"
    Write-ColoredOutput "Output Directory: $OutputDir" "Info"
    Write-ColoredOutput "Generate Report: $GenerateReport" "Info"
    
    # Initialize environment
    Initialize-BenchmarkEnvironment
    
    # Run benchmark tests
    $testResults = Invoke-BenchmarkTests
    
    # Generate reports if requested
    New-BenchmarkReport -TestResults $testResults
    
    # Show summary
    Show-BenchmarkSummary -TestResults $testResults
    
    exit $testResults.ExitCode
    
} catch {
    Write-ColoredOutput "Fatal error during benchmark validation: $_" "Error"
    exit 2
}