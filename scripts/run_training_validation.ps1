#!/usr/bin/env pwsh
# Training Time Validation Test Runner for Ultra-Fast AI Model
# Validates 24-hour training constraint on RTX 2070 Ti hardware

param(
    [string]$TestType = "quick",        # quick, standard, full
    [string]$LogLevel = "info",         # trace, debug, info, warn, error
    [switch]$SaveReports,
    [switch]$Verbose,
    [switch]$DryRun
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

function Test-Prerequisites {
    Write-SectionHeader "Checking Prerequisites"
    
    # Check Rust installation
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-ColoredOutput "❌ Cargo not found. Please install Rust." "Error"
        return $false
    }
    
    $cargoVersion = cargo --version
    Write-ColoredOutput "✅ Cargo: $cargoVersion" "Success"
    
    # Check if we're in the right directory
    if (-not (Test-Path "Cargo.toml")) {
        Write-ColoredOutput "❌ Cargo.toml not found. Run from project root." "Error"
        return $false
    }
    
    # Check for required dependencies
    $cargoToml = Get-Content "Cargo.toml" -Raw
    $requiredDeps = @("tokio", "serde", "serde_json", "chrono")
    
    foreach ($dep in $requiredDeps) {
        if ($cargoToml -notmatch $dep) {
            Write-ColoredOutput "⚠️ Missing dependency: $dep" "Warning"
        } else {
            Write-Verbose "✅ Dependency found: $dep"
        }
    }
    
    # Check available disk space (need space for checkpoints and reports)
    $availableSpace = (Get-PSDrive C).Free / 1GB
    if ($availableSpace -lt 5) {
        Write-ColoredOutput "⚠️ Low disk space: ${availableSpace:.1}GB available (recommend >5GB)" "Warning"
    } else {
        Write-ColoredOutput "✅ Disk space: ${availableSpace:.1}GB available" "Success"
    }
    
    return $true
}

function Run-QuickValidation {
    Write-SectionHeader "Quick Training Validation (5-10 minutes)"
    
    Write-ColoredOutput "🚀 Running rapid training simulation..." "Info"
    
    $env:RUST_LOG = $LogLevel
    $testCommand = "cargo test test_rapid_training_simulation --release"
    
    if ($DryRun) {
        Write-ColoredOutput "🔍 DRY RUN: Would execute: $testCommand" "Info"
        return $true
    }
    
    $result = Invoke-Expression $testCommand
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-ColoredOutput "✅ Quick validation passed" "Success"
        return $true
    } else {
        Write-ColoredOutput "❌ Quick validation failed (exit code: $exitCode)" "Error"
        return $false
    }
}

function Run-StandardValidation {
    Write-SectionHeader "Standard Training Validation (15-30 minutes)"
    
    $tests = @(
        "test_rapid_training_simulation",
        "test_constraint_validation_system", 
        "test_rtx_2070_ti_hardware_simulation",
        "test_parameter_count_validation",
        "test_energy_efficiency_validation",
        "test_training_progress_monitoring",
        "test_checkpoint_and_recovery"
    )
    
    $passed = 0
    $failed = 0
    
    foreach ($test in $tests) {
        Write-ColoredOutput "🧪 Running: $test" "Info"
        
        $env:RUST_LOG = $LogLevel
        $testCommand = "cargo test $test --release"
        
        if ($DryRun) {
            Write-ColoredOutput "🔍 DRY RUN: Would execute: $testCommand" "Info"
            $passed++
            continue
        }
        
        $result = Invoke-Expression $testCommand
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-ColoredOutput "   ✅ PASSED" "Success"
            $passed++
        } else {
            Write-ColoredOutput "   ❌ FAILED (exit code: $exitCode)" "Error"
            $failed++
        }
    }
    
    Write-ColoredOutput "📊 Standard validation results: $passed passed, $failed failed" "Info"
    return ($failed -eq 0)
}

function Run-FullValidation {
    Write-SectionHeader "Full 24-Hour Training Validation"
    
    Write-ColoredOutput "⚠️  WARNING: This test runs for up to 24 hours!" "Warning"
    Write-ColoredOutput "⚠️  It will consume significant system resources." "Warning"
    Write-ColoredOutput "⚠️  Ensure your system can run uninterrupted for 24+ hours." "Warning"
    
    if (-not $DryRun) {
        $confirmation = Read-Host "Type 'YES' to confirm you want to run the 24-hour test"
        if ($confirmation -ne "YES") {
            Write-ColoredOutput "❌ Full validation cancelled by user" "Warning"
            return $false
        }
    }
    
    Write-ColoredOutput "🕰️ Starting full 24-hour training validation..." "Info"
    Write-ColoredOutput "📝 This test will generate detailed reports and checkpoints" "Info"
    
    $env:RUST_LOG = $LogLevel
    $testCommand = "cargo test test_full_24_hour_training_validation --release --ignored -- --nocapture"
    
    if ($DryRun) {
        Write-ColoredOutput "🔍 DRY RUN: Would execute: $testCommand" "Info"
        Write-ColoredOutput "🔍 This would run for up to 24 hours" "Info"
        return $true
    }
    
    $startTime = Get-Date
    Write-ColoredOutput "⏰ Started at: $startTime" "Info"
    
    try {
        $result = Invoke-Expression $testCommand
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-ColoredOutput "⏰ Completed at: $endTime" "Info"
        Write-ColoredOutput "⏱️ Total duration: $($duration.ToString('hh\:mm\:ss'))" "Info"
        
        if ($exitCode -eq 0) {
            Write-ColoredOutput "🎉 FULL 24-HOUR VALIDATION PASSED!" "Success"
            Write-ColoredOutput "✅ System is validated for production deployment" "Success"
            return $true
        } else {
            Write-ColoredOutput "❌ Full validation failed (exit code: $exitCode)" "Error"
            return $false
        }
    } catch {
        Write-ColoredOutput "❌ Full validation failed with exception: $_" "Error"
        return $false
    }
}

function Cleanup-TestArtifacts {
    Write-SectionHeader "Cleaning Up Test Artifacts"
    
    $cleanupDirs = @(
        "target/checkpoints/24h_validation",
        "target/validation_reports",
        "target/benchmarks",
        ".cache/mcp"
    )
    
    foreach ($dir in $cleanupDirs) {
        if (Test-Path $dir) {
            if ($DryRun) {
                Write-ColoredOutput "🔍 DRY RUN: Would clean: $dir" "Info"
            } else {
                try {
                    Remove-Item $dir -Recurse -Force
                    Write-ColoredOutput "   🧹 Cleaned: $dir" "Success"
                } catch {
                    Write-ColoredOutput "   ⚠️ Failed to clean: $dir - $_" "Warning"
                }
            }
        } else {
            Write-Verbose "   ➖ Not found: $dir"
        }
    }
}

function Generate-TestReport {
    param([bool]$Success, [string]$TestType, [timespan]$Duration)
    
    if (-not $SaveReports) {
        return
    }
    
    Write-SectionHeader "Generating Test Report"
    
    $reportDir = "target/test_reports"
    if (-not (Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $reportPath = "$reportDir/training_validation_${TestType}_${timestamp}.md"
    
    $report = @"
# Training Validation Test Report

**Test Type**: $TestType
**Timestamp**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
**Duration**: $($Duration.ToString('hh\:mm\:ss'))
**Result**: $(if ($Success) { "✅ PASSED" } else { "❌ FAILED" })

## Test Configuration

- **Test Type**: $TestType
- **Log Level**: $LogLevel
- **Dry Run**: $DryRun
- **Save Reports**: $SaveReports

## System Information

- **OS**: $env:OS
- **Computer**: $env:COMPUTERNAME
- **User**: $env:USERNAME
- **PowerShell Version**: $($PSVersionTable.PSVersion)

## Test Results

$(if ($Success) {
    "All tests passed successfully. The ultra-fast AI model meets the 24-hour training constraint on RTX 2070 Ti hardware."
} else {
    "Some tests failed. Review the test output for details on constraint violations or system issues."
})

## Generated Files

- Checkpoints: `target/checkpoints/24h_validation/`
- Validation Reports: `target/validation_reports/`
- Benchmark Results: `target/benchmarks/`
- Test Reports: `target/test_reports/`

## Next Steps

$(if ($Success) {
    "- ✅ System validated for production deployment
- Consider running the full 24-hour test if not already done
- Review energy efficiency metrics for optimization opportunities"
} else {
    "- ❌ Address failing tests before production deployment
- Review constraint violations and system configuration
- Consider hardware upgrades or model optimization"
})
"@

    try {
        $report | Out-File -FilePath $reportPath -Encoding UTF8
        Write-ColoredOutput "📊 Test report saved: $reportPath" "Success"
    } catch {
        Write-ColoredOutput "⚠️ Failed to save test report: $_" "Warning"
    }
}

# Main execution
try {
    Write-SectionHeader "Ultra-Fast AI Model - Training Validation Runner"
    Write-ColoredOutput "Test Type: $TestType" "Info"
    Write-ColoredOutput "Log Level: $LogLevel" "Info"
    Write-ColoredOutput "Dry Run: $DryRun" "Info"
    
    $overallStart = Get-Date
    
    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        Write-ColoredOutput "❌ Prerequisites not met. Exiting." "Error"
        exit 1
    }
    
    # Create necessary directories
    $requiredDirs = @("target/checkpoints", "target/validation_reports", "target/benchmarks")
    foreach ($dir in $requiredDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Verbose "📁 Created directory: $dir"
        }
    }
    
    # Run tests based on type
    $success = $false
    
    switch ($TestType.ToLower()) {
        "quick" {
            $success = Run-QuickValidation
        }
        "standard" {
            $success = Run-StandardValidation
        }
        "full" {
            $success = Run-FullValidation
        }
        default {
            Write-ColoredOutput "❌ Unknown test type: $TestType" "Error"
            Write-ColoredOutput "   Valid types: quick, standard, full" "Info"
            exit 1
        }
    }
    
    $overallDuration = (Get-Date) - $overallStart
    
    # Generate report
    Generate-TestReport -Success $success -TestType $TestType -Duration $overallDuration
    
    # Cleanup (optional)
    if (-not $SaveReports) {
        $cleanup = Read-Host "Clean up test artifacts? (y/N)"
        if ($cleanup -eq "y" -or $cleanup -eq "Y") {
            Cleanup-TestArtifacts
        }
    }
    
    # Final summary
    Write-SectionHeader "Training Validation Summary"
    Write-ColoredOutput "Test Type: $TestType" "Info"
    Write-ColoredOutput "Duration: $($overallDuration.ToString('hh\:mm\:ss'))" "Info"
    
    if ($success) {
        Write-ColoredOutput "🎉 TRAINING VALIDATION SUCCESSFUL!" "Success"
        Write-ColoredOutput "✅ Ultra-fast AI model meets performance constraints" "Success"
        
        if ($TestType -eq "full") {
            Write-ColoredOutput "🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT!" "Success"
        } else {
            Write-ColoredOutput "💡 Consider running full 24-hour validation for production readiness" "Info"
        }
        
        exit 0
    } else {
        Write-ColoredOutput "❌ TRAINING VALIDATION FAILED" "Error"
        Write-ColoredOutput "🔧 Review test output and address constraint violations" "Error"
        exit 1
    }
    
} catch {
    Write-ColoredOutput "❌ Script execution failed: $_" "Error"
    exit 1
}