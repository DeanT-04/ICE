#!/usr/bin/env pwsh
# Security Scanning Pipeline for Ultra-Fast AI Model
# Orchestrates cargo audit, trivy, gosec, and additional security tools

param(
    [string]$Mode = "full",         # full, quick, ci
    [string]$Config = "security/security-scan.toml",
    [switch]$UpdateDbs,
    [switch]$FailFast,
    [string]$OutputDir = "target/security"
)

# Script configuration
$ErrorActionPreference = "Continue"
$VerbosePreference = "Continue"

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

function Test-CommandExists {
    param([string]$Command)
    $exists = Get-Command $Command -ErrorAction SilentlyContinue
    return $null -ne $exists
}

function Install-SecurityTools {
    Write-SectionHeader "Installing Security Tools"
    
    # Install Rust security tools
    if (Test-CommandExists "cargo") {
        Write-ColoredOutput "Installing cargo-audit..." "Info"
        cargo install --locked cargo-audit
        
        Write-ColoredOutput "Installing cargo-deny..." "Info"
        cargo install --locked cargo-deny
        
        Write-ColoredOutput "Installing cargo-outdated..." "Info"
        cargo install --locked cargo-outdated
    } else {
        Write-ColoredOutput "Cargo not found - skipping Rust security tools" "Warning"
    }
    
    # Install Go security tools
    if (Test-CommandExists "go") {
        Write-ColoredOutput "Installing gosec..." "Info"
        go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
        
        Write-ColoredOutput "Installing govulncheck..." "Info"
        go install golang.org/x/vuln/cmd/govulncheck@latest
    } else {
        Write-ColoredOutput "Go not found - skipping Go security tools" "Warning"
    }
    
    # Install Trivy (container/dependency scanner)
    if (-not (Test-CommandExists "trivy")) {
        Write-ColoredOutput "Installing Trivy..." "Info"
        if ($IsWindows) {
            # Install via chocolatey if available, otherwise download binary
            if (Test-CommandExists "choco") {
                choco install trivy
            } else {
                Write-ColoredOutput "Please install Trivy manually: https://aquasecurity.github.io/trivy/" "Warning"
            }
        } else {
            # Linux/Mac installation
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
        }
    }
    
    Write-ColoredOutput "Security tools installation completed" "Success"
}

function Initialize-SecurityEnvironment {
    Write-SectionHeader "Initializing Security Environment"
    
    # Create output directories
    $directories = @($OutputDir, "$OutputDir/reports", "$OutputDir/cache", "$OutputDir/logs")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -Path $dir -ItemType Directory -Force | Out-Null
            Write-ColoredOutput "Created directory: $dir" "Info"
        }
    }
    
    # Initialize security log
    $logFile = "$OutputDir/logs/security-scan-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
    Start-Transcript -Path $logFile
    
    Write-ColoredOutput "Security environment initialized" "Success"
    Write-ColoredOutput "Log file: $logFile" "Info"
    
    return $logFile
}

function Update-SecurityDatabases {
    Write-SectionHeader "Updating Security Databases"
    
    # Update Trivy database
    if (Test-CommandExists "trivy") {
        Write-ColoredOutput "Updating Trivy vulnerability database..." "Info"
        trivy --cache-dir "$OutputDir/cache/trivy" --update-db 2>&1 | Tee-Object -FilePath "$OutputDir/logs/trivy-update.log"
    }
    
    # Update cargo audit database
    if (Test-CommandExists "cargo-audit") {
        Write-ColoredOutput "Updating cargo-audit database..." "Info"
        cargo audit --update-db 2>&1 | Tee-Object -FilePath "$OutputDir/logs/cargo-audit-update.log"
    }
    
    # Update Go vulnerability database
    if (Test-CommandExists "govulncheck") {
        Write-ColoredOutput "Updating Go vulnerability database..." "Info"
        go env -w GOPROXY=https://proxy.golang.org,direct
        go env -w GOSUMDB=sum.golang.org
    }
    
    Write-ColoredOutput "Security databases updated" "Success"
}

function Invoke-RustSecurityScan {
    Write-SectionHeader "Rust Security Scanning"
    $rustResults = @{}
    
    try {
        # Cargo audit for known vulnerabilities
        if (Test-CommandExists "cargo-audit") {
            Write-ColoredOutput "Running cargo audit..." "Info"
            $auditOutput = "$OutputDir/rust-audit.json"
            
            $auditResult = cargo audit --json --output $auditOutput 2>&1
            $rustResults.Audit = @{
                Success = $LASTEXITCODE -eq 0
                Output = $auditResult
                File = $auditOutput
            }
            
            if ($rustResults.Audit.Success) {
                Write-ColoredOutput "✅ Cargo audit completed successfully" "Success"
            } else {
                Write-ColoredOutput "❌ Cargo audit found vulnerabilities" "Error"
            }
        }
        
        # Cargo deny for license and security checks
        if (Test-CommandExists "cargo-deny") {
            Write-ColoredOutput "Running cargo deny..." "Info"
            $denyOutput = "$OutputDir/rust-deny.json"
            
            # Create cargo-deny configuration if it doesn't exist
            if (-not (Test-Path "deny.toml")) {
                $denyConfig = @"
[graph]
targets = []

[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"

[licenses]
unlicensed = "deny"
copyleft = "warn"
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"]
deny = ["GPL-2.0", "GPL-3.0"]

[bans]
multiple-versions = "warn"
wildcards = "warn"
highlight = "all"

[sources]
unknown-registry = "warn"
unknown-git = "warn"
"@
                $denyConfig | Out-File -FilePath "deny.toml" -Encoding UTF8
            }
            
            $denyResult = cargo deny --format json check 2>&1
            $rustResults.Deny = @{
                Success = $LASTEXITCODE -eq 0
                Output = $denyResult
                File = $denyOutput
            }
            
            if ($rustResults.Deny.Success) {
                Write-ColoredOutput "✅ Cargo deny completed successfully" "Success"
            } else {
                Write-ColoredOutput "⚠️ Cargo deny found issues" "Warning"
            }
        }
        
        # Cargo outdated for dependency updates
        if (Test-CommandExists "cargo-outdated") {
            Write-ColoredOutput "Checking for outdated dependencies..." "Info"
            $outdatedResult = cargo outdated --format json 2>&1
            $rustResults.Outdated = @{
                Success = $LASTEXITCODE -eq 0
                Output = $outdatedResult
            }
        }
        
        # Security-focused clippy
        Write-ColoredOutput "Running security-focused clippy..." "Info"
        $clippyArgs = @(
            "clippy", "--all-targets", "--", 
            "-D", "warnings",
            "-D", "clippy::suspicious",
            "-D", "clippy::perf",
            "-D", "clippy::mem_forget",
            "-W", "clippy::ptr_arg",
            "-W", "clippy::cast_ptr_alignment"
        )
        
        $clippyResult = & cargo @clippyArgs 2>&1
        $rustResults.Clippy = @{
            Success = $LASTEXITCODE -eq 0
            Output = $clippyResult
        }
        
        if ($rustResults.Clippy.Success) {
            Write-ColoredOutput "✅ Security clippy completed successfully" "Success"
        } else {
            Write-ColoredOutput "⚠️ Security clippy found issues" "Warning"
        }
        
    } catch {
        Write-ColoredOutput "Error during Rust security scan: $_" "Error"
        $rustResults.Error = $_.Exception.Message
    }
    
    return $rustResults
}

function Invoke-GoSecurityScan {
    Write-SectionHeader "Go Security Scanning"
    $goResults = @{}
    
    try {
        # Check if Go MCP directory exists
        $goSrcDir = "src/mcp"
        if (-not (Test-Path $goSrcDir)) {
            Write-ColoredOutput "Go source directory not found: $goSrcDir" "Warning"
            return $goResults
        }
        
        # Gosec for security issues
        if (Test-CommandExists "gosec") {
            Write-ColoredOutput "Running gosec security scanner..." "Info"
            $gosecOutput = "$OutputDir/go-gosec.json"
            
            Push-Location $goSrcDir
            try {
                $gosecResult = gosec -fmt json -out $gosecOutput ./... 2>&1
                $goResults.Gosec = @{
                    Success = $LASTEXITCODE -eq 0
                    Output = $gosecResult
                    File = $gosecOutput
                }
                
                if ($goResults.Gosec.Success) {
                    Write-ColoredOutput "✅ Gosec completed successfully" "Success"
                } else {
                    Write-ColoredOutput "❌ Gosec found security issues" "Error"
                }
            } finally {
                Pop-Location
            }
        }
        
        # Go vulnerability check
        if (Test-CommandExists "govulncheck") {
            Write-ColoredOutput "Running Go vulnerability check..." "Info"
            
            Push-Location $goSrcDir
            try {
                $vulnResult = govulncheck -json ./... 2>&1
                $goResults.VulnCheck = @{
                    Success = $LASTEXITCODE -eq 0
                    Output = $vulnResult
                }
                
                if ($goResults.VulnCheck.Success) {
                    Write-ColoredOutput "✅ Go vulnerability check completed successfully" "Success"
                } else {
                    Write-ColoredOutput "❌ Go vulnerability check found issues" "Error"
                }
            } finally {
                Pop-Location
            }
        }
        
        # Go vet for additional checks
        Write-ColoredOutput "Running go vet..." "Info"
        Push-Location $goSrcDir
        try {
            $vetResult = go vet ./... 2>&1
            $goResults.Vet = @{
                Success = $LASTEXITCODE -eq 0
                Output = $vetResult
            }
            
            if ($goResults.Vet.Success) {
                Write-ColoredOutput "✅ Go vet completed successfully" "Success"
            } else {
                Write-ColoredOutput "⚠️ Go vet found issues" "Warning"
            }
        } finally {
            Pop-Location
        }
        
    } catch {
        Write-ColoredOutput "Error during Go security scan: $_" "Error"
        $goResults.Error = $_.Exception.Message
    }
    
    return $goResults
}

function Invoke-ContainerSecurityScan {
    Write-SectionHeader "Container Security Scanning"
    $containerResults = @{}
    
    try {
        if (-not (Test-CommandExists "trivy")) {
            Write-ColoredOutput "Trivy not found - skipping container security scan" "Warning"
            return $containerResults
        }
        
        # Scan Dockerfile
        if (Test-Path "Dockerfile") {
            Write-ColoredOutput "Scanning Dockerfile..." "Info"
            $dockerOutput = "$OutputDir/trivy-dockerfile.json"
            
            $dockerResult = trivy config --format json --output $dockerOutput Dockerfile 2>&1
            $containerResults.Dockerfile = @{
                Success = $LASTEXITCODE -eq 0
                Output = $dockerResult
                File = $dockerOutput
            }
        }
        
        # Scan dependencies for vulnerabilities
        Write-ColoredOutput "Scanning dependencies for vulnerabilities..." "Info"
        $depsOutput = "$OutputDir/trivy-dependencies.json"
        
        $depsResult = trivy fs --format json --output $depsOutput --scanners vuln . 2>&1
        $containerResults.Dependencies = @{
            Success = $LASTEXITCODE -eq 0
            Output = $depsResult
            File = $depsOutput
        }
        
        # Secret scanning
        Write-ColoredOutput "Scanning for secrets..." "Info"
        $secretsOutput = "$OutputDir/trivy-secrets.json"
        
        $secretsResult = trivy fs --format json --output $secretsOutput --scanners secret . 2>&1
        $containerResults.Secrets = @{
            Success = $LASTEXITCODE -eq 0
            Output = $secretsResult
            File = $secretsOutput
        }
        
        # Misconfigurations
        Write-ColoredOutput "Scanning for misconfigurations..." "Info"
        $misconfigOutput = "$OutputDir/trivy-misconfig.json"
        
        $misconfigResult = trivy fs --format json --output $misconfigOutput --scanners config . 2>&1
        $containerResults.Misconfig = @{
            Success = $LASTEXITCODE -eq 0
            Output = $misconfigResult
            File = $misconfigOutput
        }
        
        Write-ColoredOutput "✅ Container security scanning completed" "Success"
        
    } catch {
        Write-ColoredOutput "Error during container security scan: $_" "Error"
        $containerResults.Error = $_.Exception.Message
    }
    
    return $containerResults
}

function Invoke-AdditionalSecurityChecks {
    Write-SectionHeader "Additional Security Checks"
    $additionalResults = @{}
    
    try {
        # Check for hardcoded secrets in source code
        Write-ColoredOutput "Scanning for hardcoded secrets..." "Info"
        $secretPatterns = @(
            "(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[\w\-]{20,}['\"]?",
            "(?i)password\s*[:=]\s*['\"][\w\-!@#$%^&*()]{8,}['\"]",
            "-----BEGIN [A-Z ]+-----",
            "(?i)private[_-]?key"
        )
        
        $secretFindings = @()
        Get-ChildItem -Recurse -Include "*.rs", "*.go", "*.zig", "*.toml", "*.yaml", "*.yml" | ForEach-Object {
            $file = $_.FullName
            $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
            
            if ($content) {
                foreach ($pattern in $secretPatterns) {
                    if ($content -match $pattern) {
                        $secretFindings += @{
                            File = $file
                            Pattern = $pattern
                            Context = $matches[0]
                        }
                    }
                }
            }
        }
        
        $additionalResults.SecretsFound = $secretFindings.Count
        if ($secretFindings.Count -gt 0) {
            Write-ColoredOutput "❌ Found $($secretFindings.Count) potential secrets" "Error"
            $secretFindings | ConvertTo-Json | Out-File "$OutputDir/secret-findings.json"
        } else {
            Write-ColoredOutput "✅ No hardcoded secrets found" "Success"
        }
        
        # Check for unsafe Rust patterns
        Write-ColoredOutput "Scanning for unsafe Rust patterns..." "Info"
        $unsafePatterns = @(
            "unsafe\s*{",
            "std::mem::transmute",
            "std::ptr::null",
            "std::slice::from_raw_parts"
        )
        
        $unsafeFindings = @()
        Get-ChildItem -Recurse -Include "*.rs" | ForEach-Object {
            $file = $_.FullName
            $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
            
            if ($content) {
                foreach ($pattern in $unsafePatterns) {
                    if ($content -match $pattern) {
                        $unsafeFindings += @{
                            File = $file
                            Pattern = $pattern
                        }
                    }
                }
            }
        }
        
        $additionalResults.UnsafePatternsFound = $unsafeFindings.Count
        if ($unsafeFindings.Count -gt 0) {
            Write-ColoredOutput "⚠️ Found $($unsafeFindings.Count) unsafe patterns" "Warning"
            $unsafeFindings | ConvertTo-Json | Out-File "$OutputDir/unsafe-patterns.json"
        } else {
            Write-ColoredOutput "✅ No concerning unsafe patterns found" "Success"
        }
        
    } catch {
        Write-ColoredOutput "Error during additional security checks: $_" "Error"
        $additionalResults.Error = $_.Exception.Message
    }
    
    return $additionalResults
}

function New-SecurityReport {
    param(
        [hashtable]$RustResults,
        [hashtable]$GoResults,
        [hashtable]$ContainerResults,
        [hashtable]$AdditionalResults
    )
    
    Write-SectionHeader "Generating Security Report"
    
    $reportData = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        ScanMode = $Mode
        Results = @{
            Rust = $RustResults
            Go = $GoResults
            Container = $ContainerResults
            Additional = $AdditionalResults
        }
    }
    
    # Save JSON report
    $jsonReport = "$OutputDir/security-report.json"
    $reportData | ConvertTo-Json -Depth 10 | Out-File $jsonReport
    
    # Generate HTML report
    $htmlReport = "$OutputDir/security-report.html"
    $htmlContent = @"
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report - Ultra-Fast AI Model</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        .success { background: #d4edda; border-color: #c3e6cb; }
        .warning { background: #fff3cd; border-color: #ffeaa7; }
        .error { background: #f8d7da; border-color: #f5c6cb; }
        .timestamp { color: #666; font-size: 0.9em; }
        pre { background: #f8f9fa; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p class="timestamp">Generated: $($reportData.Timestamp)</p>
        <p>Scan Mode: $($reportData.ScanMode)</p>
    </div>
"@
    
    # Add sections for each scan type
    foreach ($scanType in @("Rust", "Go", "Container", "Additional")) {
        $results = $reportData.Results[$scanType]
        if ($results.Count -gt 0) {
            $sectionClass = if ($results.ContainsKey("Error")) { "error" } else { "success" }
            $htmlContent += @"
    <div class="section $sectionClass">
        <h2>$scanType Security Scan</h2>
        <pre>$($results | ConvertTo-Json -Depth 3)</pre>
    </div>
"@
        }
    }
    
    $htmlContent += @"
</body>
</html>
"@
    
    $htmlContent | Out-File $htmlReport
    
    Write-ColoredOutput "Security reports generated:" "Success"
    Write-ColoredOutput "  JSON: $jsonReport" "Info"
    Write-ColoredOutput "  HTML: $htmlReport" "Info"
    
    return @{
        JsonReport = $jsonReport
        HtmlReport = $htmlReport
    }
}

function Test-SecurityExitCode {
    param(
        [hashtable]$RustResults,
        [hashtable]$GoResults,
        [hashtable]$ContainerResults,
        [hashtable]$AdditionalResults
    )
    
    $criticalIssues = 0
    $warnings = 0
    
    # Count critical issues
    foreach ($results in @($RustResults, $GoResults, $ContainerResults, $AdditionalResults)) {
        if ($results.ContainsKey("Error")) {
            $criticalIssues++
        }
        
        foreach ($scanResult in $results.Values) {
            if ($scanResult -is [hashtable] -and -not $scanResult.Success) {
                $warnings++
            }
        }
    }
    
    Write-SectionHeader "Security Scan Summary"
    Write-ColoredOutput "Critical Issues: $criticalIssues" $(if ($criticalIssues -gt 0) { "Error" } else { "Success" })
    Write-ColoredOutput "Warnings: $warnings" $(if ($warnings -gt 0) { "Warning" } else { "Success" })
    
    # Determine exit code based on configuration
    if ($criticalIssues -gt 0) {
        Write-ColoredOutput "❌ Security scan failed due to critical issues" "Error"
        return 1
    } elseif ($warnings -gt 0 -and $FailFast) {
        Write-ColoredOutput "⚠️ Security scan failed due to warnings (fail-fast mode)" "Warning"
        return 1
    } else {
        Write-ColoredOutput "✅ Security scan completed successfully" "Success"
        return 0
    }
}

# Main execution
try {
    Write-SectionHeader "Ultra-Fast AI Model - Security Scanning Pipeline"
    Write-ColoredOutput "Mode: $Mode" "Info"
    Write-ColoredOutput "Output Directory: $OutputDir" "Info"
    
    # Initialize environment
    $logFile = Initialize-SecurityEnvironment
    
    # Install tools if needed
    if ($Mode -eq "full") {
        Install-SecurityTools
    }
    
    # Update databases if requested
    if ($UpdateDbs -or $Mode -eq "full") {
        Update-SecurityDatabases
    }
    
    # Run security scans
    $rustResults = Invoke-RustSecurityScan
    $goResults = Invoke-GoSecurityScan
    $containerResults = Invoke-ContainerSecurityScan
    $additionalResults = Invoke-AdditionalSecurityChecks
    
    # Generate reports
    $reports = New-SecurityReport -RustResults $rustResults -GoResults $goResults -ContainerResults $containerResults -AdditionalResults $additionalResults
    
    # Determine exit code
    $exitCode = Test-SecurityExitCode -RustResults $rustResults -GoResults $goResults -ContainerResults $containerResults -AdditionalResults $additionalResults
    
} catch {
    Write-ColoredOutput "Fatal error during security scanning: $_" "Error"
    $exitCode = 2
} finally {
    if (Get-Command Stop-Transcript -ErrorAction SilentlyContinue) {
        Stop-Transcript
    }
}

exit $exitCode