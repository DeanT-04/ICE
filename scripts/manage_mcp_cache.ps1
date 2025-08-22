#!/usr/bin/env pwsh
# MCP Cache Management Script for Ultra-Fast AI Model
# Provides cache monitoring, cleanup, and maintenance operations

param(
    [string]$Action = "status",        # status, cleanup, clear, monitor, warm
    [string]$CacheDir = ".cache/mcp",
    [int]$MaxAgeDays = 1,              # 24 hours default
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

function Get-CacheDirectories {
    $basePath = Resolve-Path $CacheDir -ErrorAction SilentlyContinue
    if (-not $basePath) {
        return @()
    }
    
    $subdirs = @()
    if (Test-Path "$basePath/api") { $subdirs += "$basePath/api" }
    if (Test-Path "$basePath/tool") { $subdirs += "$basePath/tool" }
    if (Test-Path "$basePath/tools") { $subdirs += "$basePath/tools" }
    if (Test-Path "$basePath/data") { $subdirs += "$basePath/data" }
    
    # Include base directory if it has cache files
    if ((Get-ChildItem $basePath -Filter "*.json" -ErrorAction SilentlyContinue).Count -gt 0) {
        $subdirs += $basePath
    }
    
    return $subdirs
}

function Get-CacheStats {
    param([string[]]$CacheDirs)
    
    $stats = @{
        TotalFiles = 0
        TotalSizeBytes = 0
        TotalSizeMB = 0
        ExpiredFiles = 0
        ValidFiles = 0
        OldestFile = $null
        NewestFile = $null
        CacheHitEstimate = 0
    }
    
    $cutoffTime = (Get-Date).AddDays(-$MaxAgeDays)
    
    foreach ($dir in $CacheDirs) {
        if (-not (Test-Path $dir)) { continue }
        
        $files = Get-ChildItem $dir -Filter "*.json" -ErrorAction SilentlyContinue
        
        foreach ($file in $files) {
            $stats.TotalFiles++
            $stats.TotalSizeBytes += $file.Length
            
            if ($file.LastWriteTime -lt $cutoffTime) {
                $stats.ExpiredFiles++
            } else {
                $stats.ValidFiles++
            }
            
            if (-not $stats.OldestFile -or $file.LastWriteTime -lt $stats.OldestFile) {
                $stats.OldestFile = $file.LastWriteTime
            }
            
            if (-not $stats.NewestFile -or $file.LastWriteTime -gt $stats.NewestFile) {
                $stats.NewestFile = $file.LastWriteTime
            }
            
            # Try to parse cache entry for additional stats
            try {
                $content = Get-Content $file.FullName -Raw | ConvertFrom-Json
                if ($content.access_count) {
                    $stats.CacheHitEstimate += $content.access_count
                }
            } catch {
                # Ignore parsing errors for individual files
            }
        }
    }
    
    $stats.TotalSizeMB = [math]::Round($stats.TotalSizeBytes / 1MB, 2)
    
    return $stats
}

function Show-CacheStatus {
    Write-SectionHeader "MCP Cache Status"
    
    $cacheDirs = Get-CacheDirectories
    
    if ($cacheDirs.Count -eq 0) {
        Write-ColoredOutput "📁 No cache directories found" "Warning"
        Write-ColoredOutput "   Cache directory: $CacheDir" "Info"
        return
    }
    
    Write-ColoredOutput "📁 Cache Directories Found:" "Info"
    foreach ($dir in $cacheDirs) {
        Write-ColoredOutput "   - $dir" "Info"
    }
    
    $stats = Get-CacheStats -CacheDirs $cacheDirs
    
    Write-ColoredOutput "📊 Cache Statistics:" "Info"
    Write-ColoredOutput "   Total Files: $($stats.TotalFiles)" "Info"
    Write-ColoredOutput "   Total Size: $($stats.TotalSizeMB) MB ($($stats.TotalSizeBytes) bytes)" "Info"
    Write-ColoredOutput "   Valid Files: $($stats.ValidFiles)" $(if ($stats.ValidFiles -gt 0) { "Success" } else { "Warning" })
    Write-ColoredOutput "   Expired Files: $($stats.ExpiredFiles)" $(if ($stats.ExpiredFiles -gt 0) { "Warning" } else { "Success" })
    Write-ColoredOutput "   Cache Hit Estimate: $($stats.CacheHitEstimate)" "Info"
    
    if ($stats.OldestFile) {
        Write-ColoredOutput "   Oldest Entry: $($stats.OldestFile)" "Info"
    }
    
    if ($stats.NewestFile) {
        Write-ColoredOutput "   Newest Entry: $($stats.NewestFile)" "Info"
    }
    
    # Cache health assessment
    $healthStatus = "Unknown"
    $healthColor = "Info"
    
    if ($stats.TotalFiles -eq 0) {
        $healthStatus = "Empty"
        $healthColor = "Warning"
    } elseif ($stats.ExpiredFiles -eq 0) {
        $healthStatus = "Excellent"
        $healthColor = "Success"
    } elseif ($stats.ExpiredFiles -lt ($stats.TotalFiles * 0.2)) {
        $healthStatus = "Good"
        $healthColor = "Success"
    } elseif ($stats.ExpiredFiles -lt ($stats.TotalFiles * 0.5)) {
        $healthStatus = "Fair"
        $healthColor = "Warning"
    } else {
        $healthStatus = "Poor"
        $healthColor = "Error"
    }
    
    Write-ColoredOutput "   Cache Health: $healthStatus" $healthColor
    
    # Recommendations
    Write-ColoredOutput "💡 Recommendations:" "Info"
    if ($stats.ExpiredFiles -gt 0) {
        Write-ColoredOutput "   - Run cleanup to remove $($stats.ExpiredFiles) expired entries" "Warning"
    }
    
    if ($stats.TotalSizeMB -gt 100) {
        Write-ColoredOutput "   - Consider reducing cache size (current: $($stats.TotalSizeMB) MB)" "Warning"
    }
    
    if ($stats.TotalFiles -eq 0) {
        Write-ColoredOutput "   - Cache is empty, consider warming with common requests" "Info"
    }
}

function Invoke-CacheCleanup {
    Write-SectionHeader "MCP Cache Cleanup"
    
    $cacheDirs = Get-CacheDirectories
    
    if ($cacheDirs.Count -eq 0) {
        Write-ColoredOutput "📁 No cache directories found to cleanup" "Warning"
        return
    }
    
    $cutoffTime = (Get-Date).AddDays(-$MaxAgeDays)
    Write-ColoredOutput "🧹 Cleaning up files older than: $cutoffTime" "Info"
    Write-ColoredOutput "🔍 Dry Run Mode: $DryRun" "Info"
    
    $totalRemoved = 0
    $totalSpaceSaved = 0
    
    foreach ($dir in $cacheDirs) {
        if (-not (Test-Path $dir)) { continue }
        
        Write-ColoredOutput "📂 Processing: $dir" "Info"
        
        $files = Get-ChildItem $dir -Filter "*.json" -ErrorAction SilentlyContinue
        $removedFromDir = 0
        $spaceSavedFromDir = 0
        
        foreach ($file in $files) {
            if ($file.LastWriteTime -lt $cutoffTime) {
                $spaceSavedFromDir += $file.Length
                
                if (-not $DryRun) {
                    try {
                        Remove-Item $file.FullName -Force
                        Write-Verbose "   Removed: $($file.Name)"
                    } catch {
                        Write-ColoredOutput "   ⚠️ Failed to remove: $($file.Name) - $($_.Exception.Message)" "Warning"
                        continue
                    }
                } else {
                    Write-Verbose "   Would remove: $($file.Name)"
                }
                
                $removedFromDir++
            }
        }
        
        $totalRemoved += $removedFromDir
        $totalSpaceSaved += $spaceSavedFromDir
        
        if ($removedFromDir -gt 0) {
            $spaceMB = [math]::Round($spaceSavedFromDir / 1MB, 2)
            Write-ColoredOutput "   ✅ $removedFromDir files, $spaceMB MB" "Success"
        } else {
            Write-ColoredOutput "   ✅ No expired files found" "Success"
        }
    }
    
    $totalSpaceMB = [math]::Round($totalSpaceSaved / 1MB, 2)
    
    if ($DryRun) {
        Write-ColoredOutput "🔍 Dry Run Summary:" "Info"
        Write-ColoredOutput "   Would remove: $totalRemoved files" "Info"
        Write-ColoredOutput "   Would save: $totalSpaceMB MB" "Info"
    } else {
        Write-ColoredOutput "✅ Cleanup Summary:" "Success"
        Write-ColoredOutput "   Removed: $totalRemoved files" "Success"
        Write-ColoredOutput "   Space saved: $totalSpaceMB MB" "Success"
    }
}

function Clear-AllCaches {
    Write-SectionHeader "Clear All MCP Caches"
    
    if (-not (Test-Path $CacheDir)) {
        Write-ColoredOutput "📁 Cache directory doesn't exist: $CacheDir" "Warning"
        return
    }
    
    Write-ColoredOutput "⚠️ This will remove ALL cache files!" "Warning"
    Write-ColoredOutput "🔍 Dry Run Mode: $DryRun" "Info"
    
    $totalFiles = 0
    $totalSize = 0
    
    # Count files first
    $allFiles = Get-ChildItem $CacheDir -Recurse -Filter "*.json" -ErrorAction SilentlyContinue
    $totalFiles = $allFiles.Count
    $totalSize = ($allFiles | Measure-Object -Property Length -Sum).Sum
    
    if ($totalFiles -eq 0) {
        Write-ColoredOutput "✅ No cache files to remove" "Success"
        return
    }
    
    $totalSizeMB = [math]::Round($totalSize / 1MB, 2)
    
    if (-not $DryRun) {
        try {
            Remove-Item "$CacheDir\*" -Recurse -Force -Include "*.json"
            Write-ColoredOutput "✅ Cleared all caches:" "Success"
            Write-ColoredOutput "   Files removed: $totalFiles" "Success"
            Write-ColoredOutput "   Space freed: $totalSizeMB MB" "Success"
        } catch {
            Write-ColoredOutput "❌ Failed to clear caches: $($_.Exception.Message)" "Error"
        }
    } else {
        Write-ColoredOutput "🔍 Would clear:" "Info"
        Write-ColoredOutput "   Files: $totalFiles" "Info"
        Write-ColoredOutput "   Space: $totalSizeMB MB" "Info"
    }
}

function Start-CacheMonitoring {
    Write-SectionHeader "MCP Cache Monitoring"
    
    Write-ColoredOutput "👀 Starting cache monitoring (Ctrl+C to stop)..." "Info"
    Write-ColoredOutput "📊 Refresh interval: 30 seconds" "Info"
    
    try {
        while ($true) {
            Clear-Host
            Write-ColoredOutput "MCP Cache Monitor - $(Get-Date)" "Header"
            
            $cacheDirs = Get-CacheDirectories
            if ($cacheDirs.Count -gt 0) {
                $stats = Get-CacheStats -CacheDirs $cacheDirs
                
                Write-ColoredOutput "📊 Live Stats:" "Info"
                Write-ColoredOutput "   Files: $($stats.TotalFiles) ($($stats.ValidFiles) valid, $($stats.ExpiredFiles) expired)" "Info"
                Write-ColoredOutput "   Size: $($stats.TotalSizeMB) MB" "Info"
                Write-ColoredOutput "   Hits: $($stats.CacheHitEstimate)" "Info"
                
                # Show recent activity
                $recentFiles = Get-ChildItem $cacheDirs -Filter "*.json" -ErrorAction SilentlyContinue |
                    Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) } |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 5
                
                if ($recentFiles) {
                    Write-ColoredOutput "🕒 Recent Activity (last 5 minutes):" "Info"
                    foreach ($file in $recentFiles) {
                        Write-ColoredOutput "   $($file.LastWriteTime.ToString('HH:mm:ss')) - $($file.Name)" "Info"
                    }
                }
            } else {
                Write-ColoredOutput "📁 No cache directories found" "Warning"
            }
            
            Start-Sleep -Seconds 30
        }
    } catch {
        Write-ColoredOutput "🛑 Monitoring stopped" "Info"
    }
}

function Invoke-CacheWarming {
    Write-SectionHeader "MCP Cache Warming"
    
    Write-ColoredOutput "🔥 Cache warming is not yet implemented" "Warning"
    Write-ColoredOutput "💡 This would pre-populate the cache with common requests" "Info"
    Write-ColoredOutput "📝 To implement, integrate with the Rust cache warming utility" "Info"
}

function Show-Help {
    Write-SectionHeader "MCP Cache Management Help"
    
    Write-ColoredOutput "Usage: .\scripts\manage_mcp_cache.ps1 [options]" "Info"
    Write-ColoredOutput "" "Info"
    Write-ColoredOutput "Actions:" "Info"
    Write-ColoredOutput "  status   - Show cache status and statistics (default)" "Info"
    Write-ColoredOutput "  cleanup  - Remove expired cache entries" "Info"
    Write-ColoredOutput "  clear    - Remove ALL cache entries" "Info"
    Write-ColoredOutput "  monitor  - Start real-time cache monitoring" "Info"
    Write-ColoredOutput "  warm     - Pre-populate cache with common requests" "Info"
    Write-ColoredOutput "  help     - Show this help message" "Info"
    Write-ColoredOutput "" "Info"
    Write-ColoredOutput "Options:" "Info"
    Write-ColoredOutput "  -CacheDir <path>     - Cache directory (default: .cache/mcp)" "Info"
    Write-ColoredOutput "  -MaxAgeDays <days>   - Max age for cleanup (default: 1)" "Info"
    Write-ColoredOutput "  -Verbose             - Show detailed output" "Info"
    Write-ColoredOutput "  -DryRun              - Show what would be done without doing it" "Info"
    Write-ColoredOutput "" "Info"
    Write-ColoredOutput "Examples:" "Info"
    Write-ColoredOutput "  .\scripts\manage_mcp_cache.ps1 status" "Info"
    Write-ColoredOutput "  .\scripts\manage_mcp_cache.ps1 cleanup -Verbose" "Info"
    Write-ColoredOutput "  .\scripts\manage_mcp_cache.ps1 cleanup -DryRun" "Info"
    Write-ColoredOutput "  .\scripts\manage_mcp_cache.ps1 clear -CacheDir '.cache/mcp'" "Info"
}

# Main execution
try {
    Write-SectionHeader "Ultra-Fast AI Model - MCP Cache Manager"
    Write-ColoredOutput "Action: $Action" "Info"
    Write-ColoredOutput "Cache Directory: $CacheDir" "Info"
    
    switch ($Action.ToLower()) {
        "status" {
            Show-CacheStatus
        }
        "cleanup" {
            Invoke-CacheCleanup
        }
        "clear" {
            Clear-AllCaches
        }
        "monitor" {
            Start-CacheMonitoring
        }
        "warm" {
            Invoke-CacheWarming
        }
        "help" {
            Show-Help
        }
        default {
            Write-ColoredOutput "❌ Unknown action: $Action" "Error"
            Show-Help
            exit 1
        }
    }
    
    Write-ColoredOutput "✅ MCP cache management completed" "Success"
    
} catch {
    Write-ColoredOutput "❌ Error during cache management: $_" "Error"
    exit 1
}