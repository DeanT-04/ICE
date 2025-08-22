//! MCP Caching System
//!
//! Provides efficient caching for Model Context Protocol (MCP) interactions
//! with configurable expiration times, cache validation, and performance optimization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::RwLock;
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc, Duration as ChronoDuration};

use crate::UltraFastAiError;

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Unique identifier for the cache entry
    pub key: String,
    /// Cached data as JSON string
    pub data: String,
    /// Timestamp when the entry was created
    pub created_at: i64,
    /// Timestamp when the entry expires
    pub expires_at: i64,
    /// Size of the cached data in bytes
    pub size_bytes: usize,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: i64,
    /// Hash of the original request for validation
    pub request_hash: String,
    /// Cache entry version for compatibility
    pub version: String,
}

impl CacheEntry {
    pub fn new(key: String, data: String, ttl_hours: u64, request_hash: String) -> Self {
        let now = Utc::now().timestamp();
        let expires_at = now + (ttl_hours as i64 * 3600);
        
        Self {
            key,
            size_bytes: data.len(),
            data,
            created_at: now,
            expires_at,
            access_count: 0,
            last_accessed: now,
            request_hash,
            version: "1.0".to_string(),
        }
    }
    
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() > self.expires_at
    }
    
    pub fn update_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now().timestamp();
    }
    
    pub fn time_to_expiry(&self) -> Duration {
        let now = Utc::now().timestamp();
        let seconds_left = (self.expires_at - now).max(0) as u64;
        Duration::from_secs(seconds_left)
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Base cache directory
    pub cache_dir: PathBuf,
    /// Default TTL in hours
    pub default_ttl_hours: u64,
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Enable compression for large entries
    pub enable_compression: bool,
    /// Cleanup interval in minutes
    pub cleanup_interval_minutes: u64,
    /// Enable cache metrics
    pub enable_metrics: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cache/mcp"),
            default_ttl_hours: 24,
            max_cache_size_mb: 500,
            max_entries: 10000,
            enable_compression: true,
            cleanup_interval_minutes: 60,
            enable_metrics: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expired_cleanups: u64,
    pub last_cleanup: i64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            total_size_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            expired_cleanups: 0,
            last_cleanup: Utc::now().timestamp(),
        }
    }
}

/// MCP Cache Manager
pub struct McpCache {
    config: CacheConfig,
    entries: RwLock<HashMap<String, CacheEntry>>,
    stats: RwLock<CacheStats>,
}

impl McpCache {
    /// Create a new MCP cache instance
    pub async fn new(config: CacheConfig) -> Result<Self, UltraFastAiError> {
        // Ensure cache directory exists
        fs::create_dir_all(&config.cache_dir).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to create cache directory: {}", e)))?;
        
        let cache = Self {
            config,
            entries: RwLock::new(HashMap::new()),
            stats: RwLock::new(CacheStats::default()),
        };
        
        // Load existing cache entries
        cache.load_cache_from_disk().await?;
        
        Ok(cache)
    }
    
    /// Generate cache key from request data
    pub fn generate_key(&self, request_type: &str, request_data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(request_type.as_bytes());
        hasher.update(request_data.as_bytes());
        let hash = hasher.finalize();
        format!("{:x}", hash)
    }
    
    /// Store data in cache
    pub async fn set(
        &self,
        key: String,
        data: String,
        request_hash: String,
        ttl_hours: Option<u64>,
    ) -> Result<(), UltraFastAiError> {
        let ttl = ttl_hours.unwrap_or(self.config.default_ttl_hours);
        let entry = CacheEntry::new(key.clone(), data, ttl, request_hash);
        
        // Check cache size limits
        {
            let mut entries = self.entries.write().await;
            let mut stats = self.stats.write().await;
            
            // Remove expired entries first
            self.cleanup_expired_entries(&mut entries, &mut stats).await;
            
            // Check if we need to evict entries
            if entries.len() >= self.config.max_entries {
                self.evict_lru_entries(&mut entries, &mut stats, 1).await;
            }
            
            // Update statistics
            if let Some(old_entry) = entries.get(&key) {
                stats.total_size_bytes -= old_entry.size_bytes as u64;
            } else {
                stats.total_entries += 1;
            }
            
            stats.total_size_bytes += entry.size_bytes as u64;
            
            // Check size limit
            let max_size_bytes = self.config.max_cache_size_mb * 1024 * 1024;
            while stats.total_size_bytes > max_size_bytes && !entries.is_empty() {
                self.evict_lru_entries(&mut entries, &mut stats, 1).await;
            }
            
            entries.insert(key.clone(), entry.clone());
        }
        
        // Persist to disk
        self.save_entry_to_disk(&entry).await?;
        
        Ok(())
    }
    
    /// Retrieve data from cache
    pub async fn get(&self, key: &str) -> Result<Option<String>, UltraFastAiError> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = entries.get_mut(key) {
            if entry.is_expired() {
                // Store size before removal
                let entry_size = entry.size_bytes as u64;
                
                // Remove expired entry
                entries.remove(key);
                stats.total_entries -= 1;
                stats.total_size_bytes -= entry_size;
                stats.misses += 1;
                
                // Clean up disk file
                let _ = self.remove_entry_from_disk(key).await;
                
                return Ok(None);
            }
            
            // Update access statistics
            entry.update_access();
            stats.hits += 1;
            
            Ok(Some(entry.data.clone()))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }
    
    /// Check if key exists and is not expired
    pub async fn contains_key(&self, key: &str) -> bool {
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(key) {
            !entry.is_expired()
        } else {
            false
        }
    }
    
    /// Remove entry from cache
    pub async fn remove(&self, key: &str) -> Result<bool, UltraFastAiError> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = entries.remove(key) {
            stats.total_entries -= 1;
            stats.total_size_bytes -= entry.size_bytes as u64;
            
            // Remove from disk
            self.remove_entry_from_disk(key).await?;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Clear all cache entries
    pub async fn clear(&self) -> Result<(), UltraFastAiError> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        entries.clear();
        *stats = CacheStats::default();
        
        // Clear disk cache
        self.clear_disk_cache().await?;
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
    
    /// Cleanup expired entries
    pub async fn cleanup_expired(&self) -> Result<usize, UltraFastAiError> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        let removed_count = self.cleanup_expired_entries(&mut entries, &mut stats).await;
        stats.last_cleanup = Utc::now().timestamp();
        
        Ok(removed_count)
    }
    
    /// Get cache entry info
    pub async fn get_entry_info(&self, key: &str) -> Option<CacheEntry> {
        let entries = self.entries.read().await;
        entries.get(key).cloned()
    }
    
    /// List all cache keys
    pub async fn list_keys(&self) -> Vec<String> {
        let entries = self.entries.read().await;
        entries.keys().cloned().collect()
    }
    
    /// Get cache hit ratio
    pub async fn hit_ratio(&self) -> f64 {
        let stats = self.stats.read().await;
        let total_requests = stats.hits + stats.misses;
        if total_requests > 0 {
            stats.hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }
    
    // Private helper methods
    
    async fn cleanup_expired_entries(
        &self,
        entries: &mut HashMap<String, CacheEntry>,
        stats: &mut CacheStats,
    ) -> usize {
        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();
        
        let mut removed_count = 0;
        for key in expired_keys {
            if let Some(entry) = entries.remove(&key) {
                stats.total_entries -= 1;
                stats.total_size_bytes -= entry.size_bytes as u64;
                stats.expired_cleanups += 1;
                removed_count += 1;
                
                // Remove from disk (ignore errors)
                let _ = self.remove_entry_from_disk(&key).await;
            }
        }
        
        removed_count
    }
    
    async fn evict_lru_entries(
        &self,
        entries: &mut HashMap<String, CacheEntry>,
        stats: &mut CacheStats,
        count: usize,
    ) {
        // Sort by last_accessed (LRU)
        let mut entry_refs: Vec<_> = entries.iter().collect();
        entry_refs.sort_by_key(|(_, entry)| entry.last_accessed);
        
        let keys_to_remove: Vec<String> = entry_refs
            .into_iter()
            .take(count)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in keys_to_remove {
            if let Some(entry) = entries.remove(&key) {
                stats.total_entries -= 1;
                stats.total_size_bytes -= entry.size_bytes as u64;
                stats.evictions += 1;
                
                // Remove from disk (ignore errors)
                let _ = self.remove_entry_from_disk(&key).await;
            }
        }
    }
    
    async fn save_entry_to_disk(&self, entry: &CacheEntry) -> Result<(), UltraFastAiError> {
        let file_path = self.config.cache_dir.join(format!("{}.json", entry.key));
        let json_data = serde_json::to_string_pretty(entry)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize cache entry: {}", e)))?;
        
        fs::write(&file_path, json_data).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to write cache entry to disk: {}", e)))?;
        
        Ok(())
    }
    
    async fn remove_entry_from_disk(&self, key: &str) -> Result<(), UltraFastAiError> {
        let file_path = self.config.cache_dir.join(format!("{}.json", key));
        if file_path.exists() {
            fs::remove_file(file_path).await
                .map_err(|e| UltraFastAiError::from(format!("Failed to remove cache file: {}", e)))?;
        }
        Ok(())
    }
    
    async fn load_cache_from_disk(&self) -> Result<(), UltraFastAiError> {
        if !self.config.cache_dir.exists() {
            return Ok(());
        }
        
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        let mut dir_entries = fs::read_dir(&self.config.cache_dir).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read cache directory: {}", e)))?;
        
        while let Some(entry) = dir_entries.next_entry().await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_entry_from_file(&path).await {
                    Ok(cache_entry) => {
                        if !cache_entry.is_expired() {
                            stats.total_entries += 1;
                            stats.total_size_bytes += cache_entry.size_bytes as u64;
                            entries.insert(cache_entry.key.clone(), cache_entry);
                        } else {
                            // Remove expired file
                            let _ = fs::remove_file(&path).await;
                        }
                    }
                    Err(_) => {
                        // Remove corrupted file
                        let _ = fs::remove_file(&path).await;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn load_entry_from_file(&self, path: &Path) -> Result<CacheEntry, UltraFastAiError> {
        let content = fs::read_to_string(path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read cache file: {}", e)))?;
        
        let entry: CacheEntry = serde_json::from_str(&content)
            .map_err(|e| UltraFastAiError::from(format!("Failed to parse cache entry: {}", e)))?;
        
        Ok(entry)
    }
    
    async fn clear_disk_cache(&self) -> Result<(), UltraFastAiError> {
        if !self.config.cache_dir.exists() {
            return Ok(());
        }
        
        let mut dir_entries = fs::read_dir(&self.config.cache_dir).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read cache directory: {}", e)))?;
        
        while let Some(entry) = dir_entries.next_entry().await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                let _ = fs::remove_file(path).await; // Ignore errors for cleanup
            }
        }
        
        Ok(())
    }
}

/// Cache manager factory for different cache types
pub struct CacheManager;

impl CacheManager {
    /// Create cache for MCP API responses
    pub async fn create_api_cache() -> Result<McpCache, UltraFastAiError> {
        let config = CacheConfig {
            cache_dir: PathBuf::from(".cache/mcp/api"),
            default_ttl_hours: 24,
            max_cache_size_mb: 200,
            max_entries: 5000,
            enable_compression: true,
            cleanup_interval_minutes: 30,
            enable_metrics: true,
        };
        
        McpCache::new(config).await
    }
    
    /// Create cache for MCP tool responses
    pub async fn create_tool_cache() -> Result<McpCache, UltraFastAiError> {
        let config = CacheConfig {
            cache_dir: PathBuf::from(".cache/mcp/tools"),
            default_ttl_hours: 12, // Tools might change more frequently
            max_cache_size_mb: 100,
            max_entries: 2000,
            enable_compression: true,
            cleanup_interval_minutes: 15,
            enable_metrics: true,
        };
        
        McpCache::new(config).await
    }
    
    /// Create cache for MCP data responses  
    pub async fn create_data_cache() -> Result<McpCache, UltraFastAiError> {
        let config = CacheConfig {
            cache_dir: PathBuf::from(".cache/mcp/data"),
            default_ttl_hours: 48, // Data can be cached longer
            max_cache_size_mb: 300,
            max_entries: 3000,
            enable_compression: true,
            cleanup_interval_minutes: 60,
            enable_metrics: true,
        };
        
        McpCache::new(config).await
    }
}