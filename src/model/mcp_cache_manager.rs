//! MCP Cache Manager Integration
//!
//! Provides transparent caching for MCP (Model Context Protocol) interactions
//! with automatic cache invalidation, request optimization, and performance monitoring.

use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::model::mcp::*;
use crate::utils::mcp_cache::*;
use crate::UltraFastAiError;

/// Cached MCP request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedMcpRequest {
    pub request_type: String,
    pub request_data: String,
    pub parameters: Option<serde_json::Value>,
    pub timestamp: i64,
}

/// Cached MCP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedMcpResponse {
    pub response_data: String,
    pub metadata: Option<serde_json::Value>,
    pub cache_hit: bool,
    pub execution_time_ms: u64,
}

/// Cache-enabled MCP client
pub struct CachedMcpClient {
    client: std::sync::Arc<tokio::sync::Mutex<McpClient>>,
    api_cache: Arc<McpCache>,
    tool_cache: Arc<McpCache>,
    data_cache: Arc<McpCache>,
    cache_stats: Arc<RwLock<CacheManagerStats>>,
}

/// Cache manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManagerStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_errors: u64,
    pub total_time_saved_ms: u64,
    pub api_requests: u64,
    pub tool_requests: u64,
    pub data_requests: u64,
}

impl Default for CacheManagerStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_errors: 0,
            total_time_saved_ms: 0,
            api_requests: 0,
            tool_requests: 0,
            data_requests: 0,
        }
    }
}

impl CachedMcpClient {
    /// Create a new cached MCP client
    pub async fn new(server_url: String) -> Result<Self, UltraFastAiError> {
        let config = McpConfig {
            api_base_url: format!("{}/api", server_url),
            tools_base_url: format!("{}/tools", server_url),
            data_base_url: format!("{}/data", server_url),
            feedback_base_url: format!("{}/feedback", server_url),
            ..McpConfig::default()
        };
        let client = std::sync::Arc::new(tokio::sync::Mutex::new(McpClient::new(config)));
        
        let api_cache = Arc::new(CacheManager::create_api_cache().await?);
        let tool_cache = Arc::new(CacheManager::create_tool_cache().await?);
        let data_cache = Arc::new(CacheManager::create_data_cache().await?);
        
        Ok(Self {
            client,
            api_cache,
            tool_cache,
            data_cache,
            cache_stats: Arc::new(RwLock::new(CacheManagerStats::default())),
        })
    }
    
    /// Execute API call with caching
    pub async fn cached_api_call(
        &self,
        endpoint: &str,
        params: Option<serde_json::Value>,
    ) -> Result<CachedMcpResponse, UltraFastAiError> {
        let request = CachedMcpRequest {
            request_type: "api".to_string(),
            request_data: endpoint.to_string(),
            parameters: params.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        let cache_key = self.api_cache.generate_key("api", &format!("{}:{:?}", endpoint, params));
        let request_hash = self.generate_request_hash(&request)?;
        
        let start_time = Instant::now();
        
        // Try cache first
        if let Ok(Some(cached_data)) = self.api_cache.get(&cache_key).await {
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            self.update_stats(|stats| {
                stats.total_requests += 1;
                stats.api_requests += 1;
                stats.cache_hits += 1;
                stats.total_time_saved_ms += execution_time;
            }).await;
            
            return Ok(CachedMcpResponse {
                response_data: cached_data,
                metadata: Some(serde_json::json!({"cache_hit": true, "cache_key": cache_key})),
                cache_hit: true,
                execution_time_ms: execution_time,
            });
        }
        
        // Cache miss - make actual API call
        let response = {
            let mut client = self.client.lock().await;
            client.call_api(endpoint, params).await?
        };
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Cache the response
        let response_json = serde_json::to_string(&response)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize API response: {}", e)))?;
        
        if let Err(e) = self.api_cache.set(cache_key.clone(), response_json.clone(), request_hash, None).await {
            eprintln!("Failed to cache API response: {}", e);
            self.update_stats(|stats| stats.cache_errors += 1).await;
        }
        
        self.update_stats(|stats| {
            stats.total_requests += 1;
            stats.api_requests += 1;
            stats.cache_misses += 1;
        }).await;
        
        Ok(CachedMcpResponse {
            response_data: response_json,
            metadata: Some(serde_json::json!({"cache_hit": false, "cache_key": cache_key})),
            cache_hit: false,
            execution_time_ms: execution_time,
        })
    }
    
    /// Execute tool call with caching
    pub async fn cached_tool_call(
        &self,
        tool_name: &str,
        args: Option<serde_json::Value>,
    ) -> Result<CachedMcpResponse, UltraFastAiError> {
        let request = CachedMcpRequest {
            request_type: "tool".to_string(),
            request_data: tool_name.to_string(),
            parameters: args.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        let cache_key = self.tool_cache.generate_key("tool", &format!("{}:{:?}", tool_name, args));
        let request_hash = self.generate_request_hash(&request)?;
        
        let start_time = Instant::now();
        
        // Try cache first
        if let Ok(Some(cached_data)) = self.tool_cache.get(&cache_key).await {
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            self.update_stats(|stats| {
                stats.total_requests += 1;
                stats.tool_requests += 1;
                stats.cache_hits += 1;
                stats.total_time_saved_ms += execution_time;
            }).await;
            
            return Ok(CachedMcpResponse {
                response_data: cached_data,
                metadata: Some(serde_json::json!({"cache_hit": true, "cache_key": cache_key})),
                cache_hit: true,
                execution_time_ms: execution_time,
            });
        }
        
        // Cache miss - make actual tool call
        let response = {
            let mut client = self.client.lock().await;
            client.call_tool(tool_name, args).await?
        };
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Cache the response (with shorter TTL for tools)
        let response_json = serde_json::to_string(&response)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize tool response: {}", e)))?;
        
        if let Err(e) = self.tool_cache.set(cache_key.clone(), response_json.clone(), request_hash, Some(12)).await {
            eprintln!("Failed to cache tool response: {}", e);
            self.update_stats(|stats| stats.cache_errors += 1).await;
        }
        
        self.update_stats(|stats| {
            stats.total_requests += 1;
            stats.tool_requests += 1;
            stats.cache_misses += 1;
        }).await;
        
        Ok(CachedMcpResponse {
            response_data: response_json,
            metadata: Some(serde_json::json!({"cache_hit": false, "cache_key": cache_key})),
            cache_hit: false,
            execution_time_ms: execution_time,
        })
    }
    
    /// Fetch data with caching
    pub async fn cached_data_fetch(
        &self,
        data_source: &str,
        query: Option<serde_json::Value>,
    ) -> Result<CachedMcpResponse, UltraFastAiError> {
        let request = CachedMcpRequest {
            request_type: "data".to_string(),
            request_data: data_source.to_string(),
            parameters: query.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        let cache_key = self.data_cache.generate_key("data", &format!("{}:{:?}", data_source, query));
        let request_hash = self.generate_request_hash(&request)?;
        
        let start_time = Instant::now();
        
        // Try cache first
        if let Ok(Some(cached_data)) = self.data_cache.get(&cache_key).await {
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            self.update_stats(|stats| {
                stats.total_requests += 1;
                stats.data_requests += 1;
                stats.cache_hits += 1;
                stats.total_time_saved_ms += execution_time;
            }).await;
            
            return Ok(CachedMcpResponse {
                response_data: cached_data,
                metadata: Some(serde_json::json!({"cache_hit": true, "cache_key": cache_key})),
                cache_hit: true,
                execution_time_ms: execution_time,
            });
        }
        
        // Cache miss - fetch actual data
        let response = {
            let mut client = self.client.lock().await;
            client.fetch_data(data_source, query).await?
        };
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Cache the response (with longer TTL for data)
        let response_json = serde_json::to_string(&response)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize data response: {}", e)))?;
        
        if let Err(e) = self.data_cache.set(cache_key.clone(), response_json.clone(), request_hash, Some(48)).await {
            eprintln!("Failed to cache data response: {}", e);
            self.update_stats(|stats| stats.cache_errors += 1).await;
        }
        
        self.update_stats(|stats| {
            stats.total_requests += 1;
            stats.data_requests += 1;
            stats.cache_misses += 1;
        }).await;
        
        Ok(CachedMcpResponse {
            response_data: response_json,
            metadata: Some(serde_json::json!({"cache_hit": false, "cache_key": cache_key})),
            cache_hit: false,
            execution_time_ms: execution_time,
        })
    }
    
    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheManagerStats {
        self.cache_stats.read().await.clone()
    }
    
    /// Get detailed cache information
    pub async fn get_cache_info(&self) -> Result<serde_json::Value, UltraFastAiError> {
        let api_stats = self.api_cache.get_stats().await;
        let tool_stats = self.tool_cache.get_stats().await;
        let data_stats = self.data_cache.get_stats().await;
        let manager_stats = self.get_cache_stats().await;
        
        let api_hit_ratio = self.api_cache.hit_ratio().await;
        let tool_hit_ratio = self.tool_cache.hit_ratio().await;
        let data_hit_ratio = self.data_cache.hit_ratio().await;
        
        Ok(serde_json::json!({
            "manager_stats": manager_stats,
            "cache_hit_ratios": {
                "api": api_hit_ratio,
                "tool": tool_hit_ratio,
                "data": data_hit_ratio,
                "overall": if manager_stats.total_requests > 0 {
                    manager_stats.cache_hits as f64 / manager_stats.total_requests as f64
                } else { 0.0 }
            },
            "cache_details": {
                "api": api_stats,
                "tool": tool_stats,
                "data": data_stats
            }
        }))
    }
    
    /// Cleanup expired cache entries
    pub async fn cleanup_caches(&self) -> Result<serde_json::Value, UltraFastAiError> {
        let api_cleaned = self.api_cache.cleanup_expired().await?;
        let tool_cleaned = self.tool_cache.cleanup_expired().await?;
        let data_cleaned = self.data_cache.cleanup_expired().await?;
        
        Ok(serde_json::json!({
            "cleaned_entries": {
                "api": api_cleaned,
                "tool": tool_cleaned,
                "data": data_cleaned,
                "total": api_cleaned + tool_cleaned + data_cleaned
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    }
    
    /// Clear all caches
    pub async fn clear_all_caches(&self) -> Result<(), UltraFastAiError> {
        self.api_cache.clear().await?;
        self.tool_cache.clear().await?;
        self.data_cache.clear().await?;
        
        // Reset stats
        let mut stats = self.cache_stats.write().await;
        *stats = CacheManagerStats::default();
        
        Ok(())
    }
    
    /// Invalidate specific cache entries by pattern
    pub async fn invalidate_cache_pattern(&self, pattern: &str) -> Result<usize, UltraFastAiError> {
        let mut total_removed = 0;
        
        // Check all cache types for matching keys
        for cache in [&self.api_cache, &self.tool_cache, &self.data_cache] {
            let keys = cache.list_keys().await;
            for key in keys {
                if key.contains(pattern) {
                    if cache.remove(&key).await? {
                        total_removed += 1;
                    }
                }
            }
        }
        
        Ok(total_removed)
    }
    
    /// Preload cache with common requests
    pub async fn preload_cache(&self, requests: Vec<CachedMcpRequest>) -> Result<usize, UltraFastAiError> {
        let mut preloaded = 0;
        
        for request in requests {
            match request.request_type.as_str() {
                "api" => {
                    if let Ok(_) = self.cached_api_call(&request.request_data, request.parameters).await {
                        preloaded += 1;
                    }
                }
                "tool" => {
                    if let Ok(_) = self.cached_tool_call(&request.request_data, request.parameters).await {
                        preloaded += 1;
                    }
                }
                "data" => {
                    if let Ok(_) = self.cached_data_fetch(&request.request_data, request.parameters).await {
                        preloaded += 1;
                    }
                }
                _ => {}
            }
        }
        
        Ok(preloaded)
    }
    
    // Private helper methods
    
    fn generate_request_hash(&self, request: &CachedMcpRequest) -> Result<String, UltraFastAiError> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize request: {}", e)))?;
        
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(request_json.as_bytes());
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }
    
    async fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut CacheManagerStats),
    {
        let mut stats = self.cache_stats.write().await;
        updater(&mut *stats);
    }
}

/// Cache warming utility
pub struct CacheWarmer {
    client: Arc<CachedMcpClient>,
}

impl CacheWarmer {
    pub fn new(client: Arc<CachedMcpClient>) -> Self {
        Self { client }
    }
    
    /// Warm cache with common API endpoints
    pub async fn warm_api_cache(&self) -> Result<usize, UltraFastAiError> {
        let common_endpoints = vec![
            ("health", None),
            ("version", None),
            ("capabilities", None),
        ];
        
        let mut warmed = 0;
        for (endpoint, params) in common_endpoints {
            if let Ok(_) = self.client.cached_api_call(endpoint, params).await {
                warmed += 1;
            }
        }
        
        Ok(warmed)
    }
    
    /// Warm cache with common tools
    pub async fn warm_tool_cache(&self) -> Result<usize, UltraFastAiError> {
        let common_tools = vec![
            ("compiler", Some(serde_json::json!({"language": "rust"}))),
            ("linter", Some(serde_json::json!({"language": "rust"}))),
            ("formatter", Some(serde_json::json!({"language": "rust"}))),
        ];
        
        let mut warmed = 0;
        for (tool, args) in common_tools {
            if let Ok(_) = self.client.cached_tool_call(tool, args).await {
                warmed += 1;
            }
        }
        
        Ok(warmed)
    }
}