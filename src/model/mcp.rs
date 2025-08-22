//! Model Context Protocol (MCP) client implementation
//!
//! Provides interface for external tool/data delegation with caching,
//! error handling, and fallback mechanisms.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use crate::Result;

/// MCP server types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum McpServerType {
    Api,        // External API calls
    Tools,      // Coding tools integration
    Data,       // Dataset access
    Feedback,   // Self-improvement loops
}

/// MCP request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub server_type: McpServerType,
    pub method: String,
    pub params: serde_json::Value,
    pub timeout_ms: u64,
    pub cache_ttl_hours: u32,
    pub retry_count: u32,
}

/// MCP response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub success: bool,
    pub data: serde_json::Value,
    pub error: Option<String>,
    pub cached: bool,
    pub timestamp: u64,
    pub execution_time_ms: u64,
}

/// Cached MCP response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResponse {
    response: McpResponse,
    expires_at: u64,
}

/// MCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub api_base_url: String,
    pub tools_base_url: String,
    pub data_base_url: String,
    pub feedback_base_url: String,
    pub default_timeout_ms: u64,
    pub default_cache_ttl_hours: u32,
    pub max_retry_count: u32,
    pub enable_cache: bool,
    pub cache_directory: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            api_base_url: "http://localhost:8001".to_string(),
            tools_base_url: "http://localhost:8002".to_string(),
            data_base_url: "http://localhost:8003".to_string(),
            feedback_base_url: "http://localhost:8004".to_string(),
            default_timeout_ms: 5000,
            default_cache_ttl_hours: 24,
            max_retry_count: 3,
            enable_cache: true,
            cache_directory: ".cache/mcp".to_string(),
        }
    }
}

/// MCP client for external service integration
pub struct McpClient {
    config: McpConfig,
    cache: HashMap<String, CachedResponse>,
    #[cfg(feature = "mcp-integration")]
    http_client: reqwest::Client,
}

impl McpClient {
    /// Create new MCP client
    pub fn new(config: McpConfig) -> Self {
        #[cfg(feature = "mcp-integration")]
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.default_timeout_ms))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            config,
            cache: HashMap::new(),
            #[cfg(feature = "mcp-integration")]
            http_client,
        }
    }

    /// Execute MCP request with caching and retries
    pub async fn execute(&mut self, request: McpRequest) -> Result<McpResponse> {
        let start_time = SystemTime::now();
        
        // Generate cache key
        let cache_key = self.generate_cache_key(&request);
        
        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.get_cached_response(&cache_key) {
                log::debug!("MCP cache hit for key: {}", cache_key);
                return Ok(cached);
            }
        }
        
        // Execute request with retries
        let mut last_error = None;
        for attempt in 0..=request.retry_count.min(self.config.max_retry_count) {
            match self.execute_request(&request).await {
                Ok(mut response) => {
                    // Calculate execution time
                    let execution_time = start_time.elapsed()
                        .unwrap_or(Duration::from_secs(0))
                        .as_millis() as u64;
                    response.execution_time_ms = execution_time;
                    response.cached = false;
                    
                    // Cache successful response
                    if self.config.enable_cache && response.success {
                        self.cache_response(&cache_key, &response, request.cache_ttl_hours);
                    }
                    
                    return Ok(response);
                },
                Err(e) => {
                    last_error = Some(e);
                    if attempt < request.retry_count {
                        log::warn!("MCP request attempt {} failed, retrying...", attempt + 1);
                        tokio::time::sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                    }
                }
            }
        }
        
        // All retries failed - return error response
        let error_msg = format!("MCP request failed after {} retries: {:?}", 
                               request.retry_count, last_error);
        Ok(McpResponse {
            success: false,
            data: serde_json::Value::Null,
            error: Some(error_msg),
            cached: false,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0)).as_secs(),
            execution_time_ms: start_time.elapsed()
                .unwrap_or(Duration::from_secs(0)).as_millis() as u64,
        })
    }

    /// Execute single MCP request
    async fn execute_request(&self, request: &McpRequest) -> Result<McpResponse> {
        let url = self.get_server_url(&request.server_type);
        let endpoint = format!("{}/{}", url, request.method);
        
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0)).as_secs();
        
        // Check if MCP integration is enabled
        #[cfg(feature = "mcp-integration")]
        {
            let response = timeout(
                Duration::from_millis(request.timeout_ms),
                self.http_client.post(&endpoint)
                    .json(&request.params)
                    .send()
            ).await.map_err(|_| crate::UltraFastAiError::McpError("Request timeout".to_string()))?
             .map_err(|e| crate::UltraFastAiError::McpError(e.to_string()))?;

            if response.status().is_success() {
                let data: serde_json::Value = response.json().await
                    .map_err(|e| crate::UltraFastAiError::McpError(e.to_string()))?;
                
                Ok(McpResponse {
                    success: true,
                    data,
                    error: None,
                    cached: false,
                    timestamp,
                    execution_time_ms: 0, // Will be set by caller
                })
            } else {
                Ok(McpResponse {
                    success: false,
                    data: serde_json::Value::Null,
                    error: Some(format!("HTTP {}: {}", response.status(), 
                                      response.text().await.unwrap_or_default())),
                    cached: false,
                    timestamp,
                    execution_time_ms: 0,
                })
            }
        }
        
        #[cfg(not(feature = "mcp-integration"))]
        {
            // Offline mode - return mock response
            log::warn!("MCP integration disabled, returning mock response for: {}", endpoint);
            Ok(McpResponse {
                success: true,
                data: serde_json::json!({
                    "mock": true,
                    "method": request.method,
                    "server_type": request.server_type,
                    "message": "MCP integration disabled, mock response returned"
                }),
                error: None,
                cached: false,
                timestamp,
                execution_time_ms: 0,
            })
        }
    }

    /// Get server URL for server type
    fn get_server_url(&self, server_type: &McpServerType) -> &str {
        match server_type {
            McpServerType::Api => &self.config.api_base_url,
            McpServerType::Tools => &self.config.tools_base_url,
            McpServerType::Data => &self.config.data_base_url,
            McpServerType::Feedback => &self.config.feedback_base_url,
        }
    }

    /// Generate cache key for request
    fn generate_cache_key(&self, request: &McpRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        request.server_type.hash(&mut hasher);
        request.method.hash(&mut hasher);
        request.params.to_string().hash(&mut hasher);
        
        format!("mcp_{}_{:x}", 
                request.server_type.to_string().to_lowercase(), 
                hasher.finish())
    }

    /// Get cached response if valid
    fn get_cached_response(&self, cache_key: &str) -> Option<McpResponse> {
        if let Some(cached) = self.cache.get(cache_key) {
            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0)).as_secs();
            
            if now < cached.expires_at {
                let mut response = cached.response.clone();
                response.cached = true;
                return Some(response);
            }
        }
        None
    }

    /// Cache response
    fn cache_response(&mut self, cache_key: &str, response: &McpResponse, ttl_hours: u32) {
        let expires_at = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0)).as_secs() + (ttl_hours as u64 * 3600);
        
        self.cache.insert(cache_key.to_string(), CachedResponse {
            response: response.clone(),
            expires_at,
        });

        // Persist cache to disk if enabled
        if let Err(e) = self.persist_cache() {
            log::warn!("Failed to persist MCP cache: {}", e);
        }
    }

    /// Clear expired cache entries
    pub fn cleanup_cache(&mut self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0)).as_secs();
        
        self.cache.retain(|_, cached| now < cached.expires_at);
    }

    /// Persist cache to disk
    fn persist_cache(&self) -> Result<()> {
        if !self.config.enable_cache {
            return Ok(());
        }

        let cache_dir = std::path::Path::new(&self.config.cache_directory);
        std::fs::create_dir_all(cache_dir)?;

        let cache_file = cache_dir.join("mcp_cache.json");
        let cache_json = serde_json::to_string_pretty(&self.cache)?;
        std::fs::write(cache_file, cache_json)?;

        Ok(())
    }

    /// Load cache from disk
    pub fn load_cache(&mut self) -> Result<()> {
        if !self.config.enable_cache {
            return Ok(());
        }

        let cache_file = std::path::Path::new(&self.config.cache_directory)
            .join("mcp_cache.json");
        
        if cache_file.exists() {
            let cache_json = std::fs::read_to_string(cache_file)?;
            self.cache = serde_json::from_str(&cache_json)?;
            self.cleanup_cache(); // Remove expired entries
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0)).as_secs();
        
        let total_entries = self.cache.len() as u64;
        let expired_entries = self.cache.values()
            .filter(|cached| now >= cached.expires_at)
            .count() as u64;
        let active_entries = total_entries - expired_entries;
        
        stats.insert("total_entries".to_string(), total_entries);
        stats.insert("active_entries".to_string(), active_entries);
        stats.insert("expired_entries".to_string(), expired_entries);
        
        stats
    }
}

impl std::fmt::Display for McpServerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpServerType::Api => write!(f, "Api"),
            McpServerType::Tools => write!(f, "Tools"),
            McpServerType::Data => write!(f, "Data"),
            McpServerType::Feedback => write!(f, "Feedback"),
        }
    }
}

/// Helper functions for common MCP operations
impl McpClient {
    /// Fetch dataset information
    pub async fn fetch_dataset(&mut self, dataset_name: &str) -> Result<McpResponse> {
        let request = McpRequest {
            server_type: McpServerType::Data,
            method: "get_dataset".to_string(),
            params: serde_json::json!({
                "name": dataset_name,
                "format": "json"
            }),
            timeout_ms: self.config.default_timeout_ms,
            cache_ttl_hours: self.config.default_cache_ttl_hours,
            retry_count: 2,
        };
        
        self.execute(request).await
    }

    /// Run code analysis tool
    pub async fn analyze_code(&mut self, code: &str, language: &str) -> Result<McpResponse> {
        let request = McpRequest {
            server_type: McpServerType::Tools,
            method: "analyze".to_string(),
            params: serde_json::json!({
                "code": code,
                "language": language,
                "checks": ["syntax", "style", "security"]
            }),
            timeout_ms: self.config.default_timeout_ms * 2, // Longer timeout for analysis
            cache_ttl_hours: 1, // Short cache for code analysis
            retry_count: 1,
        };
        
        self.execute(request).await
    }

    /// Submit feedback for model improvement
    pub async fn submit_feedback(&mut self, feedback_data: serde_json::Value) -> Result<McpResponse> {
        let request = McpRequest {
            server_type: McpServerType::Feedback,
            method: "submit".to_string(),
            params: feedback_data,
            timeout_ms: self.config.default_timeout_ms,
            cache_ttl_hours: 0, // No caching for feedback
            retry_count: 3, // Important to retry feedback
        };
        
        self.execute(request).await
    }

    /// Query external API
    pub async fn api_query(&mut self, endpoint: &str, params: serde_json::Value) -> Result<McpResponse> {
        let request = McpRequest {
            server_type: McpServerType::Api,
            method: endpoint.to_string(),
            params,
            timeout_ms: self.config.default_timeout_ms,
            cache_ttl_hours: self.config.default_cache_ttl_hours,
            retry_count: 2,
        };
        
        self.execute(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mcp_config_default() {
        let config = McpConfig::default();
        assert_eq!(config.default_timeout_ms, 5000);
        assert_eq!(config.default_cache_ttl_hours, 24);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_cache_key_generation() {
        let config = McpConfig::default();
        let client = McpClient::new(config);
        
        let request = McpRequest {
            server_type: McpServerType::Data,
            method: "test".to_string(),
            params: serde_json::json!({"key": "value"}),
            timeout_ms: 1000,
            cache_ttl_hours: 1,
            retry_count: 1,
        };
        
        let key1 = client.generate_cache_key(&request);
        let key2 = client.generate_cache_key(&request);
        
        assert_eq!(key1, key2); // Same request should generate same key
        assert!(key1.starts_with("mcp_data_"));
    }

    #[test]
    fn test_cache_operations() {
        let config = McpConfig::default();
        let mut client = McpClient::new(config);
        
        let response = McpResponse {
            success: true,
            data: serde_json::json!({"test": "data"}),
            error: None,
            cached: false,
            timestamp: 1234567890,
            execution_time_ms: 100,
        };
        
        client.cache_response("test_key", &response, 1);
        
        let cached = client.get_cached_response("test_key");
        assert!(cached.is_some());
        assert!(cached.unwrap().cached);
    }

    #[tokio::test]
    async fn test_offline_mode() {
        let config = McpConfig::default();
        let mut client = McpClient::new(config);
        
        let request = McpRequest {
            server_type: McpServerType::Data,
            method: "test".to_string(),
            params: serde_json::json!({}),
            timeout_ms: 1000,
            cache_ttl_hours: 1,
            retry_count: 1,
        };
        
        let response = client.execute(request).await.unwrap();
        
        // In offline mode (without mcp-integration feature), should return mock response
        assert!(response.success);
        assert!(response.data.get("mock").is_some());
    }
    
    #[tokio::test]
    async fn test_helper_methods() {
        let config = McpConfig::default();
        let mut client = McpClient::new(config);
        
        // Test dataset fetch
        let dataset_response = client.fetch_dataset("test_dataset").await.unwrap();
        assert!(dataset_response.success);
        
        // Test code analysis
        let code_response = client.analyze_code("fn main() {}", "rust").await.unwrap();
        assert!(code_response.success);
        
        // Test API query
        let api_response = client.api_query("test", serde_json::json!({})).await.unwrap();
        assert!(api_response.success);
    }
}