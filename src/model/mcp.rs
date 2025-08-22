//! MCP (Model Context Protocol) integration

use serde::{Deserialize, Serialize};

/// MCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub api_server: String,
    pub tools_server: String,
    pub data_server: String,
    pub feedback_server: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            api_server: "http://localhost:8001".to_string(),
            tools_server: "http://localhost:8002".to_string(),
            data_server: "http://localhost:8003".to_string(),
            feedback_server: "http://localhost:8004".to_string(),
        }
    }
}

/// MCP client
pub struct McpClient {
    // TODO: Implement MCP client
}