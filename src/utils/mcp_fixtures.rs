//! MCP Fixture System
//!
//! Provides reproducible MCP responses for testing and development.
//! Fixtures are organized by date in __fixtures__/mcp/YYYY-MM-DD/ directories
//! to enable deterministic testing and response replay functionality.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tokio::fs;
use chrono::{DateTime, Utc, NaiveDate};

use crate::UltraFastAiError;

/// MCP fixture entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpFixture {
    /// Unique identifier for this fixture
    pub id: String,
    /// Type of MCP operation (api, tool, data)
    pub operation_type: String,
    /// Target endpoint/tool/resource
    pub target: String,
    /// Request parameters as JSON
    pub request: serde_json::Value,
    /// Response data as JSON
    pub response: serde_json::Value,
    /// Response metadata
    pub metadata: FixtureMetadata,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Fixture metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureMetadata {
    /// Timestamp when fixture was created
    pub created_at: String,
    /// Author/source of the fixture
    pub created_by: String,
    /// Description of the fixture
    pub description: String,
    /// Expected response time in milliseconds
    pub expected_response_time_ms: Option<u64>,
    /// HTTP status code (for API calls)
    pub status_code: Option<u16>,
    /// Whether this fixture represents an error case
    pub is_error: bool,
    /// Fixture version for compatibility
    pub version: String,
}

/// Fixture collection for a specific date
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureCollection {
    pub date: String,
    pub fixtures: Vec<McpFixture>,
    pub metadata: CollectionMetadata,
}

/// Collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub total_fixtures: usize,
    pub api_fixtures: usize,
    pub tool_fixtures: usize,
    pub data_fixtures: usize,
    pub error_fixtures: usize,
    pub created_at: String,
    pub description: String,
}

/// Fixture manager for loading and managing MCP fixtures
pub struct McpFixtureManager {
    base_path: PathBuf,
    fixtures: HashMap<String, FixtureCollection>,
    fixture_index: HashMap<String, String>, // fixture_id -> date
}

impl McpFixtureManager {
    /// Create a new fixture manager
    pub fn new(base_path: Option<PathBuf>) -> Self {
        let base_path = base_path.unwrap_or_else(|| PathBuf::from("__fixtures__/mcp"));
        
        Self {
            base_path,
            fixtures: HashMap::new(),
            fixture_index: HashMap::new(),
        }
    }
    
    /// Initialize fixture directories
    pub async fn initialize(&self) -> Result<(), UltraFastAiError> {
        fs::create_dir_all(&self.base_path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to create fixtures directory: {}", e)))?;
        
        // Create today's fixture directory
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let today_path = self.base_path.join(&today);
        fs::create_dir_all(&today_path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to create today's fixture directory: {}", e)))?;
        
        Ok(())
    }
    
    /// Load all fixtures from disk
    pub async fn load_fixtures(&mut self) -> Result<(), UltraFastAiError> {
        if !self.base_path.exists() {
            return Ok(()); // No fixtures to load
        }
        
        let mut entries = fs::read_dir(&self.base_path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read fixtures directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.is_dir() {
                if let Some(date_str) = path.file_name().and_then(|s| s.to_str()) {
                    if self.is_valid_date_format(date_str) {
                        match self.load_collection_from_date(date_str).await {
                            Ok(collection) => {
                                // Update fixture index
                                for fixture in &collection.fixtures {
                                    self.fixture_index.insert(fixture.id.clone(), date_str.to_string());
                                }
                                self.fixtures.insert(date_str.to_string(), collection);
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to load fixtures for {}: {}", date_str, e);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Load fixture collection for a specific date
    pub async fn load_collection_from_date(&self, date: &str) -> Result<FixtureCollection, UltraFastAiError> {
        let date_path = self.base_path.join(date);
        if !date_path.exists() {
            return Err(UltraFastAiError::from(format!("Fixture directory for {} does not exist", date)));
        }
        
        let mut fixtures = Vec::new();
        let mut api_count = 0;
        let mut tool_count = 0;
        let mut data_count = 0;
        let mut error_count = 0;
        
        let mut entries = fs::read_dir(&date_path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read date directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read fixture file: {}", e)))? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_fixture_from_file(&path).await {
                    Ok(fixture) => {
                        match fixture.operation_type.as_str() {
                            "api" => api_count += 1,
                            "tool" => tool_count += 1,
                            "data" => data_count += 1,
                            _ => {}
                        }
                        
                        if fixture.metadata.is_error {
                            error_count += 1;
                        }
                        
                        fixtures.push(fixture);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load fixture from {}: {}", path.display(), e);
                    }
                }
            }
        }
        
        let metadata = CollectionMetadata {
            total_fixtures: fixtures.len(),
            api_fixtures: api_count,
            tool_fixtures: tool_count,
            data_fixtures: data_count,
            error_fixtures: error_count,
            created_at: Utc::now().to_rfc3339(),
            description: format!("MCP fixtures for {}", date),
        };
        
        Ok(FixtureCollection {
            date: date.to_string(),
            fixtures,
            metadata,
        })
    }
    
    /// Get fixture by ID
    pub async fn get_fixture(&mut self, fixture_id: &str) -> Result<Option<McpFixture>, UltraFastAiError> {
        // Ensure fixtures are loaded
        if self.fixtures.is_empty() {
            self.load_fixtures().await?;
        }
        
        if let Some(date) = self.fixture_index.get(fixture_id) {
            if let Some(collection) = self.fixtures.get(date) {
                return Ok(collection.fixtures.iter()
                    .find(|f| f.id == fixture_id)
                    .cloned());
            }
        }
        
        Ok(None)
    }
    
    /// Find fixtures by operation type and target
    pub async fn find_fixtures(
        &mut self,
        operation_type: &str,
        target: &str,
    ) -> Result<Vec<McpFixture>, UltraFastAiError> {
        if self.fixtures.is_empty() {
            self.load_fixtures().await?;
        }
        
        let mut matching_fixtures = Vec::new();
        
        for collection in self.fixtures.values() {
            for fixture in &collection.fixtures {
                if fixture.operation_type == operation_type && fixture.target == target {
                    matching_fixtures.push(fixture.clone());
                }
            }
        }
        
        Ok(matching_fixtures)
    }
    
    /// Create a new fixture
    pub async fn create_fixture(
        &mut self,
        operation_type: String,
        target: String,
        request: serde_json::Value,
        response: serde_json::Value,
        description: String,
        tags: Vec<String>,
    ) -> Result<String, UltraFastAiError> {
        let fixture_id = self.generate_fixture_id(&operation_type, &target);
        
        let metadata = FixtureMetadata {
            created_at: Utc::now().to_rfc3339(),
            created_by: "ultra-fast-ai".to_string(),
            description,
            expected_response_time_ms: None,
            status_code: None,
            is_error: false,
            version: "1.0".to_string(),
        };
        
        let fixture = McpFixture {
            id: fixture_id.clone(),
            operation_type,
            target,
            request,
            response,
            metadata,
            tags,
        };
        
        // Save to today's directory
        let today = Utc::now().format("%Y-%m-%d").to_string();
        self.save_fixture(&fixture, &today).await?;
        
        // Update in-memory collections
        self.fixture_index.insert(fixture_id.clone(), today.clone());
        
        if let Some(collection) = self.fixtures.get_mut(&today) {
            collection.fixtures.push(fixture);
            collection.metadata.total_fixtures += 1;
        } else {
            // Create new collection for today
            let collection = FixtureCollection {
                date: today.clone(),
                fixtures: vec![fixture],
                metadata: CollectionMetadata {
                    total_fixtures: 1,
                    api_fixtures: 0,
                    tool_fixtures: 0,
                    data_fixtures: 0,
                    error_fixtures: 0,
                    created_at: Utc::now().to_rfc3339(),
                    description: format!("MCP fixtures for {}", today),
                },
            };
            self.fixtures.insert(today, collection);
        }
        
        Ok(fixture_id)
    }
    
    /// Save fixture to disk
    async fn save_fixture(&self, fixture: &McpFixture, date: &str) -> Result<(), UltraFastAiError> {
        let date_dir = self.base_path.join(date);
        fs::create_dir_all(&date_dir).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to create date directory: {}", e)))?;
        
        let file_path = date_dir.join(format!("{}.json", fixture.id));
        let json_content = serde_json::to_string_pretty(fixture)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize fixture: {}", e)))?;
        
        fs::write(&file_path, json_content).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to write fixture file: {}", e)))?;
        
        Ok(())
    }
    
    /// Load fixture from file
    async fn load_fixture_from_file(&self, path: &Path) -> Result<McpFixture, UltraFastAiError> {
        let content = fs::read_to_string(path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read fixture file: {}", e)))?;
        
        let fixture: McpFixture = serde_json::from_str(&content)
            .map_err(|e| UltraFastAiError::from(format!("Failed to parse fixture: {}", e)))?;
        
        Ok(fixture)
    }
    
    /// Generate unique fixture ID
    fn generate_fixture_id(&self, operation_type: &str, target: &str) -> String {
        use sha2::{Sha256, Digest};
        
        let input = format!("{}:{}:{}", operation_type, target, Utc::now().timestamp_nanos());
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        let hash = hasher.finalize();
        
        format!("{}_{:x}", operation_type, &hash[..8].iter().map(|b| format!("{:02x}", b)).collect::<String>())
    }
    
    /// Check if string is valid date format (YYYY-MM-DD)
    fn is_valid_date_format(&self, date_str: &str) -> bool {
        NaiveDate::parse_from_str(date_str, "%Y-%m-%d").is_ok()
    }
    
    /// Get available fixture dates
    pub async fn get_available_dates(&self) -> Result<Vec<String>, UltraFastAiError> {
        if !self.base_path.exists() {
            return Ok(Vec::new());
        }
        
        let mut dates = Vec::new();
        let mut entries = fs::read_dir(&self.base_path).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read fixtures directory: {}", e)))?;
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| UltraFastAiError::from(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.is_dir() {
                if let Some(date_str) = path.file_name().and_then(|s| s.to_str()) {
                    if self.is_valid_date_format(date_str) {
                        dates.push(date_str.to_string());
                    }
                }
            }
        }
        
        dates.sort();
        Ok(dates)
    }
    
    /// Get fixture statistics
    pub async fn get_stats(&mut self) -> Result<FixtureStats, UltraFastAiError> {
        if self.fixtures.is_empty() {
            self.load_fixtures().await?;
        }
        
        let mut stats = FixtureStats::default();
        
        for collection in self.fixtures.values() {
            stats.total_fixtures += collection.metadata.total_fixtures;
            stats.api_fixtures += collection.metadata.api_fixtures;
            stats.tool_fixtures += collection.metadata.tool_fixtures;
            stats.data_fixtures += collection.metadata.data_fixtures;
            stats.error_fixtures += collection.metadata.error_fixtures;
            stats.total_dates += 1;
        }
        
        Ok(stats)
    }
    
    /// Export fixtures for a date range
    pub async fn export_fixtures(
        &mut self,
        start_date: &str,
        end_date: &str,
        output_path: &Path,
    ) -> Result<(), UltraFastAiError> {
        if self.fixtures.is_empty() {
            self.load_fixtures().await?;
        }
        
        let mut export_fixtures = Vec::new();
        
        for (date, collection) in &self.fixtures {
            if date >= start_date && date <= end_date {
                export_fixtures.extend(collection.fixtures.clone());
            }
        }
        
        let export_data = serde_json::json!({
            "export_date": Utc::now().to_rfc3339(),
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "total_fixtures": export_fixtures.len(),
            "fixtures": export_fixtures
        });
        
        let json_content = serde_json::to_string_pretty(&export_data)
            .map_err(|e| UltraFastAiError::from(format!("Failed to serialize export data: {}", e)))?;
        
        fs::write(output_path, json_content).await
            .map_err(|e| UltraFastAiError::from(format!("Failed to write export file: {}", e)))?;
        
        Ok(())
    }
    
    /// Clean up old fixtures
    pub async fn cleanup_old_fixtures(&mut self, days_to_keep: u32) -> Result<usize, UltraFastAiError> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_to_keep as i64);
        let cutoff_str = cutoff_date.format("%Y-%m-%d").to_string();
        
        let mut removed_count = 0;
        let dates_to_remove: Vec<String> = self.fixtures.keys()
            .filter(|&date| date < &cutoff_str)
            .cloned()
            .collect();
        
        for date in dates_to_remove {
            let date_path = self.base_path.join(&date);
            if date_path.exists() {
                match fs::remove_dir_all(&date_path).await {
                    Ok(_) => {
                        if let Some(collection) = self.fixtures.remove(&date) {
                            removed_count += collection.metadata.total_fixtures;
                            
                            // Remove from fixture index
                            for fixture in &collection.fixtures {
                                self.fixture_index.remove(&fixture.id);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to remove old fixtures for {}: {}", date, e);
                    }
                }
            }
        }
        
        Ok(removed_count)
    }
}

/// Fixture statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureStats {
    pub total_fixtures: usize,
    pub api_fixtures: usize,
    pub tool_fixtures: usize,
    pub data_fixtures: usize,
    pub error_fixtures: usize,
    pub total_dates: usize,
}

impl Default for FixtureStats {
    fn default() -> Self {
        Self {
            total_fixtures: 0,
            api_fixtures: 0,
            tool_fixtures: 0,
            data_fixtures: 0,
            error_fixtures: 0,
            total_dates: 0,
        }
    }
}

/// Builder for creating fixtures
pub struct FixtureBuilder {
    operation_type: String,
    target: String,
    request: serde_json::Value,
    response: serde_json::Value,
    description: String,
    tags: Vec<String>,
    expected_response_time_ms: Option<u64>,
    status_code: Option<u16>,
    is_error: bool,
}

impl FixtureBuilder {
    pub fn new(operation_type: String, target: String) -> Self {
        Self {
            operation_type,
            target,
            request: serde_json::Value::Null,
            response: serde_json::Value::Null,
            description: String::new(),
            tags: Vec::new(),
            expected_response_time_ms: None,
            status_code: None,
            is_error: false,
        }
    }
    
    pub fn request(mut self, request: serde_json::Value) -> Self {
        self.request = request;
        self
    }
    
    pub fn response(mut self, response: serde_json::Value) -> Self {
        self.response = response;
        self
    }
    
    pub fn description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
    
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn expected_time(mut self, ms: u64) -> Self {
        self.expected_response_time_ms = Some(ms);
        self
    }
    
    pub fn status_code(mut self, code: u16) -> Self {
        self.status_code = Some(code);
        self
    }
    
    pub fn error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }
    
    pub async fn build(self, manager: &mut McpFixtureManager) -> Result<String, UltraFastAiError> {
        manager.create_fixture(
            self.operation_type,
            self.target,
            self.request,
            self.response,
            self.description,
            self.tags,
        ).await
    }
}