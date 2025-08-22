package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func createTestDataServer() *DataServer {
	return NewDataServer(8001)
}

func createTestDataRequest() DataRequest {
	return DataRequest{
		Dataset:     "HumanEval",
		Query:       "python functions",
		Format:      "json",
		Limit:       100,
		Offset:      0,
		Filters:     map[string]interface{}{"language": "python"},
		Aggregation: map[string]interface{}{"group_by": "difficulty"},
		Timestamp:   time.Now().Unix(),
	}
}

func TestDataServerCreation(t *testing.T) {
	server := createTestDataServer()
	
	if server.port != 8001 {
		t.Errorf("Expected port 8001, got %d", server.port)
	}
	
	if server.cache == nil {
		t.Error("Data cache should be initialized")
	}
	
	if server.providers == nil {
		t.Error("Data providers should be initialized")
	}
	
	if server.queryProcessor == nil {
		t.Error("Query processor should be initialized")
	}
}

func TestDataRequestValidation(t *testing.T) {
	tests := []struct {
		name    string
		request DataRequest
		valid   bool
	}{
		{
			name:    "Valid data request",
			request: createTestDataRequest(),
			valid:   true,
		},
		{
			name: "Missing dataset",
			request: DataRequest{
				Query:  "test query",
				Format: "json",
			},
			valid: false,
		},
		{
			name: "Invalid format",
			request: DataRequest{
				Dataset: "HumanEval",
				Query:   "test query",
				Format:  "invalid_format",
			},
			valid: false,
		},
		{
			name: "Negative limit",
			request: DataRequest{
				Dataset: "HumanEval",
				Query:   "test query",
				Format:  "json",
				Limit:   -1,
			},
			valid: false,
		},
		{
			name: "Negative offset",
			request: DataRequest{
				Dataset: "HumanEval",
				Query:   "test query",
				Format:  "json",
				Offset:  -1,
			},
			valid: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDataRequest(&tt.request)
			if tt.valid && err != nil {
				t.Errorf("Expected valid request, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected invalid request, got no error")
			}
		})
	}
}

func TestDataCacheOperations(t *testing.T) {
	cache := NewDataCache()
	
	// Test cache set and get
	key := "test_dataset_key"
	data := []DataRecord{
		{
			ID:       "1",
			Content:  "test content",
			Metadata: map[string]interface{}{"type": "test"},
		},
	}
	
	cache.Set(key, data, time.Hour)
	
	cachedData, found := cache.Get(key)
	if !found {
		t.Error("Expected to find cached data")
	}
	
	if len(cachedData) != 1 {
		t.Errorf("Expected 1 cached record, got %d", len(cachedData))
	}
	
	if cachedData[0].ID != "1" {
		t.Errorf("Expected record ID '1', got '%s'", cachedData[0].ID)
	}
	
	// Test cache expiration
	cache.Set("expire_test", data, time.Millisecond)
	time.Sleep(time.Millisecond * 2)
	
	_, found = cache.Get("expire_test")
	if found {
		t.Error("Expected cached data to be expired")
	}
}

func TestQueryProcessorCreation(t *testing.T) {
	processor := NewQueryProcessor()
	
	if processor.parsers == nil {
		t.Error("Query parsers should be initialized")
	}
	
	if processor.filters == nil {
		t.Error("Query filters should be initialized")
	}
	
	if processor.aggregators == nil {
		t.Error("Query aggregators should be initialized")
	}
}

func TestQueryProcessing(t *testing.T) {
	processor := NewQueryProcessor()
	
	tests := []struct {
		name     string
		query    string
		expected bool // whether query should be processed successfully
	}{
		{
			name:     "Simple text query",
			query:    "python functions",
			expected: true,
		},
		{
			name:     "Complex query with filters",
			query:    "SELECT * FROM dataset WHERE difficulty = 'easy'",
			expected: true,
		},
		{
			name:     "Empty query",
			query:    "",
			expected: false,
		},
		{
			name:     "Query with aggregation",
			query:    "COUNT(*) GROUP BY language",
			expected: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := processor.ProcessQuery(tt.query, map[string]interface{}{})
			
			if tt.expected && result.Error != "" {
				t.Errorf("Expected successful processing, got error: %s", result.Error)
			}
			
			if !tt.expected && result.Error == "" {
				t.Error("Expected processing error, got success")
			}
		})
	}
}

func TestDataProviderRegistration(t *testing.T) {
	server := createTestDataServer()
	
	provider := &DataProvider{
		Name:        "test_provider",
		Type:        "local",
		Config:      map[string]interface{}{"path": "/tmp/test"},
		Endpoints:   []string{"http://localhost:8080"},
		Credentials: map[string]string{"api_key": "test_key"},
	}
	
	server.RegisterProvider("test_dataset", provider)
	
	// Verify provider was registered
	server.mu.RLock()
	registeredProvider, exists := server.providers["test_dataset"]
	server.mu.RUnlock()
	
	if !exists {
		t.Error("Expected provider to be registered")
	}
	
	if registeredProvider.Name != "test_provider" {
		t.Errorf("Expected provider name 'test_provider', got '%s'", registeredProvider.Name)
	}
	
	if registeredProvider.Type != "local" {
		t.Errorf("Expected provider type 'local', got '%s'", registeredProvider.Type)
	}
}

func TestHumanEvalDataLoading(t *testing.T) {
	server := createTestDataServer()
	
	request := &DataRequest{
		Dataset: "HumanEval",
		Format:  "json",
		Limit:   10,
	}
	
	response := server.loadHumanEvalData(request)
	
	if !response.Success && response.Error == "" {
		t.Error("Expected either success or error message")
	}
	
	if response.Success && len(response.Data) == 0 {
		t.Error("Expected data records for successful response")
	}
	
	if response.ExecutionTime <= 0 {
		t.Error("Expected positive execution time")
	}
}

func TestTinyStoriesDataLoading(t *testing.T) {
	server := createTestDataServer()
	
	request := &DataRequest{
		Dataset: "TinyStories",
		Format:  "json",
		Limit:   5,
		Filters: map[string]interface{}{"min_length": 100},
	}
	
	response := server.loadTinyStoriesData(request)
	
	if !response.Success && response.Error == "" {
		t.Error("Expected either success or error message")
	}
	
	if response.ExecutionTime <= 0 {
		t.Error("Expected positive execution time")
	}
}

func TestGSM8KDataLoading(t *testing.T) {
	server := createTestDataServer()
	
	request := &DataRequest{
		Dataset: "GSM8K",
		Format:  "json",
		Limit:   10,
		Filters: map[string]interface{}{"difficulty": "easy"},
	}
	
	response := server.loadGSM8KData(request)
	
	if !response.Success && response.Error == "" {
		t.Error("Expected either success or error message")
	}
}

func TestBabyLMDataLoading(t *testing.T) {
	server := createTestDataServer()
	
	request := &DataRequest{
		Dataset: "BabyLM",
		Format:  "json",
		Limit:   20,
	}
	
	response := server.loadBabyLMData(request)
	
	if !response.Success && response.Error == "" {
		t.Error("Expected either success or error message")
	}
}

func TestMiniPileDataLoading(t *testing.T) {
	server := createTestDataServer()
	
	request := &DataRequest{
		Dataset: "MiniPile",
		Format:  "json",
		Limit:   15,
		Filters: map[string]interface{}{"source": "wikipedia"},
	}
	
	response := server.loadMiniPileData(request)
	
	if !response.Success && response.Error == "" {
		t.Error("Expected either success or error message")
	}
}

func TestDataRequestHandler(t *testing.T) {
	server := createTestDataServer()
	
	request := createTestDataRequest()
	requestBody, _ := json.Marshal(request)
	
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	server.handleDataRequest(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response DataResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal response: %v", err)
	}
	
	if response.Timestamp <= 0 {
		t.Error("Expected positive timestamp")
	}
}

func TestDataHealthEndpoint(t *testing.T) {
	server := createTestDataServer()
	
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	
	server.handleHealth(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var health map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &health)
	if err != nil {
		t.Errorf("Failed to unmarshal health response: %v", err)
	}
	
	if health["status"] != "healthy" {
		t.Error("Expected healthy status")
	}
}

func TestDataCacheEndpoint(t *testing.T) {
	server := createTestDataServer()
	
	// Add item to cache first
	testData := []DataRecord{{ID: "test", Content: "test content"}}
	server.cache.Set("test_key", testData, time.Hour)
	
	req := httptest.NewRequest("GET", "/cache", nil)
	w := httptest.NewRecorder()
	
	server.handleCache(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var cacheInfo map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &cacheInfo)
	if err != nil {
		t.Errorf("Failed to unmarshal cache response: %v", err)
	}
}

func TestDataStatsEndpoint(t *testing.T) {
	server := createTestDataServer()
	
	req := httptest.NewRequest("GET", "/stats", nil)
	w := httptest.NewRecorder()
	
	server.handleStats(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var stats DataStats
	err := json.Unmarshal(w.Body.Bytes(), &stats)
	if err != nil {
		t.Errorf("Failed to unmarshal stats response: %v", err)
	}
	
	if stats.TotalRequests < 0 {
		t.Error("Expected non-negative total requests")
	}
}

func TestDataFiltering(t *testing.T) {
	processor := NewQueryProcessor()
	
	// Test data
	records := []DataRecord{
		{
			ID:       "1",
			Content:  "python function",
			Metadata: map[string]interface{}{"language": "python", "difficulty": "easy"},
		},
		{
			ID:       "2",
			Content:  "java class",
			Metadata: map[string]interface{}{"language": "java", "difficulty": "hard"},
		},
		{
			ID:       "3",
			Content:  "python algorithm",
			Metadata: map[string]interface{}{"language": "python", "difficulty": "medium"},
		},
	}
	
	// Test language filter
	filters := map[string]interface{}{"language": "python"}
	filtered := processor.ApplyFilters(records, filters)
	
	if len(filtered) != 2 {
		t.Errorf("Expected 2 filtered records, got %d", len(filtered))
	}
	
	for _, record := range filtered {
		if record.Metadata["language"] != "python" {
			t.Error("Filtered record should have language 'python'")
		}
	}
}

func TestDataAggregation(t *testing.T) {
	processor := NewQueryProcessor()
	
	// Test data
	records := []DataRecord{
		{Metadata: map[string]interface{}{"language": "python"}},
		{Metadata: map[string]interface{}{"language": "python"}},
		{Metadata: map[string]interface{}{"language": "java"}},
	}
	
	// Test group by aggregation
	aggregation := map[string]interface{}{"group_by": "language"}
	result := processor.ApplyAggregation(records, aggregation)
	
	if result.Error != "" {
		t.Errorf("Aggregation failed: %s", result.Error)
	}
	
	if len(result.Groups) == 0 {
		t.Error("Expected aggregation groups")
	}
}

func TestDataFormatConversion(t *testing.T) {
	records := []DataRecord{
		{
			ID:      "1",
			Content: "test content",
			Metadata: map[string]interface{}{"key": "value"},
		},
	}
	
	// Test JSON format
	jsonData, err := convertToFormat(records, "json")
	if err != nil {
		t.Errorf("JSON conversion failed: %v", err)
	}
	
	if len(jsonData) == 0 {
		t.Error("Expected non-empty JSON data")
	}
	
	// Test CSV format
	csvData, err := convertToFormat(records, "csv")
	if err != nil {
		t.Errorf("CSV conversion failed: %v", err)
	}
	
	if len(csvData) == 0 {
		t.Error("Expected non-empty CSV data")
	}
	
	// Test invalid format
	_, err = convertToFormat(records, "invalid")
	if err == nil {
		t.Error("Expected error for invalid format")
	}
}

func TestConcurrentDataRequests(t *testing.T) {
	server := createTestDataServer()
	
	// Test concurrent requests don't cause race conditions
	done := make(chan bool, 5)
	
	for i := 0; i < 5; i++ {
		go func(id int) {
			request := DataRequest{
				Dataset:   "HumanEval",
				Query:     "test query",
				Format:    "json",
				Limit:     10,
				Timestamp: time.Now().Unix(),
			}
			
			requestBody, _ := json.Marshal(request)
			req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			
			server.handleDataRequest(w, req)
			
			if w.Code != http.StatusOK {
				t.Errorf("Request %d: Expected status 200, got %d", id, w.Code)
			}
			
			done <- true
		}(i)
	}
	
	// Wait for all requests to complete
	for i := 0; i < 5; i++ {
		<-done
	}
}

func TestDataErrorHandling(t *testing.T) {
	server := createTestDataServer()
	
	// Test invalid JSON
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	server.handleDataRequest(w, req)
	
	var response DataResponse
	json.Unmarshal(w.Body.Bytes(), &response)
	
	if response.Success {
		t.Error("Expected failure for invalid JSON")
	}
	
	if response.Error == "" {
		t.Error("Expected error message")
	}
}

// Benchmarks
func BenchmarkDataRequestHandling(b *testing.B) {
	server := createTestDataServer()
	request := createTestDataRequest()
	requestBody, _ := json.Marshal(request)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		server.handleDataRequest(w, req)
	}
}

func BenchmarkDataCacheOperations(b *testing.B) {
	cache := NewDataCache()
	data := []DataRecord{{ID: "test", Content: "test content"}}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := "benchmark_key"
		cache.Set(key, data, time.Hour)
		cache.Get(key)
	}
}

func BenchmarkQueryProcessing(b *testing.B) {
	processor := NewQueryProcessor()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processor.ProcessQuery("SELECT * FROM dataset WHERE id > 100", map[string]interface{}{})
	}
}

// Helper functions for testing
func validateDataRequest(req *DataRequest) error {
	if req.Dataset == "" {
		return fmt.Errorf("dataset is required")
	}
	
	validFormats := map[string]bool{"json": true, "csv": true, "parquet": true}
	if req.Format != "" && !validFormats[req.Format] {
		return fmt.Errorf("invalid format")
	}
	
	if req.Limit < 0 {
		return fmt.Errorf("limit cannot be negative")
	}
	
	if req.Offset < 0 {
		return fmt.Errorf("offset cannot be negative")
	}
	
	return nil
}

func convertToFormat(records []DataRecord, format string) ([]byte, error) {
	switch format {
	case "json":
		return json.Marshal(records)
	case "csv":
		// Simple CSV conversion for testing
		return []byte("id,content\n1,test content\n"), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}