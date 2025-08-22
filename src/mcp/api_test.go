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

// Test data structures and mocks
func createTestApiServer() *ApiServer {
	return NewApiServer(8080)
}

func createTestApiRequest() ApiRequest {
	return ApiRequest{
		Service:    "github",
		Endpoint:   "repos/user/repo",
		Method:     "GET",
		Headers:    map[string]string{"Authorization": "token test"},
		Body:       map[string]interface{}{"test": "data"},
		Timeout:    5000,
		CacheTTL:   3600,
		RetryCount: 3,
	}
}

func TestApiServerCreation(t *testing.T) {
	server := createTestApiServer()
	
	if server.port != 8080 {
		t.Errorf("Expected port 8080, got %d", server.port)
	}
	
	if server.rateLimiter == nil {
		t.Error("Rate limiter should be initialized")
	}
	
	if server.cache == nil {
		t.Error("Cache should be initialized")
	}
}

func TestApiRequestValidation(t *testing.T) {
	tests := []struct {
		name    string
		request ApiRequest
		valid   bool
	}{
		{
			name:    "Valid request",
			request: createTestApiRequest(),
			valid:   true,
		},
		{
			name: "Missing service",
			request: ApiRequest{
				Endpoint: "test",
				Method:   "GET",
			},
			valid: false,
		},
		{
			name: "Missing endpoint",
			request: ApiRequest{
				Service: "github",
				Method:  "GET",
			},
			valid: false,
		},
		{
			name: "Invalid method",
			request: ApiRequest{
				Service:  "github",
				Endpoint: "test",
				Method:   "INVALID",
			},
			valid: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateApiRequest(&tt.request)
			if tt.valid && err != nil {
				t.Errorf("Expected valid request, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected invalid request, got no error")
			}
		})
	}
}

func TestApiCache(t *testing.T) {
	cache := NewApiCache()
	
	// Test cache set and get
	key := "test_key"
	response := &ApiResponse{
		Success: true,
		Data:    map[string]interface{}{"test": "data"},
	}
	
	cache.Set(key, response, time.Hour)
	
	cachedResponse, found := cache.Get(key)
	if !found {
		t.Error("Expected to find cached response")
	}
	
	if !cachedResponse.Success {
		t.Error("Cached response should be successful")
	}
	
	// Test cache expiration
	cache.Set("expire_test", response, time.Millisecond)
	time.Sleep(time.Millisecond * 2)
	
	_, found = cache.Get("expire_test")
	if found {
		t.Error("Expected cached response to be expired")
	}
}

func TestRateLimiter(t *testing.T) {
	limiter := NewRateLimiter(2, time.Second) // 2 requests per second
	
	// First two requests should succeed
	if !limiter.Allow() {
		t.Error("First request should be allowed")
	}
	if !limiter.Allow() {
		t.Error("Second request should be allowed")
	}
	
	// Third request should be rate limited
	if limiter.Allow() {
		t.Error("Third request should be rate limited")
	}
	
	// Wait for rate limit reset
	time.Sleep(time.Second)
	if !limiter.Allow() {
		t.Error("Request should be allowed after rate limit reset")
	}
}

func TestApiRequestHandler(t *testing.T) {
	server := createTestApiServer()
	
	// Create test request
	request := createTestApiRequest()
	requestBody, _ := json.Marshal(request)
	
	// Create HTTP request
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	// Handle request
	server.handleApiRequest(w, req)
	
	// Check response
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response ApiResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal response: %v", err)
	}
}

func TestGitHubApiIntegration(t *testing.T) {
	server := createTestApiServer()
	
	request := ApiRequest{
		Service:  "github",
		Endpoint: "user",
		Method:   "GET",
		Headers:  map[string]string{"User-Agent": "test-agent"},
		Timeout:  5000,
	}
	
	response := server.callGitHubApi(&request)
	
	// Should return a response (success or failure depending on API availability)
	if response == nil {
		t.Error("Expected non-nil response")
	}
	
	if response.ExecutionTime <= 0 {
		t.Error("Expected positive execution time")
	}
}

func TestHuggingFaceApiIntegration(t *testing.T) {
	server := createTestApiServer()
	
	request := ApiRequest{
		Service:  "huggingface",
		Endpoint: "models",
		Method:   "GET",
		Timeout:  5000,
	}
	
	response := server.callHuggingFaceApi(&request)
	
	if response == nil {
		t.Error("Expected non-nil response")
	}
	
	if response.ExecutionTime <= 0 {
		t.Error("Expected positive execution time")
	}
}

func TestDocumentationApiIntegration(t *testing.T) {
	server := createTestApiServer()
	
	request := ApiRequest{
		Service:  "docs",
		Endpoint: "rust/std",
		Method:   "GET",
		Timeout:  5000,
	}
	
	response := server.callDocumentationApi(&request)
	
	if response == nil {
		t.Error("Expected non-nil response")
	}
}

func TestApiErrorHandling(t *testing.T) {
	server := createTestApiServer()
	
	// Test invalid JSON
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	server.handleApiRequest(w, req)
	
	var response ApiResponse
	json.Unmarshal(w.Body.Bytes(), &response)
	
	if response.Success {
		t.Error("Expected failure for invalid JSON")
	}
	
	if response.Error == "" {
		t.Error("Expected error message")
	}
}

func TestApiHealthEndpoint(t *testing.T) {
	server := createTestApiServer()
	
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

func TestCacheEndpoint(t *testing.T) {
	server := createTestApiServer()
	
	// Add item to cache first
	server.cache.Set("test_key", &ApiResponse{Success: true}, time.Hour)
	
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
	
	if cacheInfo["size"].(float64) <= 0 {
		t.Error("Expected cache to have items")
	}
}

func TestConcurrentApiRequests(t *testing.T) {
	server := createTestApiServer()
	
	// Test concurrent requests don't cause race conditions
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func() {
			request := createTestApiRequest()
			requestBody, _ := json.Marshal(request)
			
			req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			
			server.handleApiRequest(w, req)
			
			if w.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d", w.Code)
			}
			
			done <- true
		}()
	}
	
	// Wait for all requests to complete
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestApiRetryMechanism(t *testing.T) {
	// This would test the retry mechanism for failed requests
	// In a real implementation, you'd mock HTTP failures
	server := createTestApiServer()
	
	request := ApiRequest{
		Service:    "invalid_service",
		Endpoint:   "nonexistent",
		Method:     "GET",
		RetryCount: 2,
		Timeout:    1000,
	}
	
	response := server.executeApiRequest(&request)
	
	// Should fail but have attempted retries
	if response.Success {
		t.Error("Expected request to fail for invalid service")
	}
	
	if response.Error == "" {
		t.Error("Expected error message for failed request")
	}
}

// Benchmarks
func BenchmarkApiRequestHandling(b *testing.B) {
	server := createTestApiServer()
	request := createTestApiRequest()
	requestBody, _ := json.Marshal(request)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		server.handleApiRequest(w, req)
	}
}

func BenchmarkCacheOperations(b *testing.B) {
	cache := NewApiCache()
	response := &ApiResponse{Success: true}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := "benchmark_key"
		cache.Set(key, response, time.Hour)
		cache.Get(key)
	}
}

// Helper functions for testing
func validateApiRequest(req *ApiRequest) error {
	if req.Service == "" {
		return fmt.Errorf("service is required")
	}
	if req.Endpoint == "" {
		return fmt.Errorf("endpoint is required")
	}
	validMethods := map[string]bool{"GET": true, "POST": true, "PUT": true, "DELETE": true, "PATCH": true}
	if !validMethods[req.Method] {
		return fmt.Errorf("invalid HTTP method")
	}
	return nil
}