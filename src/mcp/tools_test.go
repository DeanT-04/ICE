package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func createTestToolsServer() *ToolsServer {
	return NewToolsServer(8002)
}

func createTestToolRequest() ToolRequest {
	return ToolRequest{
		Tool:      "rustc",
		Action:    "compile",
		Params:    map[string]interface{}{"work_dir": "/tmp/test"},
		Files:     []string{"main.rs"},
		Config:    map[string]interface{}{"optimization": "release"},
		Timestamp: time.Now().Unix(),
	}
}

func TestToolsServerCreation(t *testing.T) {
	server := createTestToolsServer()
	
	if server.port != 8002 {
		t.Errorf("Expected port 8002, got %d", server.port)
	}
	
	if server.executor == nil {
		t.Error("Tool executor should be initialized")
	}
	
	if server.cache == nil {
		t.Error("Tool cache should be initialized")
	}
	
	if server.toolConfigs == nil {
		t.Error("Tool configurations should be initialized")
	}
}

func TestToolConfigRegistration(t *testing.T) {
	server := createTestToolsServer()
	
	config := &ToolConfig{
		Name:       "custom_tool",
		Executable: "/usr/bin/custom",
		Args:       []string{"--version"},
		WorkDir:    "/tmp",
		Env:        map[string]string{"PATH": "/usr/bin"},
		Timeout:    30 * time.Second,
	}
	
	server.RegisterTool("custom_tool", config)
	
	// Verify tool was registered
	server.mu.RLock()
	registeredConfig, exists := server.toolConfigs["custom_tool"]
	server.mu.RUnlock()
	
	if !exists {
		t.Error("Expected tool to be registered")
	}
	
	if registeredConfig.Name != "custom_tool" {
		t.Errorf("Expected tool name 'custom_tool', got '%s'", registeredConfig.Name)
	}
	
	if registeredConfig.Executable != "/usr/bin/custom" {
		t.Errorf("Expected executable '/usr/bin/custom', got '%s'", registeredConfig.Executable)
	}
}

func TestToolRequestValidation(t *testing.T) {
	tests := []struct {
		name    string
		request ToolRequest
		valid   bool
	}{
		{
			name:    "Valid tool request",
			request: createTestToolRequest(),
			valid:   true,
		},
		{
			name: "Missing tool",
			request: ToolRequest{
				Action: "compile",
				Params: map[string]interface{}{},
			},
			valid: false,
		},
		{
			name: "Missing action",
			request: ToolRequest{
				Tool:   "rustc",
				Params: map[string]interface{}{},
			},
			valid: false,
		},
		{
			name: "Empty request",
			request: ToolRequest{},
			valid: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateToolRequest(&tt.request)
			if tt.valid && err != nil {
				t.Errorf("Expected valid request, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected invalid request, got no error")
			}
		})
	}
}

func TestToolExecutorCreation(t *testing.T) {
	executor := NewToolExecutor(4, "/tmp/test", 30*time.Second)
	
	if executor.maxConcurrent != 4 {
		t.Errorf("Expected maxConcurrent 4, got %d", executor.maxConcurrent)
	}
	
	if executor.workDir != "/tmp/test" {
		t.Errorf("Expected workDir '/tmp/test', got '%s'", executor.workDir)
	}
	
	if executor.timeout != 30*time.Second {
		t.Errorf("Expected timeout 30s, got %v", executor.timeout)
	}
	
	if len(executor.semaphore) != 4 {
		t.Errorf("Expected semaphore capacity 4, got %d", len(executor.semaphore))
	}
}

func TestToolCacheOperations(t *testing.T) {
	cache := NewToolCache()
	
	response := &ToolResponse{
		Success:   true,
		Output:    "compilation successful",
		Timestamp: time.Now().Unix(),
	}
	
	// Test cache set and get
	key := "test_cache_key"
	cache.Set(key, response, time.Hour)
	
	cachedResponse, found := cache.Get(key)
	if !found {
		t.Error("Expected to find cached response")
	}
	
	if !cachedResponse.Success {
		t.Error("Cached response should be successful")
	}
	
	if cachedResponse.Output != "compilation successful" {
		t.Errorf("Expected output 'compilation successful', got '%s'", cachedResponse.Output)
	}
	
	// Test cache expiration
	cache.Set("expire_test", response, time.Millisecond)
	time.Sleep(time.Millisecond * 2)
	
	_, found = cache.Get("expire_test")
	if found {
		t.Error("Expected cached response to be expired")
	}
}

func TestRustToolHandling(t *testing.T) {
	server := createTestToolsServer()
	
	tests := []struct {
		action string
		valid  bool
	}{
		{"compile", true},
		{"test", true},
		{"check", true},
		{"bench", true},
		{"invalid_action", false},
	}
	
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			request := &ToolRequest{
				Tool:   "cargo",
				Action: tt.action,
				Params: map[string]interface{}{},
			}
			
			ctx := context.Background()
			response, err := server.handleRustTools(ctx, request)
			
			if tt.valid {
				if err != nil {
					t.Errorf("Expected no error for valid action, got: %v", err)
				}
				if response == nil {
					t.Error("Expected non-nil response for valid action")
				}
			} else {
				if err == nil {
					t.Error("Expected error for invalid action")
				}
			}
		})
	}
}

func TestZigToolHandling(t *testing.T) {
	server := createTestToolsServer()
	
	tests := []struct {
		action string
		valid  bool
	}{
		{"build", true},
		{"test", true},
		{"fmt", true},
		{"invalid_action", false},
	}
	
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			request := &ToolRequest{
				Tool:   "zig",
				Action: tt.action,
				Params: map[string]interface{}{},
			}
			
			ctx := context.Background()
			response, err := server.handleZigTools(ctx, request)
			
			if tt.valid {
				if err != nil {
					t.Errorf("Expected no error for valid action, got: %v", err)
				}
				if response == nil {
					t.Error("Expected non-nil response for valid action")
				}
			} else {
				if err == nil {
					t.Error("Expected error for invalid action")
				}
			}
		})
	}
}

func TestGoToolHandling(t *testing.T) {
	server := createTestToolsServer()
	
	tests := []struct {
		action string
		valid  bool
	}{
		{"build", true},
		{"test", true},
		{"fmt", true},
		{"vet", true},
		{"mod", true},
		{"invalid_action", false},
	}
	
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			request := &ToolRequest{
				Tool:   "go",
				Action: tt.action,
				Params: map[string]interface{}{},
			}
			
			ctx := context.Background()
			response, err := server.handleGoTools(ctx, request)
			
			if tt.valid {
				if err != nil {
					t.Errorf("Expected no error for valid action, got: %v", err)
				}
				if response == nil {
					t.Error("Expected non-nil response for valid action")
				}
			} else {
				if err == nil {
					t.Error("Expected error for invalid action")
				}
			}
		})
	}
}

func TestDockerToolHandling(t *testing.T) {
	server := createTestToolsServer()
	
	tests := []struct {
		action string
		params map[string]interface{}
		valid  bool
	}{
		{
			action: "build",
			params: map[string]interface{}{"tag": "test:latest"},
			valid:  true,
		},
		{
			action: "run",
			params: map[string]interface{}{"image": "test:latest"},
			valid:  true,
		},
		{
			action: "scan",
			params: map[string]interface{}{"image": "test:latest"},
			valid:  true,
		},
		{
			action: "invalid_action",
			params: map[string]interface{}{},
			valid:  false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			request := &ToolRequest{
				Tool:   "docker",
				Action: tt.action,
				Params: tt.params,
			}
			
			ctx := context.Background()
			response, err := server.handleDockerTools(ctx, request)
			
			if tt.valid {
				if err != nil {
					t.Errorf("Expected no error for valid action, got: %v", err)
				}
				if response == nil {
					t.Error("Expected non-nil response for valid action")
				}
			} else {
				if err == nil {
					t.Error("Expected error for invalid action")
				}
			}
		})
	}
}

func TestGitToolHandling(t *testing.T) {
	server := createTestToolsServer()
	
	tests := []struct {
		action string
		valid  bool
	}{
		{"status", true},
		{"diff", true},
		{"log", true},
		{"branch", true},
		{"invalid_action", false},
	}
	
	for _, tt := range tests {
		t.Run(tt.action, func(t *testing.T) {
			request := &ToolRequest{
				Tool:   "git",
				Action: tt.action,
				Params: map[string]interface{}{},
			}
			
			ctx := context.Background()
			response, err := server.handleGitTools(ctx, request)
			
			if tt.valid {
				if err != nil {
					t.Errorf("Expected no error for valid action, got: %v", err)
				}
				if response == nil {
					t.Error("Expected non-nil response for valid action")
				}
			} else {
				if err == nil {
					t.Error("Expected error for invalid action")
				}
			}
		})
	}
}

func TestToolRequestHandler(t *testing.T) {
	server := createTestToolsServer()
	
	request := createTestToolRequest()
	requestBody, _ := json.Marshal(request)
	
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	server.handleToolRequest(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response ToolResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal response: %v", err)
	}
	
	if response.Timestamp <= 0 {
		t.Error("Expected positive timestamp")
	}
}

func TestToolsHealthEndpoint(t *testing.T) {
	server := createTestToolsServer()
	
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
	
	if health["tools"] == nil {
		t.Error("Expected tools count in health response")
	}
}

func TestListToolsEndpoint(t *testing.T) {
	server := createTestToolsServer()
	
	// Register a test tool
	server.RegisterTool("test_tool", &ToolConfig{Name: "test_tool"})
	
	req := httptest.NewRequest("GET", "/tools", nil)
	w := httptest.NewRecorder()
	
	server.handleListTools(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal response: %v", err)
	}
	
	tools, ok := response["tools"].([]interface{})
	if !ok {
		t.Error("Expected tools array in response")
	}
	
	if len(tools) == 0 {
		t.Error("Expected at least one tool in response")
	}
}

func TestToolConfigsEndpoint(t *testing.T) {
	server := createTestToolsServer()
	
	req := httptest.NewRequest("GET", "/configs", nil)
	w := httptest.NewRecorder()
	
	server.handleConfigs(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var configs map[string]*ToolConfig
	err := json.Unmarshal(w.Body.Bytes(), &configs)
	if err != nil {
		t.Errorf("Failed to unmarshal configs response: %v", err)
	}
	
	// Should have default tool configurations
	if len(configs) == 0 {
		t.Error("Expected at least one tool configuration")
	}
}

func TestMiddlewareLogging(t *testing.T) {
	// Test the logging middleware
	called := false
	testHandler := func(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
		called = true
		return &ToolResponse{Success: true}, nil
	}
	
	wrappedHandler := LoggingToolMiddleware(testHandler)
	
	request := &ToolRequest{
		Tool:   "test",
		Action: "test_action",
	}
	
	ctx := context.Background()
	response, err := wrappedHandler(ctx, request)
	
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	
	if !called {
		t.Error("Expected handler to be called")
	}
	
	if !response.Success {
		t.Error("Expected successful response")
	}
}

func TestMiddlewareSecurity(t *testing.T) {
	// Test the security middleware
	testHandler := func(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
		return &ToolResponse{Success: true}, nil
	}
	
	wrappedHandler := SecurityToolMiddleware(testHandler)
	
	// Test dangerous tool blocking
	dangerousRequest := &ToolRequest{
		Tool:   "rm",
		Action: "delete",
	}
	
	ctx := context.Background()
	_, err := wrappedHandler(ctx, dangerousRequest)
	
	if err == nil {
		t.Error("Expected error for dangerous tool")
	}
	
	// Test path traversal protection
	traversalRequest := &ToolRequest{
		Tool:   "rustc",
		Action: "compile",
		Params: map[string]interface{}{"work_dir": "../../../etc"},
	}
	
	_, err = wrappedHandler(ctx, traversalRequest)
	if err == nil {
		t.Error("Expected error for path traversal attempt")
	}
	
	// Test safe request
	safeRequest := &ToolRequest{
		Tool:   "rustc",
		Action: "compile",
		Params: map[string]interface{}{"work_dir": "src"},
	}
	
	_, err = wrappedHandler(ctx, safeRequest)
	if err != nil {
		t.Errorf("Unexpected error for safe request: %v", err)
	}
}

func TestConcurrentToolExecution(t *testing.T) {
	server := createTestToolsServer()
	
	// Test concurrent tool requests don't cause race conditions
	done := make(chan bool, 5)
	
	for i := 0; i < 5; i++ {
		go func(id int) {
			request := ToolRequest{
				Tool:      "rustc",
				Action:    "check",
				Params:    map[string]interface{}{"id": id},
				Timestamp: time.Now().Unix(),
			}
			
			requestBody, _ := json.Marshal(request)
			req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			
			server.handleToolRequest(w, req)
			
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

// Benchmarks
func BenchmarkToolRequestHandling(b *testing.B) {
	server := createTestToolsServer()
	request := createTestToolRequest()
	requestBody, _ := json.Marshal(request)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		
		server.handleToolRequest(w, req)
	}
}

func BenchmarkToolCacheOperations(b *testing.B) {
	cache := NewToolCache()
	response := &ToolResponse{Success: true, Output: "test output"}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key := "benchmark_key"
		cache.Set(key, response, time.Hour)
		cache.Get(key)
	}
}

// Helper functions for testing
func validateToolRequest(req *ToolRequest) error {
	if req.Tool == "" {
		return fmt.Errorf("tool is required")
	}
	if req.Action == "" {
		return fmt.Errorf("action is required")
	}
	return nil
}