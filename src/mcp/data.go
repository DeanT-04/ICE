// Package data provides MCP interface for dataset access and management
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DataRequest represents a dataset access request
type DataRequest struct {
	Dataset   string                 `json:"dataset"`
	Action    string                 `json:"action"`
	Params    map[string]interface{} `json:"params"`
	Format    string                 `json:"format,omitempty"`
	Filters   map[string]interface{} `json:"filters,omitempty"`
	Limit     int                    `json:"limit,omitempty"`
	Offset    int                    `json:"offset,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}

// DataResponse represents a dataset access response
type DataResponse struct {
	Success   bool                   `json:"success"`
	Data      interface{}            `json:"data,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Error     string                 `json:"error,omitempty"`
	Count     int                    `json:"count,omitempty"`
	Total     int                    `json:"total,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
	Duration  int64                  `json:"duration_ms"`
}

// DatasetInfo represents information about a dataset
type DatasetInfo struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Source      string                 `json:"source"`
	Format      string                 `json:"format"`
	Size        int64                  `json:"size"`
	Count       int                    `json:"count"`
	Schema      map[string]interface{} `json:"schema,omitempty"`
	License     string                 `json:"license,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	LastUpdated time.Time              `json:"last_updated"`
}

// DataProvider interface for dataset providers
type DataProvider interface {
	GetInfo() *DatasetInfo
	Load(params map[string]interface{}) (interface{}, error)
	Search(query string, filters map[string]interface{}) (interface{}, error)
	Sample(count int) (interface{}, error)
	Validate() error
}

// DataServer handles dataset access and management
type DataServer struct {
	port         int
	httpServer   *http.Server
	providers    map[string]DataProvider
	cache        *DataCache
	downloadDir  string
	middleware   []DataMiddlewareFunc
	mu           sync.RWMutex
}

// DataCache provides caching for dataset queries
type DataCache struct {
	data map[string]*DataCacheEntry
	mu   sync.RWMutex
}

// DataCacheEntry represents a cached dataset query result
type DataCacheEntry struct {
	Response  *DataResponse
	ExpiresAt time.Time
}

// DataMiddlewareFunc defines middleware for data operations
type DataMiddlewareFunc func(next DataHandlerFunc) DataHandlerFunc

// DataHandlerFunc defines data handler signature
type DataHandlerFunc func(ctx context.Context, req *DataRequest) (*DataResponse, error)

// NewDataServer creates a new data server instance
func NewDataServer(port int, downloadDir string) *DataServer {
	// Create download directory if it doesn't exist
	os.MkdirAll(downloadDir, 0755)
	
	return &DataServer{
		port:        port,
		providers:   make(map[string]DataProvider),
		cache:       NewDataCache(),
		downloadDir: downloadDir,
	}
}

// NewDataCache creates a new data cache
func NewDataCache() *DataCache {
	return &DataCache{
		data: make(map[string]*DataCacheEntry),
	}
}

// Start starts the data server
func (s *DataServer) Start() error {
	s.registerDefaultProviders()

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleDataRequest)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/datasets", s.handleListDatasets)
	mux.HandleFunc("/info", s.handleDatasetInfo)
	mux.HandleFunc("/download", s.handleDownload)

	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  240 * time.Second,
	}

	log.Printf("Starting MCP Data server on port %d", s.port)
	return s.httpServer.ListenAndServe()
}

// Stop stops the data server
func (s *DataServer) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// RegisterProvider registers a new dataset provider
func (s *DataServer) RegisterProvider(name string, provider DataProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.providers[name] = provider
}

// AddMiddleware adds middleware to the server
func (s *DataServer) AddMiddleware(middleware DataMiddlewareFunc) {
	s.middleware = append(s.middleware, middleware)
}

// handleDataRequest handles incoming dataset requests
func (s *DataServer) handleDataRequest(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	var req DataRequest
	if err := json.Unmarshal(body, &req); err != nil {
		response := &DataResponse{
			Success:   false,
			Error:     "Invalid JSON request",
			Timestamp: time.Now().Unix(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Set timestamp if not provided
	if req.Timestamp == 0 {
		req.Timestamp = time.Now().Unix()
	}

	// Check cache
	cacheKey := s.generateCacheKey(&req)
	if cachedResponse, found := s.cache.Get(cacheKey); found {
		json.NewEncoder(w).Encode(cachedResponse)
		return
	}

	// Process request
	start := time.Now()
	ctx := context.WithValue(r.Context(), "request_id", req.ID)
	response := s.processDataRequest(ctx, &req)
	response.Duration = time.Since(start).Milliseconds()

	// Cache successful responses
	if response.Success {
		s.cache.Set(cacheKey, response, time.Hour*24) // 24 hour cache
	}

	json.NewEncoder(w).Encode(response)
}

// processDataRequest processes a data request through middleware chain
func (s *DataServer) processDataRequest(ctx context.Context, req *DataRequest) *DataResponse {
	handler := s.getDataHandler(req.Dataset)
	
	// Apply middleware
	for i := len(s.middleware) - 1; i >= 0; i-- {
		handler = s.middleware[i](handler)
	}

	response, err := handler(ctx, req)
	if err != nil {
		return &DataResponse{
			Success:   false,
			Error:     err.Error(),
			ID:        req.ID,
			Timestamp: time.Now().Unix(),
		}
	}

	return response
}

// getDataHandler returns the appropriate handler for a dataset
func (s *DataServer) getDataHandler(dataset string) DataHandlerFunc {
	return func(ctx context.Context, req *DataRequest) (*DataResponse, error) {
		s.mu.RLock()
		provider, exists := s.providers[dataset]
		s.mu.RUnlock()

		if !exists {
			return nil, fmt.Errorf("unknown dataset: %s", dataset)
		}

		switch req.Action {
		case "info":
			return s.handleInfo(provider, req)
		case "load":
			return s.handleLoad(provider, req)
		case "search":
			return s.handleSearch(provider, req)
		case "sample":
			return s.handleSample(provider, req)
		case "validate":
			return s.handleValidate(provider, req)
		default:
			return nil, fmt.Errorf("unknown action: %s", req.Action)
		}
	}
}

// handleInfo returns dataset information
func (s *DataServer) handleInfo(provider DataProvider, req *DataRequest) (*DataResponse, error) {
	info := provider.GetInfo()
	
	return &DataResponse{
		Success:   true,
		Data:      info,
		Metadata:  map[string]interface{}{"type": "dataset_info"},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleLoad loads dataset with optional filters
func (s *DataServer) handleLoad(provider DataProvider, req *DataRequest) (*DataResponse, error) {
	params := req.Params
	if params == nil {
		params = make(map[string]interface{})
	}
	
	// Add filters, limit, offset to params
	if req.Filters != nil {
		params["filters"] = req.Filters
	}
	if req.Limit > 0 {
		params["limit"] = req.Limit
	}
	if req.Offset > 0 {
		params["offset"] = req.Offset
	}
	if req.Format != "" {
		params["format"] = req.Format
	}

	data, err := provider.Load(params)
	if err != nil {
		return nil, err
	}

	return &DataResponse{
		Success:   true,
		Data:      data,
		Metadata:  map[string]interface{}{"type": "dataset_load"},
		Count:     s.getDataCount(data),
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleSearch searches dataset
func (s *DataServer) handleSearch(provider DataProvider, req *DataRequest) (*DataResponse, error) {
	query, ok := req.Params["query"].(string)
	if !ok {
		return nil, fmt.Errorf("missing search query")
	}

	data, err := provider.Search(query, req.Filters)
	if err != nil {
		return nil, err
	}

	return &DataResponse{
		Success:   true,
		Data:      data,
		Metadata:  map[string]interface{}{"type": "dataset_search", "query": query},
		Count:     s.getDataCount(data),
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleSample returns a sample of the dataset
func (s *DataServer) handleSample(provider DataProvider, req *DataRequest) (*DataResponse, error) {
	count := 10 // default sample size
	if req.Limit > 0 {
		count = req.Limit
	}
	if sampleCount, ok := req.Params["count"].(float64); ok {
		count = int(sampleCount)
	}

	data, err := provider.Sample(count)
	if err != nil {
		return nil, err
	}

	return &DataResponse{
		Success:   true,
		Data:      data,
		Metadata:  map[string]interface{}{"type": "dataset_sample", "sample_size": count},
		Count:     s.getDataCount(data),
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleValidate validates dataset integrity
func (s *DataServer) handleValidate(provider DataProvider, req *DataRequest) (*DataResponse, error) {
	err := provider.Validate()
	
	response := &DataResponse{
		Success:   err == nil,
		Metadata:  map[string]interface{}{"type": "dataset_validation"},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}
	
	if err != nil {
		response.Error = err.Error()
	} else {
		response.Data = map[string]interface{}{"status": "valid"}
	}

	return response, nil
}

// getDataCount estimates count from data response
func (s *DataServer) getDataCount(data interface{}) int {
	switch v := data.(type) {
	case []interface{}:
		return len(v)
	case map[string]interface{}:
		if items, ok := v["items"].([]interface{}); ok {
			return len(items)
		}
		return 1
	default:
		return 1
	}
}

// generateCacheKey generates a cache key for the data request
func (s *DataServer) generateCacheKey(req *DataRequest) string {
	data, _ := json.Marshal(map[string]interface{}{
		"dataset": req.Dataset,
		"action":  req.Action,
		"params":  req.Params,
		"filters": req.Filters,
		"limit":   req.Limit,
		"offset":  req.Offset,
	})
	return fmt.Sprintf("data:%x", data)
}

// Cache operations
func (dc *DataCache) Get(key string) (*DataResponse, bool) {
	dc.mu.RLock()
	defer dc.mu.RUnlock()

	entry, exists := dc.data[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		delete(dc.data, key)
		return nil, false
	}

	return entry.Response, true
}

func (dc *DataCache) Set(key string, response *DataResponse, ttl time.Duration) {
	dc.mu.Lock()
	defer dc.mu.Unlock()

	dc.data[key] = &DataCacheEntry{
		Response:  response,
		ExpiresAt: time.Now().Add(ttl),
	}
}

// HTTP handlers
func (s *DataServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"datasets":  len(s.providers),
	})
}

func (s *DataServer) handleListDatasets(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	datasets := make([]string, 0, len(s.providers))
	for name := range s.providers {
		datasets = append(datasets, name)
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"datasets":  datasets,
		"timestamp": time.Now().Unix(),
	})
}

func (s *DataServer) handleDatasetInfo(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	if dataset == "" {
		http.Error(w, "Missing dataset parameter", http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	provider, exists := s.providers[dataset]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, "Dataset not found", http.StatusNotFound)
		return
	}

	info := provider.GetInfo()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

func (s *DataServer) handleDownload(w http.ResponseWriter, r *http.Request) {
	dataset := r.URL.Query().Get("dataset")
	if dataset == "" {
		http.Error(w, "Missing dataset parameter", http.StatusBadRequest)
		return
	}

	// Mock download functionality
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"dataset":     dataset,
		"status":      "download_initiated",
		"destination": filepath.Join(s.downloadDir, dataset),
		"timestamp":   time.Now().Unix(),
	})
}

// Dataset provider implementations

// HumanEvalProvider provides access to HumanEval dataset
type HumanEvalProvider struct {
	dataPath string
}

func NewHumanEvalProvider(dataPath string) *HumanEvalProvider {
	return &HumanEvalProvider{dataPath: dataPath}
}

func (p *HumanEvalProvider) GetInfo() *DatasetInfo {
	return &DatasetInfo{
		Name:        "HumanEval",
		Description: "Hand-written programming problems for measuring functional correctness of code synthesis",
		Source:      "OpenAI",
		Format:      "JSONL",
		Count:       164,
		Schema: map[string]interface{}{
			"task_id":    "string",
			"prompt":     "string",
			"canonical_solution": "string",
			"test":       "string",
			"entry_point": "string",
		},
		License:     "MIT",
		Tags:        []string{"coding", "evaluation", "python"},
		LastUpdated: time.Now(),
	}
}

func (p *HumanEvalProvider) Load(params map[string]interface{}) (interface{}, error) {
	// Mock HumanEval data
	problems := []map[string]interface{}{
		{
			"task_id": "HumanEval/0",
			"prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"",
			"canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False",
			"test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True",
			"entry_point": "has_close_elements",
		},
	}
	
	return map[string]interface{}{
		"problems": problems,
		"count":    len(problems),
	}, nil
}

func (p *HumanEvalProvider) Search(query string, filters map[string]interface{}) (interface{}, error) {
	// Mock search functionality
	return map[string]interface{}{
		"query":   query,
		"results": []map[string]interface{}{},
		"count":   0,
	}, nil
}

func (p *HumanEvalProvider) Sample(count int) (interface{}, error) {
	// Return sample problems
	data, err := p.Load(nil)
	if err != nil {
		return nil, err
	}
	
	if dataMap, ok := data.(map[string]interface{}); ok {
		if problems, ok := dataMap["problems"].([]map[string]interface{}); ok {
			if count > len(problems) {
				count = len(problems)
			}
			return map[string]interface{}{
				"problems": problems[:count],
				"count":    count,
			}, nil
		}
	}
	
	return data, nil
}

func (p *HumanEvalProvider) Validate() error {
	// Mock validation
	return nil
}

// TinyStoriesProvider provides access to TinyStories dataset
type TinyStoriesProvider struct {
	dataPath string
}

func NewTinyStoriesProvider(dataPath string) *TinyStoriesProvider {
	return &TinyStoriesProvider{dataPath: dataPath}
}

func (p *TinyStoriesProvider) GetInfo() *DatasetInfo {
	return &DatasetInfo{
		Name:        "TinyStories",
		Description: "A dataset of simple stories for language model training",
		Source:      "Microsoft",
		Format:      "JSON",
		Count:       2100000,
		Schema: map[string]interface{}{
			"story": "string",
			"summary": "string",
		},
		License:     "MIT",
		Tags:        []string{"text", "stories", "language-modeling"},
		LastUpdated: time.Now(),
	}
}

func (p *TinyStoriesProvider) Load(params map[string]interface{}) (interface{}, error) {
	// Mock TinyStories data
	stories := []map[string]interface{}{
		{
			"story": "Once upon a time, there was a little girl named Lily. She loved to play with her toys.",
			"summary": "A little girl named Lily plays with toys.",
		},
	}
	
	return map[string]interface{}{
		"stories": stories,
		"count":   len(stories),
	}, nil
}

func (p *TinyStoriesProvider) Search(query string, filters map[string]interface{}) (interface{}, error) {
	return map[string]interface{}{
		"query":   query,
		"results": []map[string]interface{}{},
		"count":   0,
	}, nil
}

func (p *TinyStoriesProvider) Sample(count int) (interface{}, error) {
	data, err := p.Load(nil)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (p *TinyStoriesProvider) Validate() error {
	return nil
}

// registerDefaultProviders registers default dataset providers
func (s *DataServer) registerDefaultProviders() {
	// Register dataset providers
	s.RegisterProvider("humaneval", NewHumanEvalProvider(filepath.Join(s.downloadDir, "humaneval")))
	s.RegisterProvider("tinystories", NewTinyStoriesProvider(filepath.Join(s.downloadDir, "tinystories")))
	
	// Register other providers
	s.RegisterProvider("gsm8k", &MockDataProvider{
		name: "GSM8K",
		description: "Grade School Math 8K - math word problems",
		count: 8500,
		tags: []string{"math", "reasoning", "education"},
	})
	
	s.RegisterProvider("babylm", &MockDataProvider{
		name: "BabyLM",
		description: "Baby language modeling dataset",
		count: 100000000,
		tags: []string{"language-modeling", "child-directed-speech"},
	})
	
	s.RegisterProvider("minipile", &MockDataProvider{
		name: "MiniPile",
		description: "A smaller version of The Pile dataset",
		count: 1000000,
		tags: []string{"text", "diverse", "web-crawl"},
	})
}

// MockDataProvider provides a mock implementation for testing
type MockDataProvider struct {
	name        string
	description string
	count       int
	tags        []string
}

func (p *MockDataProvider) GetInfo() *DatasetInfo {
	return &DatasetInfo{
		Name:        p.name,
		Description: p.description,
		Source:      "Mock",
		Format:      "JSON",
		Count:       p.count,
		Tags:        p.tags,
		LastUpdated: time.Now(),
	}
}

func (p *MockDataProvider) Load(params map[string]interface{}) (interface{}, error) {
	return map[string]interface{}{
		"data":  []interface{}{},
		"count": 0,
	}, nil
}

func (p *MockDataProvider) Search(query string, filters map[string]interface{}) (interface{}, error) {
	return map[string]interface{}{
		"query":   query,
		"results": []interface{}{},
		"count":   0,
	}, nil
}

func (p *MockDataProvider) Sample(count int) (interface{}, error) {
	return map[string]interface{}{
		"data":  []interface{}{},
		"count": 0,
	}, nil
}

func (p *MockDataProvider) Validate() error {
	return nil
}

// Middleware functions
func LoggingDataMiddleware(next DataHandlerFunc) DataHandlerFunc {
	return func(ctx context.Context, req *DataRequest) (*DataResponse, error) {
		start := time.Now()
		log.Printf("Data Request: %s/%s - Limit: %d", req.Dataset, req.Action, req.Limit)
		
		response, err := next(ctx, req)
		
		duration := time.Since(start)
		success := response != nil && response.Success
		log.Printf("Data Response: %s/%s - Success: %v - Duration: %v - Count: %d", 
			req.Dataset, req.Action, success, duration, response.Count)
		
		return response, err
	}
}

func CachingDataMiddleware(next DataHandlerFunc) DataHandlerFunc {
	return func(ctx context.Context, req *DataRequest) (*DataResponse, error) {
		// Add cache headers to response metadata
		response, err := next(ctx, req)
		if response != nil && response.Metadata == nil {
			response.Metadata = make(map[string]interface{})
		}
		if response != nil {
			response.Metadata["cached"] = false
		}
		return response, err
	}
}

// main function to run the data server
func main() {
	port := 8003
	downloadDir := "./datasets"
	
	if len(os.Args) > 1 {
		if p, err := strconv.Atoi(os.Args[1]); err == nil {
			port = p
		}
	}
	if len(os.Args) > 2 {
		downloadDir = os.Args[2]
	}

	server := NewDataServer(port, downloadDir)
	
	// Add middleware
	server.AddMiddleware(LoggingDataMiddleware)
	server.AddMiddleware(CachingDataMiddleware)
	
	// Start server
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start Data server: %v", err)
	}
}