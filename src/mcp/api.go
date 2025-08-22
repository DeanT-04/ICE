// Package mcp provides Model Context Protocol interfaces for external service integration
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// APIRequest represents a generic API request structure
type APIRequest struct {
	Method    string                 `json:"method"`
	Params    map[string]interface{} `json:"params"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}

// APIResponse represents a generic API response structure
type APIResponse struct {
	Success   bool                   `json:"success"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}

// APIServer handles external API calls and integrations
type APIServer struct {
	port         int
	httpServer   *http.Server
	apiClients   map[string]*http.Client
	rateLimiter  *RateLimiter
	cache        *APICache
	middleware   []MiddlewareFunc
	routes       map[string]HandlerFunc
	mu           sync.RWMutex
}

// HandlerFunc defines the signature for API handlers
type HandlerFunc func(ctx context.Context, req *APIRequest) (*APIResponse, error)

// MiddlewareFunc defines the signature for middleware functions
type MiddlewareFunc func(next HandlerFunc) HandlerFunc

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	tokens   int
	capacity int
	refill   time.Duration
	mu       sync.Mutex
	lastRefill time.Time
}

// APICache provides caching for API responses
type APICache struct {
	data map[string]*CacheEntry
	mu   sync.RWMutex
}

// CacheEntry represents a cached API response
type CacheEntry struct {
	Response  *APIResponse
	ExpiresAt time.Time
}

// NewAPIServer creates a new API server instance
func NewAPIServer(port int) *APIServer {
	return &APIServer{
		port:        port,
		apiClients:  make(map[string]*http.Client),
		rateLimiter: NewRateLimiter(100, time.Minute), // 100 requests per minute
		cache:       NewAPICache(),
		routes:      make(map[string]HandlerFunc),
	}
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(capacity int, refillInterval time.Duration) *RateLimiter {
	return &RateLimiter{
		tokens:     capacity,
		capacity:   capacity,
		refill:     refillInterval,
		lastRefill: time.Now(),
	}
}

// NewAPICache creates a new API cache
func NewAPICache() *APICache {
	return &APICache{
		data: make(map[string]*CacheEntry),
	}
}

// Allow checks if a request is allowed under rate limiting
func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	if now.Sub(rl.lastRefill) >= rl.refill {
		rl.tokens = rl.capacity
		rl.lastRefill = now
	}

	if rl.tokens > 0 {
		rl.tokens--
		return true
	}
	return false
}

// Get retrieves a cached response if valid
func (ac *APICache) Get(key string) (*APIResponse, bool) {
	ac.mu.RLock()
	defer ac.mu.RUnlock()

	entry, exists := ac.data[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		delete(ac.data, key)
		return nil, false
	}

	return entry.Response, true
}

// Set stores a response in the cache
func (ac *APICache) Set(key string, response *APIResponse, ttl time.Duration) {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	ac.data[key] = &CacheEntry{
		Response:  response,
		ExpiresAt: time.Now().Add(ttl),
	}
}

// Start starts the API server
func (s *APIServer) Start() error {
	s.registerDefaultRoutes()

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRequest)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)

	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("Starting MCP API server on port %d", s.port)
	return s.httpServer.ListenAndServe()
}

// Stop stops the API server gracefully
func (s *APIServer) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// RegisterRoute registers a new API route handler
func (s *APIServer) RegisterRoute(method string, handler HandlerFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.routes[method] = handler
}

// AddMiddleware adds middleware to the server
func (s *APIServer) AddMiddleware(middleware MiddlewareFunc) {
	s.middleware = append(s.middleware, middleware)
}

// handleRequest handles incoming HTTP requests
func (s *APIServer) handleRequest(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
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

	// Check rate limiting
	if !s.rateLimiter.Allow() {
		response := &APIResponse{
			Success:   false,
			Error:     "Rate limit exceeded",
			Timestamp: time.Now().Unix(),
		}
		json.NewEncoder(w).Encode(response)
		return
	}

	// Parse request
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	var req APIRequest
	if err := json.Unmarshal(body, &req); err != nil {
		response := &APIResponse{
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
	ctx := context.WithValue(r.Context(), "request_id", req.ID)
	response, err := s.processRequest(ctx, &req)
	if err != nil {
		response = &APIResponse{
			Success:   false,
			Error:     err.Error(),
			ID:        req.ID,
			Timestamp: time.Now().Unix(),
		}
	}

	// Cache successful responses
	if response.Success {
		s.cache.Set(cacheKey, response, time.Hour)
	}

	json.NewEncoder(w).Encode(response)
}

// processRequest processes an API request through the middleware chain
func (s *APIServer) processRequest(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	s.mu.RLock()
	handler, exists := s.routes[req.Method]
	s.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown method: %s", req.Method)
	}

	// Apply middleware
	for i := len(s.middleware) - 1; i >= 0; i-- {
		handler = s.middleware[i](handler)
	}

	return handler(ctx, req)
}

// generateCacheKey generates a cache key for the request
func (s *APIServer) generateCacheKey(req *APIRequest) string {
	data, _ := json.Marshal(map[string]interface{}{
		"method": req.Method,
		"params": req.Params,
	})
	return fmt.Sprintf("api:%x", data)
}

// registerDefaultRoutes registers default API routes
func (s *APIServer) registerDefaultRoutes() {
	// GitHub API integration
	s.RegisterRoute("github_search", s.handleGitHubSearch)
	s.RegisterRoute("github_repo", s.handleGitHubRepo)
	
	// Library documentation
	s.RegisterRoute("docs_search", s.handleDocsSearch)
	s.RegisterRoute("docs_get", s.handleDocsGet)
	
	// External APIs
	s.RegisterRoute("http_get", s.handleHTTPGet)
	s.RegisterRoute("http_post", s.handleHTTPPost)
	
	// AI model APIs
	s.RegisterRoute("openai_chat", s.handleOpenAIChat)
	s.RegisterRoute("huggingface_model", s.handleHuggingFaceModel)
}

// handleGitHubSearch handles GitHub repository search
func (s *APIServer) handleGitHubSearch(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	query, ok := req.Params["query"].(string)
	if !ok {
		return nil, fmt.Errorf("missing query parameter")
	}

	// Mock GitHub search response
	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"query": query,
			"results": []map[string]interface{}{
				{
					"name":        "ultra-fast-ai",
					"full_name":   "user/ultra-fast-ai",
					"description": "Ultra-fast AI model implementation",
					"language":    "Rust",
					"stars":       100,
				},
			},
			"total_count": 1,
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleGitHubRepo handles GitHub repository information
func (s *APIServer) handleGitHubRepo(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	repo, ok := req.Params["repo"].(string)
	if !ok {
		return nil, fmt.Errorf("missing repo parameter")
	}

	// Mock repository response
	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"name":        repo,
			"description": "Repository description",
			"language":    "Rust",
			"stars":       100,
			"forks":       10,
			"issues":      5,
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleDocsSearch handles documentation search
func (s *APIServer) handleDocsSearch(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	query, ok := req.Params["query"].(string)
	if !ok {
		return nil, fmt.Errorf("missing query parameter")
	}

	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"query": query,
			"docs": []map[string]interface{}{
				{
					"title":   "Rust Documentation",
					"url":     "https://doc.rust-lang.org/",
					"snippet": "The Rust Programming Language documentation",
				},
			},
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleDocsGet handles documentation retrieval
func (s *APIServer) handleDocsGet(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	url, ok := req.Params["url"].(string)
	if !ok {
		return nil, fmt.Errorf("missing url parameter")
	}

	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"url":     url,
			"content": "Documentation content would be fetched here",
			"type":    "documentation",
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleHTTPGet handles generic HTTP GET requests
func (s *APIServer) handleHTTPGet(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	url, ok := req.Params["url"].(string)
	if !ok {
		return nil, fmt.Errorf("missing url parameter")
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"url":         url,
			"status_code": resp.StatusCode,
			"content":     string(body),
			"headers":     resp.Header,
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleHTTPPost handles generic HTTP POST requests
func (s *APIServer) handleHTTPPost(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	url, ok := req.Params["url"].(string)
	if !ok {
		return nil, fmt.Errorf("missing url parameter")
	}

	// Implementation would go here for POST requests
	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"url":     url,
			"method":  "POST",
			"status":  "completed",
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleOpenAIChat handles OpenAI API integration (mock)
func (s *APIServer) handleOpenAIChat(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	prompt, ok := req.Params["prompt"].(string)
	if !ok {
		return nil, fmt.Errorf("missing prompt parameter")
	}

	// Mock OpenAI response
	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"prompt":   prompt,
			"response": "Mock AI response to: " + prompt,
			"model":    "gpt-3.5-turbo",
			"tokens":   100,
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleHuggingFaceModel handles Hugging Face model API (mock)
func (s *APIServer) handleHuggingFaceModel(ctx context.Context, req *APIRequest) (*APIResponse, error) {
	model, ok := req.Params["model"].(string)
	if !ok {
		return nil, fmt.Errorf("missing model parameter")
	}

	return &APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"model":    model,
			"status":   "available",
			"endpoint": "https://api-inference.huggingface.co/models/" + model,
		},
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}, nil
}

// handleHealth handles health check requests
func (s *APIServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"version":   "1.0.0",
	})
}

// handleMetrics handles metrics requests
func (s *APIServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	cacheStats := map[string]interface{}{
		"cache_size": len(s.cache.data),
		"routes":     len(s.routes),
	}
	
	json.NewEncoder(w).Encode(map[string]interface{}{
		"metrics":   cacheStats,
		"timestamp": time.Now().Unix(),
	})
}

// LoggingMiddleware adds request logging
func LoggingMiddleware(next HandlerFunc) HandlerFunc {
	return func(ctx context.Context, req *APIRequest) (*APIResponse, error) {
		start := time.Now()
		log.Printf("API Request: %s - %v", req.Method, req.Params)
		
		response, err := next(ctx, req)
		
		duration := time.Since(start)
		log.Printf("API Response: %s - Success: %v - Duration: %v", 
			req.Method, response != nil && response.Success, duration)
		
		return response, err
	}
}

// AuthMiddleware adds simple API key authentication
func AuthMiddleware(apiKey string) MiddlewareFunc {
	return func(next HandlerFunc) HandlerFunc {
		return func(ctx context.Context, req *APIRequest) (*APIResponse, error) {
			if key, ok := req.Params["api_key"].(string); !ok || key != apiKey {
				return nil, fmt.Errorf("invalid or missing API key")
			}
			return next(ctx, req)
		}
	}
}

// main function to run the API server
func main() {
	port := 8001
	if len(os.Args) > 1 {
		if p, err := strconv.Atoi(os.Args[1]); err == nil {
			port = p
		}
	}

	server := NewAPIServer(port)
	
	// Add middleware
	server.AddMiddleware(LoggingMiddleware)
	
	// Start server
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start API server: %v", err)
	}
}