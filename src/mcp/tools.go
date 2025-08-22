// Package tools provides MCP interface for development tools integration
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ToolRequest represents a tool execution request
type ToolRequest struct {
	Tool      string                 `json:"tool"`
	Action    string                 `json:"action"`
	Params    map[string]interface{} `json:"params"`
	Files     []string               `json:"files,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}

// ToolResponse represents a tool execution response
type ToolResponse struct {
	Success     bool                   `json:"success"`
	Output      string                 `json:"output,omitempty"`
	Errors      []string               `json:"errors,omitempty"`
	Warnings    []string               `json:"warnings,omitempty"`
	Suggestions []string               `json:"suggestions,omitempty"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
	ID          string                 `json:"id,omitempty"`
	Timestamp   int64                  `json:"timestamp"`
	Duration    int64                  `json:"duration_ms"`
}

// ToolConfig represents configuration for specific tools
type ToolConfig struct {
	Name       string            `json:"name"`
	Executable string            `json:"executable"`
	Args       []string          `json:"args"`
	WorkDir    string            `json:"work_dir"`
	Env        map[string]string `json:"env"`
	Timeout    time.Duration     `json:"timeout"`
}

// ToolsServer handles development tools integration
type ToolsServer struct {
	port        int
	httpServer  *http.Server
	toolConfigs map[string]*ToolConfig
	executor    *ToolExecutor
	cache       *ToolCache
	middleware  []ToolMiddlewareFunc
	mu          sync.RWMutex
}

// ToolExecutor handles tool execution with sandboxing
type ToolExecutor struct {
	maxConcurrent int
	semaphore     chan struct{}
	workDir       string
	timeout       time.Duration
}

// ToolCache provides caching for tool results
type ToolCache struct {
	data map[string]*ToolCacheEntry
	mu   sync.RWMutex
}

// ToolCacheEntry represents a cached tool result
type ToolCacheEntry struct {
	Response  *ToolResponse
	ExpiresAt time.Time
}

// ToolMiddlewareFunc defines middleware for tool execution
type ToolMiddlewareFunc func(next ToolHandlerFunc) ToolHandlerFunc

// ToolHandlerFunc defines tool handler signature
type ToolHandlerFunc func(ctx context.Context, req *ToolRequest) (*ToolResponse, error)

// NewToolsServer creates a new tools server instance
func NewToolsServer(port int) *ToolsServer {
	return &ToolsServer{
		port:        port,
		toolConfigs: make(map[string]*ToolConfig),
		executor:    NewToolExecutor(4, "/tmp/mcp-tools", 30*time.Second),
		cache:       NewToolCache(),
	}
}

// NewToolExecutor creates a new tool executor
func NewToolExecutor(maxConcurrent int, workDir string, timeout time.Duration) *ToolExecutor {
	// Create work directory if it doesn't exist
	os.MkdirAll(workDir, 0755)
	
	return &ToolExecutor{
		maxConcurrent: maxConcurrent,
		semaphore:     make(chan struct{}, maxConcurrent),
		workDir:       workDir,
		timeout:       timeout,
	}
}

// NewToolCache creates a new tool cache
func NewToolCache() *ToolCache {
	return &ToolCache{
		data: make(map[string]*ToolCacheEntry),
	}
}

// Start starts the tools server
func (s *ToolsServer) Start() error {
	s.registerDefaultTools()

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleToolRequest)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/tools", s.handleListTools)
	mux.HandleFunc("/configs", s.handleConfigs)

	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("Starting MCP Tools server on port %d", s.port)
	return s.httpServer.ListenAndServe()
}

// Stop stops the tools server
func (s *ToolsServer) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// RegisterTool registers a new tool configuration
func (s *ToolsServer) RegisterTool(name string, config *ToolConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.toolConfigs[name] = config
}

// AddMiddleware adds middleware to the server
func (s *ToolsServer) AddMiddleware(middleware ToolMiddlewareFunc) {
	s.middleware = append(s.middleware, middleware)
}

// handleToolRequest handles incoming tool execution requests
func (s *ToolsServer) handleToolRequest(w http.ResponseWriter, r *http.Request) {
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

	var req ToolRequest
	if err := json.Unmarshal(body, &req); err != nil {
		response := &ToolResponse{
			Success:   false,
			Errors:    []string{"Invalid JSON request"},
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

	// Execute tool
	start := time.Now()
	ctx := context.WithValue(r.Context(), "request_id", req.ID)
	response := s.executeTool(ctx, &req)
	response.Duration = time.Since(start).Milliseconds()

	// Cache successful responses
	if response.Success {
		s.cache.Set(cacheKey, response, time.Hour)
	}

	json.NewEncoder(w).Encode(response)
}

// executeTool executes a tool request through middleware chain
func (s *ToolsServer) executeTool(ctx context.Context, req *ToolRequest) *ToolResponse {
	handler := s.getToolHandler(req.Tool)
	
	// Apply middleware
	for i := len(s.middleware) - 1; i >= 0; i-- {
		handler = s.middleware[i](handler)
	}

	response, err := handler(ctx, req)
	if err != nil {
		return &ToolResponse{
			Success:   false,
			Errors:    []string{err.Error()},
			ID:        req.ID,
			Timestamp: time.Now().Unix(),
		}
	}

	return response
}

// getToolHandler returns the appropriate handler for a tool
func (s *ToolsServer) getToolHandler(tool string) ToolHandlerFunc {
	switch tool {
	case "rustc", "cargo":
		return s.handleRustTools
	case "zig":
		return s.handleZigTools
	case "go", "gofmt", "golint":
		return s.handleGoTools
	case "clippy":
		return s.handleClippy
	case "rustfmt":
		return s.handleRustfmt
	case "eslint", "prettier":
		return s.handleJSTools
	case "pylint", "black", "mypy":
		return s.handlePythonTools
	case "docker":
		return s.handleDockerTools
	case "git":
		return s.handleGitTools
	default:
		return s.handleGenericTool
	}
}

// handleRustTools handles Rust compilation and tools
func (s *ToolsServer) handleRustTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Action {
	case "compile":
		return s.executor.Execute(ctx, "cargo", []string{"build", "--release"}, req)
	case "test":
		return s.executor.Execute(ctx, "cargo", []string{"test"}, req)
	case "check":
		return s.executor.Execute(ctx, "cargo", []string{"check"}, req)
	case "bench":
		return s.executor.Execute(ctx, "cargo", []string{"bench"}, req)
	default:
		return nil, fmt.Errorf("unknown Rust action: %s", req.Action)
	}
}

// handleZigTools handles Zig compilation and tools
func (s *ToolsServer) handleZigTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Action {
	case "build":
		return s.executor.Execute(ctx, "zig", []string{"build", "-Doptimize=ReleaseFast"}, req)
	case "test":
		return s.executor.Execute(ctx, "zig", []string{"test"}, req)
	case "fmt":
		return s.executor.Execute(ctx, "zig", []string{"fmt", "--check"}, req)
	default:
		return nil, fmt.Errorf("unknown Zig action: %s", req.Action)
	}
}

// handleGoTools handles Go compilation and tools
func (s *ToolsServer) handleGoTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Action {
	case "build":
		return s.executor.Execute(ctx, "go", []string{"build", "./..."}, req)
	case "test":
		return s.executor.Execute(ctx, "go", []string{"test", "./..."}, req)
	case "fmt":
		return s.executor.Execute(ctx, "gofmt", []string{"-l", "."}, req)
	case "vet":
		return s.executor.Execute(ctx, "go", []string{"vet", "./..."}, req)
	case "mod":
		return s.executor.Execute(ctx, "go", []string{"mod", "tidy"}, req)
	default:
		return nil, fmt.Errorf("unknown Go action: %s", req.Action)
	}
}

// handleClippy handles Rust Clippy linting
func (s *ToolsServer) handleClippy(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	args := []string{"clippy", "--", "-D", "warnings"}
	if req.Action == "fix" {
		args = []string{"clippy", "--fix", "--allow-dirty"}
	}
	return s.executor.Execute(ctx, "cargo", args, req)
}

// handleRustfmt handles Rust formatting
func (s *ToolsServer) handleRustfmt(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	args := []string{"fmt"}
	if req.Action == "check" {
		args = append(args, "--check")
	}
	return s.executor.Execute(ctx, "cargo", args, req)
}

// handleJSTools handles JavaScript/TypeScript tools
func (s *ToolsServer) handleJSTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Tool {
	case "eslint":
		return s.executor.Execute(ctx, "eslint", []string{"."}, req)
	case "prettier":
		args := []string{"--check", "."}
		if req.Action == "format" {
			args = []string{"--write", "."}
		}
		return s.executor.Execute(ctx, "prettier", args, req)
	}
	return nil, fmt.Errorf("unknown JS tool: %s", req.Tool)
}

// handlePythonTools handles Python development tools
func (s *ToolsServer) handlePythonTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Tool {
	case "pylint":
		return s.executor.Execute(ctx, "pylint", []string{"."}, req)
	case "black":
		args := []string{"--check", "."}
		if req.Action == "format" {
			args = []string{"."}
		}
		return s.executor.Execute(ctx, "black", args, req)
	case "mypy":
		return s.executor.Execute(ctx, "mypy", []string{"."}, req)
	}
	return nil, fmt.Errorf("unknown Python tool: %s", req.Tool)
}

// handleDockerTools handles Docker operations
func (s *ToolsServer) handleDockerTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Action {
	case "build":
		tag, _ := req.Params["tag"].(string)
		if tag == "" {
			tag = "ultra-fast-ai:latest"
		}
		return s.executor.Execute(ctx, "docker", []string{"build", "-t", tag, "."}, req)
	case "run":
		image, _ := req.Params["image"].(string)
		if image == "" {
			image = "ultra-fast-ai:latest"
		}
		return s.executor.Execute(ctx, "docker", []string{"run", "--rm", image}, req)
	case "scan":
		image, _ := req.Params["image"].(string)
		return s.executor.Execute(ctx, "docker", []string{"scan", image}, req)
	default:
		return nil, fmt.Errorf("unknown Docker action: %s", req.Action)
	}
}

// handleGitTools handles Git operations
func (s *ToolsServer) handleGitTools(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	switch req.Action {
	case "status":
		return s.executor.Execute(ctx, "git", []string{"status", "--porcelain"}, req)
	case "diff":
		return s.executor.Execute(ctx, "git", []string{"diff", "--stat"}, req)
	case "log":
		return s.executor.Execute(ctx, "git", []string{"log", "--oneline", "-10"}, req)
	case "branch":
		return s.executor.Execute(ctx, "git", []string{"branch"}, req)
	default:
		return nil, fmt.Errorf("unknown Git action: %s", req.Action)
	}
}

// handleGenericTool handles custom tool execution
func (s *ToolsServer) handleGenericTool(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	s.mu.RLock()
	config, exists := s.toolConfigs[req.Tool]
	s.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown tool: %s", req.Tool)
	}

	return s.executor.ExecuteWithConfig(ctx, config, req)
}

// Execute executes a command with the given arguments
func (e *ToolExecutor) Execute(ctx context.Context, command string, args []string, req *ToolRequest) (*ToolResponse, error) {
	// Acquire semaphore
	select {
	case e.semaphore <- struct{}{}:
		defer func() { <-e.semaphore }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Create command
	cmd := exec.CommandContext(ctx, command, args...)
	cmd.Dir = e.workDir
	
	// Set working directory if specified in request
	if workDir, ok := req.Params["work_dir"].(string); ok {
		cmd.Dir = workDir
	}

	// Execute command
	output, err := cmd.CombinedOutput()
	
	response := &ToolResponse{
		Success:   err == nil,
		Output:    string(output),
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}

	if err != nil {
		response.Errors = []string{err.Error()}
		// Try to parse tool-specific error formats
		response.Errors = append(response.Errors, s.parseToolErrors(req.Tool, string(output))...)
	}

	return response, nil
}

// ExecuteWithConfig executes a command with custom configuration
func (e *ToolExecutor) ExecuteWithConfig(ctx context.Context, config *ToolConfig, req *ToolRequest) (*ToolResponse, error) {
	// Acquire semaphore
	select {
	case e.semaphore <- struct{}{}:
		defer func() { <-e.semaphore }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Create command
	cmd := exec.CommandContext(ctx, config.Executable, config.Args...)
	cmd.Dir = config.WorkDir
	if cmd.Dir == "" {
		cmd.Dir = e.workDir
	}

	// Set environment variables
	cmd.Env = os.Environ()
	for key, value := range config.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
	}

	// Execute command
	output, err := cmd.CombinedOutput()
	
	response := &ToolResponse{
		Success:   err == nil,
		Output:    string(output),
		ID:        req.ID,
		Timestamp: time.Now().Unix(),
	}

	if err != nil {
		response.Errors = []string{err.Error()}
	}

	return response, nil
}

// parseToolErrors parses tool-specific error formats
func (s *ToolsServer) parseToolErrors(tool, output string) []string {
	var errors []string
	
	switch tool {
	case "rustc", "cargo":
		// Parse Rust compiler errors
		lines := strings.Split(output, "\n")
		for _, line := range lines {
			if strings.Contains(line, "error:") {
				errors = append(errors, strings.TrimSpace(line))
			}
		}
	case "eslint":
		// Parse ESLint errors
		lines := strings.Split(output, "\n")
		for _, line := range lines {
			if strings.Contains(line, "error") {
				errors = append(errors, strings.TrimSpace(line))
			}
		}
	}
	
	return errors
}

// generateCacheKey generates a cache key for the tool request
func (s *ToolsServer) generateCacheKey(req *ToolRequest) string {
	data, _ := json.Marshal(map[string]interface{}{
		"tool":   req.Tool,
		"action": req.Action,
		"params": req.Params,
		"files":  req.Files,
	})
	return fmt.Sprintf("tool:%x", data)
}

// Cache operations
func (tc *ToolCache) Get(key string) (*ToolResponse, bool) {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	entry, exists := tc.data[key]
	if !exists {
		return nil, false
	}

	if time.Now().After(entry.ExpiresAt) {
		delete(tc.data, key)
		return nil, false
	}

	return entry.Response, true
}

func (tc *ToolCache) Set(key string, response *ToolResponse, ttl time.Duration) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.data[key] = &ToolCacheEntry{
		Response:  response,
		ExpiresAt: time.Now().Add(ttl),
	}
}

// registerDefaultTools registers default tool configurations
func (s *ToolsServer) registerDefaultTools() {
	// Rust tools
	s.RegisterTool("rustc", &ToolConfig{
		Name:       "rustc",
		Executable: "rustc",
		Args:       []string{},
		Timeout:    60 * time.Second,
	})

	// Zig tools
	s.RegisterTool("zig", &ToolConfig{
		Name:       "zig",
		Executable: "zig",
		Args:       []string{},
		Timeout:    30 * time.Second,
	})

	// Go tools
	s.RegisterTool("go", &ToolConfig{
		Name:       "go",
		Executable: "go",
		Args:       []string{},
		Timeout:    30 * time.Second,
	})

	// Docker
	s.RegisterTool("docker", &ToolConfig{
		Name:       "docker",
		Executable: "docker",
		Args:       []string{},
		Timeout:    120 * time.Second,
	})
}

// HTTP handlers
func (s *ToolsServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"tools":     len(s.toolConfigs),
	})
}

func (s *ToolsServer) handleListTools(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	tools := make([]string, 0, len(s.toolConfigs))
	for name := range s.toolConfigs {
		tools = append(tools, name)
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tools":     tools,
		"timestamp": time.Now().Unix(),
	})
}

func (s *ToolsServer) handleConfigs(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	configs := make(map[string]*ToolConfig)
	for name, config := range s.toolConfigs {
		configs[name] = config
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(configs)
}

// Middleware functions
func LoggingToolMiddleware(next ToolHandlerFunc) ToolHandlerFunc {
	return func(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
		start := time.Now()
		log.Printf("Tool Request: %s/%s - Files: %v", req.Tool, req.Action, req.Files)
		
		response, err := next(ctx, req)
		
		duration := time.Since(start)
		success := response != nil && response.Success
		log.Printf("Tool Response: %s/%s - Success: %v - Duration: %v", 
			req.Tool, req.Action, success, duration)
		
		return response, err
	}
}

func SecurityToolMiddleware(next ToolHandlerFunc) ToolHandlerFunc {
	return func(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
		// Check for dangerous commands
		dangerousTools := map[string]bool{
			"rm": true, "del": true, "format": true, "fdisk": true,
		}
		
		if dangerousTools[req.Tool] {
			return nil, fmt.Errorf("tool %s is not allowed for security reasons", req.Tool)
		}
		
		// Check for dangerous parameters
		if workDir, ok := req.Params["work_dir"].(string); ok {
			if strings.Contains(workDir, "..") || strings.HasPrefix(workDir, "/") {
				return nil, fmt.Errorf("invalid work directory path")
			}
		}
		
		return next(ctx, req)
	}
}

// main function to run the tools server
func main() {
	port := 8002
	if len(os.Args) > 1 {
		if p, err := strconv.Atoi(os.Args[1]); err == nil {
			port = p
		}
	}

	server := NewToolsServer(port)
	
	// Add middleware
	server.AddMiddleware(LoggingToolMiddleware)
	server.AddMiddleware(SecurityToolMiddleware)
	
	// Start server
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start Tools server: %v", err)
	}
}