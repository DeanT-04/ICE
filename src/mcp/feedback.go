// Package feedback provides MCP interface for self-improvement loops and model evolution
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// FeedbackRequest represents a feedback submission request
type FeedbackRequest struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Target    string                 `json:"target,omitempty"`
	Data      map[string]interface{} `json:"data"`
	Priority  string                 `json:"priority,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}

// FeedbackResponse represents a feedback processing response
type FeedbackResponse struct {
	Success     bool                   `json:"success"`
	FeedbackID  string                 `json:"feedback_id,omitempty"`
	Insights    []Insight              `json:"insights,omitempty"`
	Actions     []Action               `json:"actions,omitempty"`
	Error       string                 `json:"error,omitempty"`
	ID          string                 `json:"id,omitempty"`
	Timestamp   int64                  `json:"timestamp"`
	Duration    int64                  `json:"duration_ms"`
}

// ProcessedFeedback represents processed feedback data
type ProcessedFeedback struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Source      string                 `json:"source"`
	Category    string                 `json:"category"`
	Summary     string                 `json:"summary"`
	Impact      float64                `json:"impact"`
	Confidence  float64                `json:"confidence"`
	Data        map[string]interface{} `json:"data"`
	Processed   time.Time              `json:"processed"`
}

// Insight represents an insight derived from feedback
type Insight struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Confidence  float64   `json:"confidence"`
	Priority    int       `json:"priority"`
	Generated   time.Time `json:"generated"`
}

// Action represents a recommended action based on feedback
type Action struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Target      string                 `json:"target"`
	Parameters  map[string]interface{} `json:"parameters"`
	Priority    int                    `json:"priority"`
	Status      string                 `json:"status"`
	Created     time.Time              `json:"created"`
}

// FeedbackServer handles feedback processing and self-improvement
type FeedbackServer struct {
	port          int
	httpServer    *http.Server
	feedbacks     map[string]*ProcessedFeedback
	insights      map[string]*Insight
	actions       map[string]*Action
	analytics     *Analytics
	mu            sync.RWMutex
}

// Analytics provides basic analytics
type Analytics struct {
	TotalFeedbacks int                    `json:"total_feedbacks"`
	ByType         map[string]int         `json:"by_type"`
	BySource       map[string]int         `json:"by_source"`
	AvgImpact      float64                `json:"avg_impact"`
	Trends         map[string][]float64   `json:"trends"`
	mu             sync.RWMutex
}

// NewFeedbackServer creates a new feedback server instance
func NewFeedbackServer(port int) *FeedbackServer {
	return &FeedbackServer{
		port:      port,
		feedbacks: make(map[string]*ProcessedFeedback),
		insights:  make(map[string]*Insight),
		actions:   make(map[string]*Action),
		analytics: &Analytics{
			ByType:   make(map[string]int),
			BySource: make(map[string]int),
			Trends:   make(map[string][]float64),
		},
	}
}

// Start starts the feedback server
func (s *FeedbackServer) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleFeedbackRequest)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/insights", s.handleInsights)
	mux.HandleFunc("/actions", s.handleActions)
	mux.HandleFunc("/analytics", s.handleAnalytics)

	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start background processing
	go s.processInsightsLoop()
	go s.executeActionsLoop()

	log.Printf("Starting MCP Feedback server on port %d", s.port)
	return s.httpServer.ListenAndServe()
}

// Stop stops the feedback server
func (s *FeedbackServer) Stop(ctx context.Context) error {
	if s.httpServer != nil {
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// handleFeedbackRequest handles incoming feedback requests
func (s *FeedbackServer) handleFeedbackRequest(w http.ResponseWriter, r *http.Request) {
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

	var req FeedbackRequest
	if err := json.Unmarshal(body, &req); err != nil {
		response := &FeedbackResponse{
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

	// Process feedback
	start := time.Now()
	response := s.processFeedback(&req)
	response.Duration = time.Since(start).Milliseconds()

	json.NewEncoder(w).Encode(response)
}

// processFeedback processes feedback and generates insights/actions
func (s *FeedbackServer) processFeedback(req *FeedbackRequest) *FeedbackResponse {
	// Create processed feedback
	processed := &ProcessedFeedback{
		ID:         generateID(),
		Type:       req.Type,
		Source:     req.Source,
		Category:   s.categorize(req.Type),
		Summary:    fmt.Sprintf("%s feedback from %s", req.Type, req.Source),
		Impact:     s.calculateImpact(req),
		Confidence: s.calculateConfidence(req),
		Data:       req.Data,
		Processed:  time.Now(),
	}

	// Store feedback
	s.mu.Lock()
	s.feedbacks[processed.ID] = processed
	s.mu.Unlock()

	// Generate insights
	insights := s.generateInsights(processed)
	for _, insight := range insights {
		s.mu.Lock()
		s.insights[insight.ID] = &insight
		s.mu.Unlock()
	}

	// Generate actions
	actions := s.generateActions(processed, insights)
	for _, action := range actions {
		s.mu.Lock()
		s.actions[action.ID] = &action
		s.mu.Unlock()
	}

	// Update analytics
	s.updateAnalytics(processed)

	return &FeedbackResponse{
		Success:     true,
		FeedbackID:  processed.ID,
		Insights:    insights,
		Actions:     actions,
		ID:          req.ID,
		Timestamp:   time.Now().Unix(),
	}
}

// categorize determines feedback category
func (s *FeedbackServer) categorize(feedbackType string) string {
	categories := map[string]string{
		"performance": "optimization",
		"error":       "bug_fix",
		"accuracy":    "model_improvement",
		"latency":     "performance",
		"memory":      "resource_optimization",
	}
	
	if category, exists := categories[feedbackType]; exists {
		return category
	}
	return "general"
}

// calculateImpact estimates impact score
func (s *FeedbackServer) calculateImpact(req *FeedbackRequest) float64 {
	baseImpact := 0.5
	
	switch req.Type {
	case "error":
		baseImpact = 0.8
	case "performance":
		baseImpact = 0.7
	case "accuracy":
		baseImpact = 0.9
	case "latency":
		baseImpact = 0.6
	}
	
	switch req.Priority {
	case "high", "critical":
		baseImpact *= 1.5
	case "low":
		baseImpact *= 0.7
	}
	
	if baseImpact > 1.0 {
		baseImpact = 1.0
	}
	
	return baseImpact
}

// calculateConfidence estimates confidence score
func (s *FeedbackServer) calculateConfidence(req *FeedbackRequest) float64 {
	confidence := 0.7
	
	if sourceType, ok := req.Data["source_type"].(string); ok {
		switch sourceType {
		case "automated_test":
			confidence = 0.9
		case "monitoring":
			confidence = 0.8
		case "user_report":
			confidence = 0.6
		}
	}
	
	return confidence
}

// generateInsights creates insights from feedback
func (s *FeedbackServer) generateInsights(feedback *ProcessedFeedback) []Insight {
	var insights []Insight
	
	// High impact feedback generates insights
	if feedback.Impact > 0.7 {
		insights = append(insights, Insight{
			ID:          generateID(),
			Type:        "high_impact",
			Description: fmt.Sprintf("High impact %s detected in %s", feedback.Type, feedback.Source),
			Confidence:  feedback.Confidence,
			Priority:    8,
			Generated:   time.Now(),
		})
	}
	
	// Error patterns
	if feedback.Type == "error" {
		insights = append(insights, Insight{
			ID:          generateID(),
			Type:        "error_pattern",
			Description: "Error pattern detected - investigate root cause",
			Confidence:  0.8,
			Priority:    9,
			Generated:   time.Now(),
		})
	}
	
	return insights
}

// generateActions creates actions from feedback and insights
func (s *FeedbackServer) generateActions(feedback *ProcessedFeedback, insights []Insight) []Action {
	var actions []Action
	
	switch feedback.Type {
	case "performance":
		actions = append(actions, Action{
			ID:          generateID(),
			Type:        "optimize",
			Description: "Optimize performance based on feedback",
			Target:      feedback.Source,
			Parameters:  map[string]interface{}{"feedback_id": feedback.ID},
			Priority:    s.calculateActionPriority(feedback),
			Status:      "pending",
			Created:     time.Now(),
		})
		
	case "error":
		actions = append(actions, Action{
			ID:          generateID(),
			Type:        "debug",
			Description: "Debug and fix error",
			Target:      feedback.Source,
			Parameters:  map[string]interface{}{"error_data": feedback.Data},
			Priority:    10,
			Status:      "pending",
			Created:     time.Now(),
		})
		
	case "accuracy":
		actions = append(actions, Action{
			ID:          generateID(),
			Type:        "retrain",
			Description: "Retrain model to improve accuracy",
			Target:      "model",
			Parameters:  map[string]interface{}{"accuracy_target": feedback.Data},
			Priority:    s.calculateActionPriority(feedback),
			Status:      "pending",
			Created:     time.Now(),
		})
	}
	
	return actions
}

// calculateActionPriority calculates action priority
func (s *FeedbackServer) calculateActionPriority(feedback *ProcessedFeedback) int {
	priority := 5
	
	if feedback.Impact > 0.8 {
		priority += 3
	} else if feedback.Impact > 0.6 {
		priority += 1
	}
	
	if feedback.Type == "error" {
		priority += 2
	}
	
	if priority > 10 {
		priority = 10
	}
	
	return priority
}

// updateAnalytics updates analytics data
func (s *FeedbackServer) updateAnalytics(feedback *ProcessedFeedback) {
	s.analytics.mu.Lock()
	defer s.analytics.mu.Unlock()
	
	s.analytics.TotalFeedbacks++
	s.analytics.ByType[feedback.Type]++
	s.analytics.BySource[feedback.Source]++
	
	// Update average impact
	s.analytics.AvgImpact = (s.analytics.AvgImpact*(float64(s.analytics.TotalFeedbacks-1)) + feedback.Impact) / float64(s.analytics.TotalFeedbacks)
	
	// Add to trends
	if s.analytics.Trends["impact"] == nil {
		s.analytics.Trends["impact"] = make([]float64, 0)
	}
	s.analytics.Trends["impact"] = append(s.analytics.Trends["impact"], feedback.Impact)
	
	// Keep only last 100 trend points
	if len(s.analytics.Trends["impact"]) > 100 {
		s.analytics.Trends["impact"] = s.analytics.Trends["impact"][1:]
	}
}

// Background processing loops
func (s *FeedbackServer) processInsightsLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		s.mu.RLock()
		feedbackCount := len(s.feedbacks)
		s.mu.RUnlock()
		
		if feedbackCount > 0 {
			log.Printf("Processing insights for %d feedbacks", feedbackCount)
			// Additional insight processing could go here
		}
	}
}

func (s *FeedbackServer) executeActionsLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		s.mu.Lock()
		var pendingActions []*Action
		for _, action := range s.actions {
			if action.Status == "pending" {
				pendingActions = append(pendingActions, action)
			}
		}
		s.mu.Unlock()
		
		// Execute high priority actions
		for _, action := range pendingActions {
			if action.Priority >= 8 {
				s.executeAction(action)
			}
		}
	}
}

func (s *FeedbackServer) executeAction(action *Action) {
	log.Printf("Executing action: %s", action.Description)
	
	// Mock action execution
	time.Sleep(time.Second * time.Duration(action.Priority))
	
	s.mu.Lock()
	action.Status = "completed"
	s.mu.Unlock()
	
	log.Printf("Action completed: %s", action.ID)
}

// HTTP handlers
func (s *FeedbackServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	stats := map[string]interface{}{
		"status":     "healthy",
		"feedbacks":  len(s.feedbacks),
		"insights":   len(s.insights),
		"actions":    len(s.actions),
		"timestamp":  time.Now().Unix(),
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (s *FeedbackServer) handleInsights(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	insights := make([]*Insight, 0, len(s.insights))
	for _, insight := range s.insights {
		insights = append(insights, insight)
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"insights":  insights,
		"count":     len(insights),
		"timestamp": time.Now().Unix(),
	})
}

func (s *FeedbackServer) handleActions(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	actions := make([]*Action, 0, len(s.actions))
	for _, action := range s.actions {
		actions = append(actions, action)
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"actions":   actions,
		"count":     len(actions),
		"timestamp": time.Now().Unix(),
	})
}

func (s *FeedbackServer) handleAnalytics(w http.ResponseWriter, r *http.Request) {
	s.analytics.mu.RLock()
	data := *s.analytics
	s.analytics.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

// Utility functions
func generateID() string {
	return fmt.Sprintf("fb_%d", time.Now().UnixNano())
}

// main function to run the feedback server
func main() {
	port := 8004
	if len(os.Args) > 1 {
		if p, err := strconv.Atoi(os.Args[1]); err == nil {
			port = p
		}
	}

	server := NewFeedbackServer(port)
	
	// Start server
	if err := server.Start(); err != nil {
		log.Fatalf("Failed to start Feedback server: %v", err)
	}
}