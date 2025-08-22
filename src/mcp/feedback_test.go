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

func createTestFeedbackServer() *FeedbackServer {
	return NewFeedbackServer(8004)
}

func createTestFeedbackRequest() FeedbackRequest {
	return FeedbackRequest{
		Type:      "performance",
		Source:    "training_loop",
		Target:    "model_optimizer",
		Data:      map[string]interface{}{"latency_ms": 95, "accuracy": 0.85},
		Priority:  "high",
		ID:        "feedback_001",
		Timestamp: time.Now().Unix(),
	}
}

func TestFeedbackServerCreation(t *testing.T) {
	server := createTestFeedbackServer()
	
	if server.port != 8004 {
		t.Errorf("Expected port 8004, got %d", server.port)
	}
	
	if server.feedbacks == nil {
		t.Error("Feedbacks map should be initialized")
	}
	
	if server.insights == nil {
		t.Error("Insights map should be initialized")
	}
	
	if server.actions == nil {
		t.Error("Actions map should be initialized")
	}
	
	if server.analytics == nil {
		t.Error("Analytics should be initialized")
	}
}

func TestFeedbackRequestValidation(t *testing.T) {
	tests := []struct {
		name    string
		request FeedbackRequest
		valid   bool
	}{
		{
			name:    "Valid feedback request",
			request: createTestFeedbackRequest(),
			valid:   true,
		},
		{
			name: "Missing type",
			request: FeedbackRequest{
				Source: "test_source",
				Data:   map[string]interface{}{"value": 1},
			},
			valid: false,
		},
		{
			name: "Missing source",
			request: FeedbackRequest{
				Type: "performance",
				Data: map[string]interface{}{"value": 1},
			},
			valid: false,
		},
		{
			name: "Missing data",
			request: FeedbackRequest{
				Type:   "performance",
				Source: "test_source",
			},
			valid: false,
		},
		{
			name: "Invalid priority",
			request: FeedbackRequest{
				Type:     "performance",
				Source:   "test_source",
				Data:     map[string]interface{}{"value": 1},
				Priority: "invalid_priority",
			},
			valid: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateFeedbackRequest(&tt.request)
			if tt.valid && err != nil {
				t.Errorf("Expected valid request, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected invalid request, got no error")
			}
		})
	}
}

func TestFeedbackProcessing(t *testing.T) {
	server := createTestFeedbackServer()
	
	request := createTestFeedbackRequest()
	response := server.processFeedback(&request)
	
	if !response.Success {
		t.Errorf("Expected successful processing, got error: %s", response.Error)
	}
	
	if response.FeedbackID == "" {
		t.Error("Expected feedback ID in response")
	}
	
	if response.Timestamp <= 0 {
		t.Error("Expected positive timestamp")
	}
	
	if response.Duration <= 0 {
		t.Error("Expected positive duration")
	}
	
	// Check if feedback was stored
	server.mu.RLock()
	_, exists := server.feedbacks[response.FeedbackID]
	server.mu.RUnlock()
	
	if !exists {
		t.Error("Expected feedback to be stored")
	}
}

func TestFeedbackCategorization(t *testing.T) {
	server := createTestFeedbackServer()
	
	tests := []struct {
		feedbackType     string
		expectedCategory string
	}{
		{"performance", "optimization"},
		{"error", "bug_fix"},
		{"accuracy", "model_improvement"},
		{"latency", "performance"},
		{"memory", "resource_optimization"},
		{"unknown_type", "general"},
	}
	
	for _, tt := range tests {
		t.Run(tt.feedbackType, func(t *testing.T) {
			category := server.categorize(tt.feedbackType)
			if category != tt.expectedCategory {
				t.Errorf("Expected category '%s', got '%s'", tt.expectedCategory, category)
			}
		})
	}
}

func TestImpactCalculation(t *testing.T) {
	server := createTestFeedbackServer()
	
	tests := []struct {
		name           string
		request        FeedbackRequest
		expectedImpact float64
		tolerance      float64
	}{
		{
			name: "High priority error",
			request: FeedbackRequest{
				Type:     "error",
				Priority: "critical",
				Data:     map[string]interface{}{},
			},
			expectedImpact: 1.0, // Should be capped at 1.0
			tolerance:      0.1,
		},
		{
			name: "Low priority performance",
			request: FeedbackRequest{
				Type:     "performance",
				Priority: "low",
				Data:     map[string]interface{}{},
			},
			expectedImpact: 0.49, // 0.7 * 0.7
			tolerance:      0.1,
		},
		{
			name: "Medium accuracy feedback",
			request: FeedbackRequest{
				Type: "accuracy",
				Data: map[string]interface{}{},
			},
			expectedImpact: 0.9,
			tolerance:      0.1,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			impact := server.calculateImpact(&tt.request)
			if abs(impact-tt.expectedImpact) > tt.tolerance {
				t.Errorf("Expected impact ~%.2f, got %.2f", tt.expectedImpact, impact)
			}
		})
	}
}

func TestConfidenceCalculation(t *testing.T) {
	server := createTestFeedbackServer()
	
	tests := []struct {
		name               string
		request            FeedbackRequest
		expectedConfidence float64
		tolerance          float64
	}{
		{
			name: "Automated test source",
			request: FeedbackRequest{
				Type:   "performance",
				Source: "test_source",
				Data:   map[string]interface{}{"source_type": "automated_test"},
			},
			expectedConfidence: 0.9,
			tolerance:          0.1,
		},
		{
			name: "Monitoring source",
			request: FeedbackRequest{
				Type:   "performance",
				Source: "monitoring_system",
				Data:   map[string]interface{}{"source_type": "monitoring"},
			},
			expectedConfidence: 0.8,
			tolerance:          0.1,
		},
		{
			name: "User report source",
			request: FeedbackRequest{
				Type:   "error",
				Source: "user_report",
				Data:   map[string]interface{}{"source_type": "user_report"},
			},
			expectedConfidence: 0.6,
			tolerance:          0.1,
		},
		{
			name: "Default confidence",
			request: FeedbackRequest{
				Type:   "performance",
				Source: "unknown_source",
				Data:   map[string]interface{}{},
			},
			expectedConfidence: 0.7,
			tolerance:          0.1,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			confidence := server.calculateConfidence(&tt.request)
			if abs(confidence-tt.expectedConfidence) > tt.tolerance {
				t.Errorf("Expected confidence ~%.2f, got %.2f", tt.expectedConfidence, confidence)
			}
		})
	}
}

func TestInsightGeneration(t *testing.T) {
	server := createTestFeedbackServer()
	
	// High impact feedback should generate insights
	highImpactFeedback := &ProcessedFeedback{
		ID:         "high_impact_001",
		Type:       "error",
		Source:     "training_loop",
		Category:   "bug_fix",
		Impact:     0.8,
		Confidence: 0.9,
		Data:       map[string]interface{}{"severity": "critical"},
		Processed:  time.Now(),
	}
	
	insights := server.generateInsights(highImpactFeedback)
	
	if len(insights) == 0 {
		t.Error("Expected insights for high impact feedback")
	}
	
	// Check for high impact insight
	hasHighImpactInsight := false
	for _, insight := range insights {
		if insight.Type == "high_impact" {
			hasHighImpactInsight = true
			break
		}
	}
	
	if !hasHighImpactInsight {
		t.Error("Expected high impact insight to be generated")
	}
	
	// Error feedback should generate error pattern insight
	errorFeedback := &ProcessedFeedback{
		Type:   "error",
		Impact: 0.5,
	}
	
	errorInsights := server.generateInsights(errorFeedback)
	hasErrorPattern := false
	for _, insight := range errorInsights {
		if insight.Type == "error_pattern" {
			hasErrorPattern = true
			break
		}
	}
	
	if !hasErrorPattern {
		t.Error("Expected error pattern insight for error feedback")
	}
}

func TestActionGeneration(t *testing.T) {
	server := createTestFeedbackServer()
	
	tests := []struct {
		name               string
		feedback           *ProcessedFeedback
		expectedActionType string
	}{
		{
			name: "Performance feedback",
			feedback: &ProcessedFeedback{
				ID:       "perf_001",
				Type:     "performance",
				Source:   "training_loop",
				Impact:   0.7,
				Data:     map[string]interface{}{"latency_ms": 150},
			},
			expectedActionType: "optimize",
		},
		{
			name: "Error feedback",
			feedback: &ProcessedFeedback{
				ID:     "error_001",
				Type:   "error",
				Source: "model_inference",
				Impact: 0.9,
				Data:   map[string]interface{}{"error_type": "memory_overflow"},
			},
			expectedActionType: "debug",
		},
		{
			name: "Accuracy feedback",
			feedback: &ProcessedFeedback{
				ID:     "acc_001",
				Type:   "accuracy",
				Source: "validation_loop",
				Impact: 0.8,
				Data:   map[string]interface{}{"current_accuracy": 0.75},
			},
			expectedActionType: "retrain",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := server.generateActions(tt.feedback, []Insight{})
			
			if len(actions) == 0 {
				t.Error("Expected at least one action to be generated")
			}
			
			hasExpectedAction := false
			for _, action := range actions {
				if action.Type == tt.expectedActionType {
					hasExpectedAction = true
					break
				}
			}
			
			if !hasExpectedAction {
				t.Errorf("Expected action type '%s' to be generated", tt.expectedActionType)
			}
		})
	}
}

func TestActionPriorityCalculation(t *testing.T) {
	server := createTestFeedbackServer()
	
	tests := []struct {
		name             string
		feedback         *ProcessedFeedback
		expectedPriority int
	}{
		{
			name: "High impact error",
			feedback: &ProcessedFeedback{
				Type:   "error",
				Impact: 0.9,
			},
			expectedPriority: 10, // 5 + 3 + 2 = 10 (capped)
		},
		{
			name: "Medium impact performance",
			feedback: &ProcessedFeedback{
				Type:   "performance",
				Impact: 0.7,
			},
			expectedPriority: 6, // 5 + 1 = 6
		},
		{
			name: "Low impact general",
			feedback: &ProcessedFeedback{
				Type:   "general",
				Impact: 0.3,
			},
			expectedPriority: 5, // 5 (base priority)
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			priority := server.calculateActionPriority(tt.feedback)
			if priority != tt.expectedPriority {
				t.Errorf("Expected priority %d, got %d", tt.expectedPriority, priority)
			}
		})
	}
}

func TestAnalyticsUpdate(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Process some feedback to update analytics
	feedbacks := []*ProcessedFeedback{
		{Type: "performance", Source: "training", Impact: 0.8},
		{Type: "error", Source: "inference", Impact: 0.9},
		{Type: "performance", Source: "training", Impact: 0.6},
	}
	
	for _, feedback := range feedbacks {
		server.updateAnalytics(feedback)
	}
	
	server.analytics.mu.RLock()
	defer server.analytics.mu.RUnlock()
	
	// Check total feedbacks
	if server.analytics.TotalFeedbacks != 3 {
		t.Errorf("Expected 3 total feedbacks, got %d", server.analytics.TotalFeedbacks)
	}
	
	// Check by type counts
	if server.analytics.ByType["performance"] != 2 {
		t.Errorf("Expected 2 performance feedbacks, got %d", server.analytics.ByType["performance"])
	}
	
	if server.analytics.ByType["error"] != 1 {
		t.Errorf("Expected 1 error feedback, got %d", server.analytics.ByType["error"])
	}
	
	// Check by source counts
	if server.analytics.BySource["training"] != 2 {
		t.Errorf("Expected 2 training feedbacks, got %d", server.analytics.BySource["training"])
	}
	
	// Check average impact
	expectedAvgImpact := (0.8 + 0.9 + 0.6) / 3.0
	if abs(server.analytics.AvgImpact-expectedAvgImpact) > 0.01 {
		t.Errorf("Expected avg impact %.3f, got %.3f", expectedAvgImpact, server.analytics.AvgImpact)
	}
	
	// Check trends
	if len(server.analytics.Trends["impact"]) != 3 {
		t.Errorf("Expected 3 impact trend points, got %d", len(server.analytics.Trends["impact"]))
	}
}

func TestFeedbackRequestHandler(t *testing.T) {
	server := createTestFeedbackServer()
	
	request := createTestFeedbackRequest()
	requestBody, _ := json.Marshal(request)
	
	req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	server.handleFeedbackRequest(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response FeedbackResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal response: %v", err)
	}
	
	if !response.Success {
		t.Errorf("Expected successful response, got error: %s", response.Error)
	}
	
	if response.FeedbackID == "" {
		t.Error("Expected feedback ID in response")
	}
}

func TestFeedbackHealthEndpoint(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Process some feedback first
	feedback := &ProcessedFeedback{Type: "test", Source: "test", Impact: 0.5}
	server.mu.Lock()
	server.feedbacks["test"] = feedback
	server.mu.Unlock()
	
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
	
	if health["feedbacks"].(float64) != 1 {
		t.Error("Expected 1 feedback in health stats")
	}
}

func TestInsightsEndpoint(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Add test insight
	insight := &Insight{
		ID:          "test_insight",
		Type:        "test_type",
		Description: "test description",
		Confidence:  0.8,
		Priority:    5,
		Generated:   time.Now(),
	}
	
	server.mu.Lock()
	server.insights["test_insight"] = insight
	server.mu.Unlock()
	
	req := httptest.NewRequest("GET", "/insights", nil)
	w := httptest.NewRecorder()
	
	server.handleInsights(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal insights response: %v", err)
	}
	
	insights, ok := response["insights"].([]interface{})
	if !ok {
		t.Error("Expected insights array in response")
	}
	
	if len(insights) != 1 {
		t.Errorf("Expected 1 insight, got %d", len(insights))
	}
}

func TestActionsEndpoint(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Add test action
	action := &Action{
		ID:          "test_action",
		Type:        "optimize",
		Description: "test optimization",
		Target:      "model",
		Priority:    8,
		Status:      "pending",
		Created:     time.Now(),
	}
	
	server.mu.Lock()
	server.actions["test_action"] = action
	server.mu.Unlock()
	
	req := httptest.NewRequest("GET", "/actions", nil)
	w := httptest.NewRecorder()
	
	server.handleActions(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	if err != nil {
		t.Errorf("Failed to unmarshal actions response: %v", err)
	}
	
	actions, ok := response["actions"].([]interface{})
	if !ok {
		t.Error("Expected actions array in response")
	}
	
	if len(actions) != 1 {
		t.Errorf("Expected 1 action, got %d", len(actions))
	}
}

func TestAnalyticsEndpoint(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Add some analytics data
	server.analytics.mu.Lock()
	server.analytics.TotalFeedbacks = 5
	server.analytics.ByType["performance"] = 3
	server.analytics.ByType["error"] = 2
	server.analytics.AvgImpact = 0.75
	server.analytics.mu.Unlock()
	
	req := httptest.NewRequest("GET", "/analytics", nil)
	w := httptest.NewRecorder()
	
	server.handleAnalytics(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var analytics Analytics
	err := json.Unmarshal(w.Body.Bytes(), &analytics)
	if err != nil {
		t.Errorf("Failed to unmarshal analytics response: %v", err)
	}
	
	if analytics.TotalFeedbacks != 5 {
		t.Errorf("Expected 5 total feedbacks, got %d", analytics.TotalFeedbacks)
	}
	
	if analytics.ByType["performance"] != 3 {
		t.Errorf("Expected 3 performance feedbacks, got %d", analytics.ByType["performance"])
	}
}

func TestConcurrentFeedbackProcessing(t *testing.T) {
	server := createTestFeedbackServer()
	
	// Test concurrent feedback processing doesn't cause race conditions
	done := make(chan bool, 10)
	
	for i := 0; i < 10; i++ {
		go func(id int) {
			request := FeedbackRequest{
				Type:      "performance",
				Source:    "test_source",
				Data:      map[string]interface{}{"id": id},
				Timestamp: time.Now().Unix(),
			}
			
			requestBody, _ := json.Marshal(request)
			req := httptest.NewRequest("POST", "/", bytes.NewBuffer(requestBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			
			server.handleFeedbackRequest(w, req)
			
			if w.Code != http.StatusOK {
				t.Errorf("Request %d: Expected status 200, got %d", id, w.Code)
			}
			
			done <- true
		}(i)
	}
	
	// Wait for all requests to complete
	for i := 0; i < 10; i++ {
		<-done
	}
	
	// Check that all feedbacks were stored
	server.mu.RLock()
	feedbackCount := len(server.feedbacks)
	server.mu.RUnlock()
	
	if feedbackCount != 10 {
		t.Errorf("Expected 10 feedbacks stored, got %d", feedbackCount)
	}
}

// Benchmarks
func BenchmarkFeedbackProcessing(b *testing.B) {
	server := createTestFeedbackServer()
	request := createTestFeedbackRequest()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.processFeedback(&request)
	}
}

func BenchmarkInsightGeneration(b *testing.B) {
	server := createTestFeedbackServer()
	feedback := &ProcessedFeedback{
		Type:   "performance",
		Impact: 0.8,
		Data:   map[string]interface{}{"latency": 95},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		server.generateInsights(feedback)
	}
}

// Helper functions for testing
func validateFeedbackRequest(req *FeedbackRequest) error {
	if req.Type == "" {
		return fmt.Errorf("type is required")
	}
	if req.Source == "" {
		return fmt.Errorf("source is required")
	}
	if req.Data == nil {
		return fmt.Errorf("data is required")
	}
	if req.Priority != "" {
		validPriorities := map[string]bool{"low": true, "medium": true, "high": true, "critical": true}
		if !validPriorities[req.Priority] {
			return fmt.Errorf("invalid priority")
		}
	}
	return nil
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}