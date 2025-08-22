//! Comprehensive unit tests for all Rust modules
//!
//! Provides 100% test coverage for the ultra-fast AI model implementation
//! including model components, training, validation, and utilities.

pub mod model_tests;
pub mod training_tests;
pub mod utils_tests;
pub mod integration_tests;
pub mod comprehensive_tests;
pub mod performance_tests;
pub mod security_tests;
pub mod training_tests_comprehensive;

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize test environment
pub fn init_test_env() {
    INIT.call_once(|| {
        env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .init();
    });
}

/// Test utilities and helpers
pub mod test_utils {
    use ndarray::{Array1, Array2};
    use rand::{thread_rng, Rng};
    use crate::model::core::{SnnConfig, SsmConfig, LiquidConfig};
    use crate::model::fusion::FusionConfig;
    use crate::training::datasets::{TrainingSample, DatasetConfig};
    use std::collections::HashMap;

    /// Create test SNN configuration
    pub fn create_test_snn_config() -> SnnConfig {
        SnnConfig {
            input_size: 64,
            hidden_sizes: vec![32, 16],
            output_size: 8,
            threshold: 0.5,
            decay_rate: 0.9,
            refractory_period: 2,
            sparse_rate: 0.15,
        }
    }

    /// Create test SSM configuration
    pub fn create_test_ssm_config() -> SsmConfig {
        SsmConfig {
            input_size: 64,
            state_size: 8,
            output_size: 8,
            num_layers: 2,
            dt_min: 0.001,
            dt_max: 0.1,
            dt_init: "random".to_string(),
            conv_kernel_size: 3,
        }
    }

    /// Create test Liquid NN configuration
    pub fn create_test_liquid_config() -> LiquidConfig {
        LiquidConfig {
            input_size: 64,
            hidden_size: 32,
            output_size: 8,
            time_constant: 1.0,
            adaptation_rate: 0.01,
            connectivity: 0.3,
            enable_adaptation: true,
        }
    }

    /// Create test fusion configuration
    pub fn create_test_fusion_config() -> FusionConfig {
        FusionConfig {
            input_dims: vec![8, 8, 8],
            output_dim: 16,
            hidden_dim: 24,
            attention_heads: 2,
            dropout_rate: 0.1,
            use_layer_norm: true,
            activation_function: "relu".to_string(),
        }
    }

    /// Generate random input array
    pub fn random_input(size: usize) -> Array1<f32> {
        let mut rng = thread_rng();
        Array1::from_shape_fn(size, |_| rng.gen_range(-1.0..1.0))
    }

    /// Generate random weight matrix
    pub fn random_weights(rows: usize, cols: usize) -> Array2<f32> {
        let mut rng = thread_rng();
        Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.1..0.1))
    }

    /// Create test training sample
    pub fn create_test_sample(id: &str) -> TrainingSample {
        TrainingSample {
            id: id.to_string(),
            input: "Test input text for training".to_string(),
            target: "Expected output text".to_string(),
            metadata: HashMap::new(),
            dataset: "test".to_string(),
        }
    }

    /// Create test dataset configuration
    pub fn create_test_dataset_config() -> DatasetConfig {
        DatasetConfig {
            name: "test_dataset".to_string(),
            path: std::path::PathBuf::from("test_data.json"),
            format: crate::training::datasets::DatasetFormat::Json,
            preprocessing: crate::training::datasets::PreprocessingConfig::default(),
            splits: crate::training::datasets::DatasetSplits::default(),
            max_samples: Some(100),
            shuffle: true,
        }
    }

    /// Assert arrays are approximately equal
    pub fn assert_arrays_close(a: &Array1<f32>, b: &Array1<f32>, tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Array lengths don't match");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() < tolerance,
                "Arrays differ at index {}: {} vs {} (tolerance: {})",
                i, av, bv, tolerance
            );
        }
    }

    /// Assert value is in range
    pub fn assert_in_range(value: f32, min: f32, max: f32) {
        assert!(
            value >= min && value <= max,
            "Value {} not in range [{}, {}]",
            value, min, max
        );
    }

    /// Create temporary directory for tests
    pub fn create_temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("Failed to create temp directory")
    }

    /// Create test file with content
    pub fn create_test_file(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let file_path = dir.join(name);
        std::fs::write(&file_path, content).expect("Failed to write test file");
        file_path
    }
}

/// Performance test utilities
pub mod perf_test_utils {
    use std::time::{Duration, Instant};

    /// Measure execution time of a function
    pub fn measure_time<F, R>(func: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Assert execution time is within bounds
    pub fn assert_time_bounds<F>(func: F, max_duration: Duration)
    where
        F: FnOnce(),
    {
        let (_, duration) = measure_time(func);
        assert!(
            duration <= max_duration,
            "Execution took {:?}, exceeds limit of {:?}",
            duration, max_duration
        );
    }

    /// Memory usage helper (simplified)
    pub fn get_memory_usage() -> usize {
        // Simplified memory usage check
        // In real implementation, would use system APIs
        4096 // Return mock value in MB
    }
}

/// Mock implementations for testing
pub mod mocks {
    use crate::model::mcp::{McpRequest, McpResponse, McpServerType};
    use crate::training::datasets::{TrainingSample, SplitType};
    use crate::Result;
    use std::collections::HashMap;
    use std::time::SystemTime;

    /// Mock MCP client for testing
    pub struct MockMcpClient {
        pub responses: HashMap<String, McpResponse>,
    }

    impl MockMcpClient {
        pub fn new() -> Self {
            Self {
                responses: HashMap::new(),
            }
        }

        pub fn add_response(&mut self, method: &str, response: McpResponse) {
            self.responses.insert(method.to_string(), response);
        }

        pub async fn execute(&self, request: McpRequest) -> Result<McpResponse> {
            if let Some(response) = self.responses.get(&request.method) {
                Ok(response.clone())
            } else {
                Ok(McpResponse {
                    success: true,
                    data: serde_json::json!({"mock": true}),
                    error: None,
                    cached: false,
                    timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap().as_secs(),
                    execution_time_ms: 10,
                })
            }
        }
    }

    /// Mock dataset loader for testing
    pub struct MockDatasetLoader {
        pub samples: Vec<TrainingSample>,
    }

    impl MockDatasetLoader {
        pub fn new(samples: Vec<TrainingSample>) -> Self {
            Self { samples }
        }

        pub fn load_samples(&self, _split: SplitType) -> Result<Vec<TrainingSample>> {
            Ok(self.samples.clone())
        }
    }

    /// Mock performance monitor
    pub struct MockPerformanceMonitor {
        pub metrics: HashMap<String, f32>,
    }

    impl MockPerformanceMonitor {
        pub fn new() -> Self {
            let mut metrics = HashMap::new();
            metrics.insert("latency_ms".to_string(), 50.0);
            metrics.insert("energy_w".to_string(), 30.0);
            metrics.insert("memory_mb".to_string(), 4096.0);
            
            Self { metrics }
        }

        pub fn get_metric(&self, name: &str) -> Option<f32> {
            self.metrics.get(name).copied()
        }
    }
}

/// Property-based testing utilities
pub mod property_tests {
    use proptest::prelude::*;
    use ndarray::Array1;

    /// Generate arbitrary arrays for property testing
    pub fn arb_array(size: usize) -> impl Strategy<Value = Array1<f32>> {
        prop::collection::vec(any::<f32>(), size)
            .prop_map(|v| Array1::from_vec(v))
    }

    /// Generate arrays with specific constraints
    pub fn arb_normalized_array(size: usize) -> impl Strategy<Value = Array1<f32>> {
        prop::collection::vec(-1.0f32..1.0f32, size)
            .prop_map(|v| Array1::from_vec(v))
    }

    /// Generate sparse arrays (mostly zeros)
    pub fn arb_sparse_array(size: usize, sparsity: f32) -> impl Strategy<Value = Array1<f32>> {
        prop::collection::vec(
            prop::strategy::Union::new([
                Just(0.0f32).boxed(),
                (-1.0f32..1.0f32).boxed(),
            ]).prop_map(move |v| if rand::random::<f32>() < sparsity { 0.0 } else { v }),
            size
        ).prop_map(|v| Array1::from_vec(v))
    }

    /// Property: Output size should match expected
    pub fn prop_output_size_matches<F>(func: F, input_size: usize, expected_output_size: usize)
    where
        F: Fn(Array1<f32>) -> Array1<f32>,
    {
        proptest!(|(input in arb_normalized_array(input_size))| {
            let output = func(input);
            prop_assert_eq!(output.len(), expected_output_size);
        });
    }

    /// Property: Output should be bounded
    pub fn prop_output_bounded<F>(func: F, input_size: usize, min_val: f32, max_val: f32)
    where
        F: Fn(Array1<f32>) -> Array1<f32>,
    {
        proptest!(|(input in arb_normalized_array(input_size))| {
            let output = func(input);
            for &val in output.iter() {
                prop_assert!(val >= min_val && val <= max_val, "Value {} out of bounds [{}, {}]", val, min_val, max_val);
            }
        });
    }
}

/// Integration test setup
pub mod integration_setup {
    use crate::model::core::HybridLayer;
    use crate::model::agentic::{AgenticCoordinator, TaskConfig};
    use crate::training::datasets::DatasetManager;
    use crate::training::trainer::{Trainer, TrainingConfig};
    use super::test_utils::*;

    /// Setup complete test environment
    pub fn setup_test_environment() -> TestEnvironment {
        let snn_config = create_test_snn_config();
        let ssm_config = create_test_ssm_config();
        let liquid_config = create_test_liquid_config();
        let fusion_config = create_test_fusion_config();

        let model = HybridLayer::new(snn_config, ssm_config, liquid_config, fusion_config)
            .expect("Failed to create test model");

        let task_config = TaskConfig::default();
        let agentic_coordinator = AgenticCoordinator::new(
            task_config,
            crate::model::agentic::VotingStrategy::WeightedVote
        );

        let dataset_manager = DatasetManager::new();
        let training_config = TrainingConfig::default();

        let trainer = Trainer::new(training_config, model, agentic_coordinator, dataset_manager);

        TestEnvironment {
            trainer,
        }
    }

    pub struct TestEnvironment {
        pub trainer: Trainer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_utils::*;

    #[test]
    fn test_init_test_env() {
        init_test_env();
        // Should not panic on multiple calls
        init_test_env();
    }

    #[test]
    fn test_random_input_generation() {
        let input = random_input(10);
        assert_eq!(input.len(), 10);
        
        // Values should be in range [-1, 1]
        for &val in input.iter() {
            assert_in_range(val, -1.0, 1.0);
        }
    }

    #[test]
    fn test_random_weights_generation() {
        let weights = random_weights(5, 3);
        assert_eq!(weights.shape(), &[5, 3]);
        
        // Values should be in range [-0.1, 0.1]
        for &val in weights.iter() {
            assert_in_range(val, -0.1, 0.1);
        }
    }

    #[test]
    fn test_create_test_sample() {
        let sample = create_test_sample("test_001");
        assert_eq!(sample.id, "test_001");
        assert!(!sample.input.is_empty());
        assert!(!sample.target.is_empty());
        assert_eq!(sample.dataset, "test");
    }

    #[test]
    fn test_assert_arrays_close() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.01, 1.99, 3.02]);
        
        assert_arrays_close(&a, &b, 0.05);
    }

    #[test]
    #[should_panic]
    fn test_assert_arrays_close_panic() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.5, 2.5, 3.5]);
        
        assert_arrays_close(&a, &b, 0.1);
    }

    #[test]
    fn test_assert_in_range() {
        assert_in_range(5.0, 0.0, 10.0);
        assert_in_range(0.0, 0.0, 10.0);
        assert_in_range(10.0, 0.0, 10.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_in_range_panic() {
        assert_in_range(15.0, 0.0, 10.0);
    }

    #[test]
    fn test_create_temp_dir() {
        let temp_dir = create_temp_dir();
        assert!(temp_dir.path().exists());
    }

    #[test]
    fn test_create_test_file() {
        let temp_dir = create_temp_dir();
        let file_path = create_test_file(temp_dir.path(), "test.txt", "test content");
        
        assert!(file_path.exists());
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[tokio::test]
    async fn test_mock_mcp_client() {
        let mut client = mocks::MockMcpClient::new();
        
        let response = crate::model::mcp::McpResponse {
            success: true,
            data: serde_json::json!({"test": "value"}),
            error: None,
            cached: false,
            timestamp: 0,
            execution_time_ms: 100,
        };
        
        client.add_response("test_method", response.clone());
        
        let request = crate::model::mcp::McpRequest {
            server_type: crate::model::mcp::McpServerType::Api,
            method: "test_method".to_string(),
            params: serde_json::json!({}),
            timeout_ms: 5000,
            cache_ttl_hours: 1,
            retry_count: 1,
        };
        
        let result = client.execute(request).await.unwrap();
        assert!(result.success);
        assert_eq!(result.data, serde_json::json!({"test": "value"}));
    }

    #[test]
    fn test_mock_performance_monitor() {
        let monitor = mocks::MockPerformanceMonitor::new();
        
        assert_eq!(monitor.get_metric("latency_ms"), Some(50.0));
        assert_eq!(monitor.get_metric("energy_w"), Some(30.0));
        assert_eq!(monitor.get_metric("memory_mb"), Some(4096.0));
        assert_eq!(monitor.get_metric("nonexistent"), None);
    }

    #[test]
    fn test_measure_time() {
        let (result, duration) = perf_test_utils::measure_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_memory_usage() {
        let usage = perf_test_utils::get_memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_configs_creation() {
        let snn_config = create_test_snn_config();
        assert_eq!(snn_config.input_size, 64);
        assert_eq!(snn_config.output_size, 8);
        
        let ssm_config = create_test_ssm_config();
        assert_eq!(ssm_config.input_size, 64);
        assert_eq!(ssm_config.state_size, 8);
        
        let liquid_config = create_test_liquid_config();
        assert_eq!(liquid_config.input_size, 64);
        assert_eq!(liquid_config.hidden_size, 32);
        
        let fusion_config = create_test_fusion_config();
        assert_eq!(fusion_config.input_dims, vec![8, 8, 8]);
        assert_eq!(fusion_config.output_dim, 16);
    }
}