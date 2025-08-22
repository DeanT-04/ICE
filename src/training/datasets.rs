//! Dataset loaders for training data
//!
//! Provides unified interface for loading and preprocessing various datasets
//! including HumanEval, TinyStories, GSM8K, BabyLM, and MiniPile.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tokio::fs;
use crate::Result;

/// Training sample structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub id: String,
    pub input: String,
    pub target: String,
    pub metadata: HashMap<String, String>,
    pub dataset: String,
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub path: PathBuf,
    pub format: DatasetFormat,
    pub preprocessing: PreprocessingConfig,
    pub splits: DatasetSplits,
    pub max_samples: Option<usize>,
    pub shuffle: bool,
}

/// Dataset format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetFormat {
    Json,
    Jsonl,
    Csv,
    Txt,
    Parquet,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub max_length: usize,
    pub truncate: bool,
    pub pad_token: String,
    pub eos_token: String,
    pub bos_token: String,
    pub normalize_whitespace: bool,
    pub remove_special_chars: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            max_length: 2048,
            truncate: true,
            pad_token: "<pad>".to_string(),
            eos_token: "<eos>".to_string(),
            bos_token: "<bos>".to_string(),
            normalize_whitespace: true,
            remove_special_chars: false,
        }
    }
}

/// Dataset splits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplits {
    pub train: f32,
    pub validation: f32,
    pub test: f32,
}

impl Default for DatasetSplits {
    fn default() -> Self {
        Self {
            train: 0.8,
            validation: 0.1,
            test: 0.1,
        }
    }
}

/// Dataset split type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitType {
    Train,
    Validation,
    Test,
}

/// Unified dataset loader trait
pub trait DatasetLoader: Send + Sync {
    fn load_samples(&self, split: SplitType) -> Result<Vec<TrainingSample>>;
    fn get_info(&self) -> DatasetInfo;
    fn preprocess_sample(&self, sample: &mut TrainingSample) -> Result<()>;
    fn get_vocab_size(&self) -> usize;
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub description: String,
    pub size: usize,
    pub vocab_size: usize,
    pub avg_length: f32,
    pub license: String,
}

/// HumanEval dataset loader
pub struct HumanEvalLoader {
    config: DatasetConfig,
    samples: Vec<HumanEvalSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HumanEvalSample {
    task_id: String,
    prompt: String,
    canonical_solution: String,
    test: String,
    entry_point: String,
}

impl HumanEvalLoader {
    pub async fn new(config: DatasetConfig) -> Result<Self> {
        let samples = Self::load_humaneval_data(&config.path).await?;
        Ok(Self { config, samples })
    }

    async fn load_humaneval_data(path: &Path) -> Result<Vec<HumanEvalSample>> {
        let content = fs::read_to_string(path).await?;
        let samples: Vec<HumanEvalSample> = serde_json::from_str(&content)?;
        Ok(samples)
    }
}

impl DatasetLoader for HumanEvalLoader {
    fn load_samples(&self, split: SplitType) -> Result<Vec<TrainingSample>> {
        let (start, end) = self.get_split_indices(split);
        let mut samples = Vec::new();
        
        for (i, sample) in self.samples.iter().enumerate() {
            if i >= start && i < end {
                let mut metadata = HashMap::new();
                metadata.insert("task_id".to_string(), sample.task_id.clone());
                metadata.insert("entry_point".to_string(), sample.entry_point.clone());
                
                let training_sample = TrainingSample {
                    id: sample.task_id.clone(),
                    input: format!("{}\n# Complete the function", sample.prompt),
                    target: sample.canonical_solution.clone(),
                    metadata,
                    dataset: "humaneval".to_string(),
                };
                
                samples.push(training_sample);
            }
        }
        
        Ok(samples)
    }

    fn get_info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "HumanEval".to_string(),
            description: "Hand-written programming problems for code synthesis evaluation".to_string(),
            size: self.samples.len(),
            vocab_size: 50000, // Estimated for code
            avg_length: 150.0,
            license: "MIT".to_string(),
        }
    }

    fn preprocess_sample(&self, sample: &mut TrainingSample) -> Result<()> {
        if self.config.preprocessing.normalize_whitespace {
            sample.input = sample.input.split_whitespace().collect::<Vec<_>>().join(" ");
            sample.target = sample.target.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        if self.config.preprocessing.truncate {
            if sample.input.len() > self.config.preprocessing.max_length {
                sample.input.truncate(self.config.preprocessing.max_length);
            }
            if sample.target.len() > self.config.preprocessing.max_length {
                sample.target.truncate(self.config.preprocessing.max_length);
            }
        }
        
        Ok(())
    }

    fn get_vocab_size(&self) -> usize {
        50000 // Code vocabulary size
    }
}

impl HumanEvalLoader {
    fn get_split_indices(&self, split: SplitType) -> (usize, usize) {
        let total = self.samples.len();
        let train_end = (total as f32 * self.config.splits.train) as usize;
        let val_end = train_end + (total as f32 * self.config.splits.validation) as usize;
        
        match split {
            SplitType::Train => (0, train_end),
            SplitType::Validation => (train_end, val_end),
            SplitType::Test => (val_end, total),
        }
    }
}

/// TinyStories dataset loader
pub struct TinyStoriesLoader {
    config: DatasetConfig,
    samples: Vec<TinyStoriesSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyStoriesSample {
    story: String,
    summary: Option<String>,
}

impl TinyStoriesLoader {
    pub async fn new(config: DatasetConfig) -> Result<Self> {
        let samples = Self::load_tinystories_data(&config.path).await?;
        Ok(Self { config, samples })
    }

    async fn load_tinystories_data(path: &Path) -> Result<Vec<TinyStoriesSample>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();
        
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let sample: TinyStoriesSample = serde_json::from_str(&line)?;
                samples.push(sample);
            }
        }
        
        Ok(samples)
    }
}

impl DatasetLoader for TinyStoriesLoader {
    fn load_samples(&self, split: SplitType) -> Result<Vec<TrainingSample>> {
        let (start, end) = self.get_split_indices(split);
        let mut samples = Vec::new();
        
        for (i, sample) in self.samples.iter().enumerate() {
            if i >= start && i < end {
                let mut metadata = HashMap::new();
                if let Some(summary) = &sample.summary {
                    metadata.insert("summary".to_string(), summary.clone());
                }
                
                let training_sample = TrainingSample {
                    id: format!("tinystories_{}", i),
                    input: format!("Continue the story: {}", sample.story.split('.').next().unwrap_or("")),
                    target: sample.story.clone(),
                    metadata,
                    dataset: "tinystories".to_string(),
                };
                
                samples.push(training_sample);
            }
        }
        
        Ok(samples)
    }

    fn get_info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "TinyStories".to_string(),
            description: "Simple stories for language model training".to_string(),
            size: self.samples.len(),
            vocab_size: 30000,
            avg_length: 80.0,
            license: "MIT".to_string(),
        }
    }

    fn preprocess_sample(&self, sample: &mut TrainingSample) -> Result<()> {
        if self.config.preprocessing.normalize_whitespace {
            sample.input = sample.input.split_whitespace().collect::<Vec<_>>().join(" ");
            sample.target = sample.target.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        // Add BOS/EOS tokens
        sample.target = format!("{} {} {}", 
            self.config.preprocessing.bos_token,
            sample.target,
            self.config.preprocessing.eos_token
        );
        
        Ok(())
    }

    fn get_vocab_size(&self) -> usize {
        30000
    }
}

impl TinyStoriesLoader {
    fn get_split_indices(&self, split: SplitType) -> (usize, usize) {
        let total = self.samples.len();
        let train_end = (total as f32 * self.config.splits.train) as usize;
        let val_end = train_end + (total as f32 * self.config.splits.validation) as usize;
        
        match split {
            SplitType::Train => (0, train_end),
            SplitType::Validation => (train_end, val_end),
            SplitType::Test => (val_end, total),
        }
    }
}

/// GSM8K dataset loader
pub struct GSM8KLoader {
    config: DatasetConfig,
    samples: Vec<GSM8KSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GSM8KSample {
    question: String,
    answer: String,
}

impl GSM8KLoader {
    pub async fn new(config: DatasetConfig) -> Result<Self> {
        let samples = Self::load_gsm8k_data(&config.path).await?;
        Ok(Self { config, samples })
    }

    async fn load_gsm8k_data(path: &Path) -> Result<Vec<GSM8KSample>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();
        
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let sample: GSM8KSample = serde_json::from_str(&line)?;
                samples.push(sample);
            }
        }
        
        Ok(samples)
    }
}

impl DatasetLoader for GSM8KLoader {
    fn load_samples(&self, split: SplitType) -> Result<Vec<TrainingSample>> {
        let (start, end) = self.get_split_indices(split);
        let mut samples = Vec::new();
        
        for (i, sample) in self.samples.iter().enumerate() {
            if i >= start && i < end {
                let training_sample = TrainingSample {
                    id: format!("gsm8k_{}", i),
                    input: format!("Solve this math problem step by step: {}", sample.question),
                    target: sample.answer.clone(),
                    metadata: HashMap::new(),
                    dataset: "gsm8k".to_string(),
                };
                
                samples.push(training_sample);
            }
        }
        
        Ok(samples)
    }

    fn get_info(&self) -> DatasetInfo {
        DatasetInfo {
            name: "GSM8K".to_string(),
            description: "Grade School Math 8K problems".to_string(),
            size: self.samples.len(),
            vocab_size: 25000,
            avg_length: 120.0,
            license: "MIT".to_string(),
        }
    }

    fn preprocess_sample(&self, sample: &mut TrainingSample) -> Result<()> {
        if self.config.preprocessing.normalize_whitespace {
            sample.input = sample.input.split_whitespace().collect::<Vec<_>>().join(" ");
            sample.target = sample.target.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        Ok(())
    }

    fn get_vocab_size(&self) -> usize {
        25000
    }
}

impl GSM8KLoader {
    fn get_split_indices(&self, split: SplitType) -> (usize, usize) {
        let total = self.samples.len();
        let train_end = (total as f32 * self.config.splits.train) as usize;
        let val_end = train_end + (total as f32 * self.config.splits.validation) as usize;
        
        match split {
            SplitType::Train => (0, train_end),
            SplitType::Validation => (train_end, val_end),
            SplitType::Test => (val_end, total),
        }
    }
}

/// Generic text dataset loader (for BabyLM, MiniPile)
pub struct TextDatasetLoader {
    config: DatasetConfig,
    samples: Vec<String>,
}

impl TextDatasetLoader {
    pub async fn new(config: DatasetConfig) -> Result<Self> {
        let samples = Self::load_text_data(&config.path).await?;
        Ok(Self { config, samples })
    }

    async fn load_text_data(path: &Path) -> Result<Vec<String>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        
        // Split into samples (paragraphs or sentences)
        let samples: Vec<String> = content
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 20)
            .collect();
        
        Ok(samples)
    }
}

impl DatasetLoader for TextDatasetLoader {
    fn load_samples(&self, split: SplitType) -> Result<Vec<TrainingSample>> {
        let (start, end) = self.get_split_indices(split);
        let mut samples = Vec::new();
        
        for (i, text) in self.samples.iter().enumerate() {
            if i >= start && i < end {
                // Create language modeling samples
                let words: Vec<&str> = text.split_whitespace().collect();
                if words.len() > 10 {
                    let mid_point = words.len() / 2;
                    let input = words[..mid_point].join(" ");
                    let target = words[mid_point..].join(" ");
                    
                    let training_sample = TrainingSample {
                        id: format!("{}_{}", self.config.name, i),
                        input,
                        target,
                        metadata: HashMap::new(),
                        dataset: self.config.name.clone(),
                    };
                    
                    samples.push(training_sample);
                }
            }
        }
        
        Ok(samples)
    }

    fn get_info(&self) -> DatasetInfo {
        DatasetInfo {
            name: self.config.name.clone(),
            description: format!("{} text dataset", self.config.name),
            size: self.samples.len(),
            vocab_size: 32000,
            avg_length: 100.0,
            license: "Various".to_string(),
        }
    }

    fn preprocess_sample(&self, sample: &mut TrainingSample) -> Result<()> {
        if self.config.preprocessing.normalize_whitespace {
            sample.input = sample.input.split_whitespace().collect::<Vec<_>>().join(" ");
            sample.target = sample.target.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        if self.config.preprocessing.remove_special_chars {
            sample.input = sample.input.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-".contains(*c))
                .collect();
            sample.target = sample.target.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-".contains(*c))
                .collect();
        }
        
        Ok(())
    }

    fn get_vocab_size(&self) -> usize {
        32000
    }
}

impl TextDatasetLoader {
    fn get_split_indices(&self, split: SplitType) -> (usize, usize) {
        let total = self.samples.len();
        let train_end = (total as f32 * self.config.splits.train) as usize;
        let val_end = train_end + (total as f32 * self.config.splits.validation) as usize;
        
        match split {
            SplitType::Train => (0, train_end),
            SplitType::Validation => (train_end, val_end),
            SplitType::Test => (val_end, total),
        }
    }
}

/// Dataset manager for handling multiple datasets
pub struct DatasetManager {
    loaders: HashMap<String, Box<dyn DatasetLoader>>,
    configs: HashMap<String, DatasetConfig>,
}

impl DatasetManager {
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    pub async fn register_dataset(&mut self, name: &str, config: DatasetConfig) -> Result<()> {
        let loader: Box<dyn DatasetLoader> = match name {
            "humaneval" => Box::new(HumanEvalLoader::new(config.clone()).await?),
            "tinystories" => Box::new(TinyStoriesLoader::new(config.clone()).await?),
            "gsm8k" => Box::new(GSM8KLoader::new(config.clone()).await?),
            "babylm" | "minipile" => Box::new(TextDatasetLoader::new(config.clone()).await?),
            _ => return Err(crate::UltraFastAiError::DatasetError(format!("Unknown dataset: {}", name))),
        };

        self.loaders.insert(name.to_string(), loader);
        self.configs.insert(name.to_string(), config);
        Ok(())
    }

    pub fn get_samples(&self, dataset: &str, split: SplitType) -> Result<Vec<TrainingSample>> {
        match self.loaders.get(dataset) {
            Some(loader) => loader.load_samples(split),
            None => Err(crate::UltraFastAiError::DatasetError(format!("Dataset not found: {}", dataset))),
        }
    }

    pub fn get_mixed_samples(&self, split: SplitType, max_per_dataset: Option<usize>) -> Result<Vec<TrainingSample>> {
        let mut all_samples = Vec::new();
        
        for (name, loader) in &self.loaders {
            let mut samples = loader.load_samples(split)?;
            
            if let Some(max) = max_per_dataset {
                samples.truncate(max);
            }
            
            // Add dataset prefix to maintain diversity
            for sample in &mut samples {
                sample.input = format!("[{}] {}", name, sample.input);
            }
            
            all_samples.extend(samples);
        }
        
        if self.configs.values().any(|c| c.shuffle) {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            all_samples.shuffle(&mut thread_rng());
        }
        
        Ok(all_samples)
    }

    pub fn preprocess_samples(&self, samples: &mut [TrainingSample]) -> Result<()> {
        for sample in samples {
            if let Some(loader) = self.loaders.get(&sample.dataset) {
                loader.preprocess_sample(sample)?;
            }
        }
        Ok(())
    }

    pub fn get_dataset_info(&self, dataset: &str) -> Option<DatasetInfo> {
        self.loaders.get(dataset).map(|loader| loader.get_info())
    }

    pub fn get_total_vocab_size(&self) -> usize {
        self.loaders.values()
            .map(|loader| loader.get_vocab_size())
            .max()
            .unwrap_or(32000)
    }

    pub fn list_datasets(&self) -> Vec<String> {
        self.loaders.keys().cloned().collect()
    }
}

/// Helper function to create default dataset configurations
pub fn create_default_configs() -> HashMap<String, DatasetConfig> {
    let mut configs = HashMap::new();
    
    configs.insert("humaneval".to_string(), DatasetConfig {
        name: "humaneval".to_string(),
        path: PathBuf::from("data/humaneval.json"),
        format: DatasetFormat::Json,
        preprocessing: PreprocessingConfig::default(),
        splits: DatasetSplits::default(),
        max_samples: Some(164),
        shuffle: true,
    });
    
    configs.insert("tinystories".to_string(), DatasetConfig {
        name: "tinystories".to_string(),
        path: PathBuf::from("data/tinystories.jsonl"),
        format: DatasetFormat::Jsonl,
        preprocessing: PreprocessingConfig::default(),
        splits: DatasetSplits::default(),
        max_samples: Some(100000),
        shuffle: true,
    });
    
    configs.insert("gsm8k".to_string(), DatasetConfig {
        name: "gsm8k".to_string(),
        path: PathBuf::from("data/gsm8k.jsonl"),
        format: DatasetFormat::Jsonl,
        preprocessing: PreprocessingConfig::default(),
        splits: DatasetSplits::default(),
        max_samples: Some(8500),
        shuffle: true,
    });
    
    configs.insert("babylm".to_string(), DatasetConfig {
        name: "babylm".to_string(),
        path: PathBuf::from("data/babylm.txt"),
        format: DatasetFormat::Txt,
        preprocessing: PreprocessingConfig::default(),
        splits: DatasetSplits::default(),
        max_samples: Some(50000),
        shuffle: true,
    });
    
    configs.insert("minipile".to_string(), DatasetConfig {
        name: "minipile".to_string(),
        path: PathBuf::from("data/minipile.txt"),
        format: DatasetFormat::Txt,
        preprocessing: PreprocessingConfig::default(),
        splits: DatasetSplits::default(),
        max_samples: Some(100000),
        shuffle: true,
    });
    
    configs
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;

    #[tokio::test]
    async fn test_dataset_manager() {
        let dir = tempdir().unwrap();
        
        // Create mock dataset file
        let humaneval_path = dir.path().join("humaneval.json");
        let mut file = File::create(&humaneval_path).unwrap();
        writeln!(file, r#"[{{"task_id": "test", "prompt": "def test():", "canonical_solution": "return True", "test": "assert test()", "entry_point": "test"}}]"#).unwrap();
        
        let config = DatasetConfig {
            name: "humaneval".to_string(),
            path: humaneval_path,
            format: DatasetFormat::Json,
            preprocessing: PreprocessingConfig::default(),
            splits: DatasetSplits::default(),
            max_samples: None,
            shuffle: false,
        };
        
        let mut manager = DatasetManager::new();
        manager.register_dataset("humaneval", config).await.unwrap();
        
        let samples = manager.get_samples("humaneval", SplitType::Train).unwrap();
        assert!(!samples.is_empty());
        assert_eq!(samples[0].dataset, "humaneval");
    }

    #[test]
    fn test_preprocessing() {
        let config = PreprocessingConfig::default();
        let mut sample = TrainingSample {
            id: "test".to_string(),
            input: "  Hello    world  ".to_string(),
            target: "  Target   text  ".to_string(),
            metadata: HashMap::new(),
            dataset: "test".to_string(),
        };
        
        let loader = TextDatasetLoader {
            config: DatasetConfig {
                name: "test".to_string(),
                path: PathBuf::from("test"),
                format: DatasetFormat::Txt,
                preprocessing: config,
                splits: DatasetSplits::default(),
                max_samples: None,
                shuffle: false,
            },
            samples: vec!["test".to_string()],
        };
        
        loader.preprocess_sample(&mut sample).unwrap();
        assert_eq!(sample.input, "Hello world");
        assert_eq!(sample.target, "Target text");
    }
}