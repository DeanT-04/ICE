//! Training binary for the ultra-fast AI model
//!
//! Dedicated binary for training the model on datasets.

use anyhow::Result;
use clap::Parser;
use ultra_fast_ai::training::{TrainingConfig, Trainer};

#[derive(Parser)]
#[command(name = "train")]
#[command(about = "Train the ultra-fast AI model")]
struct Args {
    /// Dataset to train on
    #[arg(short, long, default_value = "mixed")]
    dataset: String,
    
    /// Number of epochs
    #[arg(short, long, default_value = "20")]
    epochs: u32,
    
    /// Batch size
    #[arg(short, long, default_value = "16")]
    batch_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let config = TrainingConfig {
        dataset_name: args.dataset,
        epochs: args.epochs,
        batch_size: args.batch_size,
        ..Default::default()
    };
    
    let mut trainer = Trainer::new(config)?;
    trainer.train().await?;
    
    println!("Training completed!");
    Ok(())
}