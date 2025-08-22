#!/usr/bin/env python3
"""
Dataset Download Script for ICE Project

This script downloads all the required datasets for the ICE (Intelligent Compact Engine) project:
- HumanEval: Code evaluation dataset
- TinyStories: Language understanding dataset  
- GSM8K: Mathematical reasoning dataset
- BabyLM: Language modeling dataset
- MiniPile: General knowledge dataset

Usage:
    python download_datasets.py
"""

import os
import sys
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create datasets directory
DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.chdir(DATASETS_DIR)

def download_humaneval():
    """Download HumanEval dataset for code evaluation"""
    logger.info("Downloading HumanEval dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("openai/humaneval")
        
        # Save to disk
        dataset.save_to_disk("humaneval")
        logger.info("HumanEval dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download HumanEval dataset: {e}")
        return False

def download_tinystories():
    """Download TinyStories dataset for language understanding"""
    logger.info("Downloading TinyStories dataset...")
    try:
        # Load a subset for efficiency (10MB as mentioned in docs)
        dataset = load_dataset("roneneldan/TinyStories", streaming=True)
        
        # Take a sample to keep it small
        train_dataset = dataset["train"].take(10000)  # Limit to 10k examples
        
        # Convert to regular dataset and save
        from datasets import Dataset
        examples = list(train_dataset)
        small_dataset = Dataset.from_list(examples)
        small_dataset.save_to_disk("tinystories")
        
        logger.info("TinyStories dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download TinyStories dataset: {e}")
        return False

def download_gsm8k():
    """Download GSM8K dataset for mathematical reasoning"""
    logger.info("Downloading GSM8K dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("openai/gsm8k", "main")
        
        # Save to disk
        dataset.save_to_disk("gsm8k")
        logger.info("GSM8K dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download GSM8K dataset: {e}")
        return False

def download_babylm():
    """Download BabyLM dataset for language modeling"""
    logger.info("Downloading BabyLM dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("cambridge-climb/BabyLM")
        
        # Save to disk
        dataset.save_to_disk("babylm")
        logger.info("BabyLM dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download BabyLM dataset: {e}")
        return False

def download_minipile():
    """Download MiniPile dataset for general knowledge"""
    logger.info("Downloading MiniPile dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("JeanKaddour/minipile")
        
        # Save to disk
        dataset.save_to_disk("minipile")
        logger.info("MiniPile dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download MiniPile dataset: {e}")
        return False

def main():
    """Main function to download all datasets"""
    logger.info("Starting dataset download process...")
    
    # Track success/failure
    results = {}
    
    # Download each dataset
    results["HumanEval"] = download_humaneval()
    results["TinyStories"] = download_tinystories()
    results["GSM8K"] = download_gsm8k()
    results["BabyLM"] = download_babylm()
    results["MiniPile"] = download_minipile()
    
    # Summary
    logger.info("Dataset download summary:")
    for dataset, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {dataset}: {status}")
    
    # Check if all downloads were successful
    if all(results.values()):
        logger.info("All datasets downloaded successfully!")
        return 0
    else:
        logger.warning("Some datasets failed to download. Check logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())