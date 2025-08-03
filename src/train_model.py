#!/usr/bin/env python3
"""
Train T5 Model for SDH Subtitle Normalization
Uses Hugging Face transformers to train a text-to-text model
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.metrics import accuracy_score
import os

class SubtitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        input_text = f"normalize subtitle: {item['input']}"
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            item['output'],
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class SubtitleNormalizationTrainer:
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = ['<SDH>', '<MUSIC>', '<SOUND>', '<SPEAKER>']
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def load_training_data(self, data_file):
        """Load training data from JSON file"""
        print(f"Loading training data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = data['train']
        val_data = data['validation']
        metadata = data['metadata']
        
        print(f"Loaded {len(train_data)} training samples")
        print(f"Loaded {len(val_data)} validation samples")
        print(f"Real SDH pairs: {metadata['real_sdh_pairs']}")
        print(f"Synthetic pairs: {metadata['synthetic_pairs']}")
        
        return train_data, val_data, metadata
    
    def create_datasets(self, train_data, val_data):
        """Create PyTorch datasets"""
        train_dataset = SubtitleDataset(train_data, self.tokenizer)
        val_dataset = SubtitleDataset(val_data, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple accuracy metric (exact match)
        accuracy = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        return {"accuracy": accuracy}
    
    def train_model(self, train_dataset, val_dataset, output_dir="./subtitle_normalizer", 
                   num_epochs=3, batch_size=8, learning_rate=5e-5):
        """Train the model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def test_model(self, model_path, test_inputs):
        """Test the trained model on sample inputs"""
        print(f"Loading model from {model_path}")
        
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        print("Testing model on sample inputs:")
        
        for i, test_input in enumerate(test_inputs):
            input_text = f"normalize subtitle: {test_input}"
            
            # Tokenize
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
            
            # Decode
            normalized = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Input: {test_input}")
            print(f"Output: {normalized}")

def main():
    # Initialize trainer
    trainer = SubtitleNormalizationTrainer(model_name="t5-small")  # You can use t5-base for better results
    
    # Load training data
    data_file = input("Enter path to training data JSON file: ").strip()
    if not data_file:
        data_file = "training_data.json"
    
    if not os.path.exists(data_file):
        print(f"Training data file {data_file} does not exist!")
        print("Please run training_data_generator.py first to create the training data.")
        return
    
    train_data, val_data, metadata = trainer.load_training_data(data_file)
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)
    
    # Set training parameters
    output_dir = input("Enter output directory for trained model (default: ./subtitle_normalizer): ").strip()
    if not output_dir:
        output_dir = "./subtitle_normalizer"
    
    epochs_input = input("Enter number of epochs (default: 3): ").strip()
    num_epochs = int(epochs_input) if epochs_input else 3
    
    batch_size_input = input("Enter batch size (default: 8): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 8
    
    # Train model
    trained_model = trainer.train_model(
        train_dataset, 
        val_dataset, 
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Test on sample inputs
    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a great day\nfor a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)"
    ]
    
    print("\nTesting trained model...")
    trainer.test_model(output_dir, test_inputs)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("You can now use the trained model to normalize SDH subtitles.")

if __name__ == "__main__":
    main()
