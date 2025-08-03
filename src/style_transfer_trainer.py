#!/usr/bin/env python3
"""
Style Transfer Model Trainer
Trains a model to apply clean subtitle style to any input
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import os

class StyleTransferDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # For style transfer, we use different prompts based on type
        if item['type'] == 'clean_reference':
            # Clean examples: "maintain clean style: [clean text]"
            input_text = f"maintain clean style: {item['input']}"
        else:
            # Contrastive examples: "clean subtitle style: [dirty text]"
            input_text = f"clean subtitle style: {item['input']}"
        
        # Tokenize input
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

class StyleTransferTrainer:
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add style transfer specific tokens
        special_tokens = ['<CLEAN_STYLE>', '<MAINTAIN_STYLE>', '<APPLY_STYLE>']
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def load_style_transfer_data(self, data_file):
        """Load style transfer training data"""
        print(f"Loading style transfer data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        train_data = data['train']
        val_data = data['validation']
        clean_patterns = data['clean_patterns']
        metadata = data['metadata']
        
        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(val_data)} validation examples")
        print(f"Clean references: {metadata['clean_references']}")
        print(f"Contrastive examples: {metadata['contrastive_examples']}")
        print(f"Learned from {clean_patterns['stats']['total_clean_examples']} clean examples")
        
        return train_data, val_data, clean_patterns, metadata
    
    def create_datasets(self, train_data, val_data):
        """Create PyTorch datasets for style transfer"""
        train_dataset = StyleTransferDataset(train_data, self.tokenizer)
        val_dataset = StyleTransferDataset(val_data, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Style transfer specific metrics
        exact_match = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        # Length similarity (clean subtitles tend to be shorter)
        pred_lengths = [len(pred.strip()) for pred in decoded_preds]
        label_lengths = [len(label.strip()) for label in decoded_labels]
        
        avg_pred_length = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        avg_label_length = sum(label_lengths) / len(label_lengths) if label_lengths else 0
        
        length_ratio = avg_pred_length / avg_label_length if avg_label_length > 0 else 1.0
        
        return {
            "exact_match": exact_match,
            "length_ratio": length_ratio,
            "avg_pred_length": avg_pred_length,
            "avg_label_length": avg_label_length
        }
    
    def train_style_transfer_model(self, train_dataset, val_dataset, clean_patterns, 
                                  output_dir="models/style_transfer_normalizer", 
                                  num_epochs=5, batch_size=8, learning_rate=3e-5):
        """Train the style transfer model"""
        
        # Training arguments optimized for style transfer
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_exact_match",
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,  # Effective batch size = 16
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
        
        print("Starting style transfer training...")
        trainer.train()
        
        # Save the model and patterns
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save the learned clean patterns for inference
        with open(os.path.join(output_dir, 'clean_patterns.json'), 'w') as f:
            json.dump(clean_patterns, f, indent=2)
        
        print(f"Style transfer model saved to {output_dir}")
        
        return trainer
    
    def test_style_transfer(self, model_path, test_inputs):
        """Test the style transfer model"""
        print(f"Loading style transfer model from {model_path}")
        
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        print("Testing style transfer model:")
        
        for i, test_input in enumerate(test_inputs):
            # Use the style transfer prompt
            input_text = f"clean subtitle style: {test_input}"
            
            # Tokenize
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            
            # Generate with style-aware parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=256,  # Shorter for cleaner output
                    num_beams=5,     # More beams for better quality
                    length_penalty=1.2,  # Prefer shorter outputs
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.1
                )
            
            # Decode
            cleaned = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Style Transfer Test {i+1} ---")
            print(f"Dirty Input: {test_input}")
            print(f"Clean Output: {cleaned}")

def main():
    # Initialize trainer
    trainer = StyleTransferTrainer(model_name="t5-small")
    
    # Load style transfer data
    data_file = input("Enter path to style transfer data file: ").strip()
    if not data_file:
        data_file = "data/style_transfer_data.json"
    
    if not os.path.exists(data_file):
        print(f"Style transfer data file {data_file} does not exist!")
        print("Please run style_transfer_generator.py first.")
        return
    
    train_data, val_data, clean_patterns, metadata = trainer.load_style_transfer_data(data_file)
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)
    
    # Set training parameters
    output_dir = input("Enter output directory (default: models/style_transfer_normalizer): ").strip()
    if not output_dir:
        output_dir = "models/style_transfer_normalizer"
    
    epochs_input = input("Enter number of epochs (default: 5): ").strip()
    num_epochs = int(epochs_input) if epochs_input else 5
    
    batch_size_input = input("Enter batch size (default: 8): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 8
    
    # Train model
    trained_model = trainer.train_style_transfer_model(
        train_dataset, 
        val_dataset, 
        clean_patterns,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Test on sample inputs
    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a beautiful day for a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)",
        "Normal subtitle that should stay clean."
    ]
    
    print("\nTesting style transfer model...")
    trainer.test_style_transfer(output_dir, test_inputs)
    
    print(f"\nStyle transfer training complete!")
    print(f"Model saved to: {output_dir}")
    print("\nYour model now learned from clean examples and can apply that style to dirty subtitles!")

if __name__ == "__main__":
    main()
