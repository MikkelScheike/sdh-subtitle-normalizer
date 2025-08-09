#!/usr/bin/env python3
import sys
sys.path.append('src')
import torch
import torch_directml
from train_model import SubtitleNormalizationTrainer
import json
import os
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

def main():
    print("🚀 AMD GPU Training - DirectML Compatible!")
    print("🎮 GPU: AMD Radeon RX 9070 (16GB VRAM)")
    print("📦 Model: T5-base")
    print("📊 Training data: 50,000 samples")
    print("🎯 Epochs: 5")
    print("📏 Batch size: 4 (DirectML memory optimized)")
    print("⏱️ Estimated time: 45-60 minutes")
    print()

    # Set up DirectML device
    device = torch_directml.device()
    print(f"🔥 Using device: {device}")
    
    # Initialize trainer
    trainer = SubtitleNormalizationTrainer(model_name="t5-base")
    model = trainer.model.to(device)
    tokenizer = trainer.tokenizer
    
    # Load training data
    with open("training_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train']
    val_data = data['validation']
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    num_epochs = 5
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps
    )
    
    print(f"\n🔥 Starting training for {num_epochs} epochs...")
    print(f"📊 {len(train_loader)} training steps per epoch")
    print(f"📊 {num_training_steps} total training steps")
    
    # Create models directory
    os.makedirs("models/subtitle_normalizer_amd", exist_ok=True)
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n🏃‍♂️ Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log every 50 steps
            if global_step % 50 == 0:
                print(f"Step {global_step}: Loss = {loss.item():.4f}")
        
        # Validation at end of epoch
        print(f"\n📊 Epoch {epoch + 1} complete. Average loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save checkpoint
        checkpoint_dir = f"models/subtitle_normalizer_amd/checkpoint-epoch-{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"💾 Checkpoint saved to {checkpoint_dir}")
        
        # Quick validation
        if (epoch + 1) % 2 == 0:  # Validate every 2 epochs
            model.eval()
            val_loss = 0
            val_steps = 0
            
            print("🧪 Running validation...")
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)
                    
                    val_outputs = model(
                        input_ids=val_input_ids,
                        attention_mask=val_attention_mask,
                        labels=val_labels
                    )
                    
                    val_loss += val_outputs.loss.item()
                    val_steps += 1
                    
                    # Only validate on subset for speed
                    if val_steps >= 50:
                        break
            
            avg_val_loss = val_loss / val_steps
            print(f"📊 Validation Loss: {avg_val_loss:.4f}")
            
            model.train()
    
    # Save final model
    final_model_dir = "models/subtitle_normalizer_amd/final"
    os.makedirs(final_model_dir, exist_ok=True)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved to: {final_model_dir}")
    
    # Test the model
    print("\n🧪 Testing trained model...")
    
    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a great day\nfor a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)",
        "- SARAH: THIS IS ALL CAPS TEXT\nTHAT NEEDS FIXING!"
    ]
    
    model.eval()
    for i, test_input in enumerate(test_inputs):
        input_text = f"normalize subtitle: {test_input}"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        normalized = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n--- AMD GPU Test {i+1} ---")
        print(f"Input:  {test_input}")
        print(f"Output: {normalized}")
    
    print(f"\n🏆 AMD GPU Training Complete!")
    print(f"🎮 Trained on AMD RX 9070 with DirectML")
    print(f"📊 5 epochs with 50,000 training samples")
    print(f"💾 Model ready for subtitle normalization!")

if __name__ == "__main__":
    main()
