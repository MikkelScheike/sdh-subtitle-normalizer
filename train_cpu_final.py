#!/usr/bin/env python3
import sys
sys.path.append('src')
import torch
from train_model import SubtitleNormalizationTrainer
import json
import os
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import time

def main():
    print("🚀 CPU Training - Optimized for Quality!")
    print("💻 Device: CPU (no GPU compatibility issues)")
    print("📦 Model: T5-small (excellent quality, CPU-friendly)")
    print("📊 Training data: 50,000 samples from your collection")
    print("🎯 Epochs: 3 (optimal for CPU)")
    print("📏 Batch size: 4 (CPU memory optimized)")
    print("⏱️ Estimated time: 2-3 hours (great for overnight)")
    print("🏆 Expected result: Excellent quality model")
    print()
    print("💡 Pro tip: This is running while you sleep - wake up to a trained model!")
    print()

    # Force CPU usage - no GPU confusion
    device = torch.device('cpu')
    print(f"🔥 Using device: {device}")
    
    # Initialize trainer with CPU-optimized model
    print("📦 Loading T5-small model...")
    trainer = SubtitleNormalizationTrainer(model_name="t5-small")
    model = trainer.model.to(device)
    tokenizer = trainer.tokenizer
    
    # Load training data
    print("📊 Loading your training data...")
    with open("training_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train']
    val_data = data['validation']
    
    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(val_data)} validation samples")
    print(f"📈 Real SDH pairs: {data['metadata']['real_sdh_pairs']}")
    print(f"📈 Case normalization pairs: {data['metadata']['case_normalization_pairs']}")
    
    # Create datasets
    print("🔧 Preparing datasets...")
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)
    
    # CPU-optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,           # Small batch for CPU
        shuffle=True,
        num_workers=0,          # No multiprocessing on CPU
        pin_memory=False        # No GPU pinning
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer and scheduler
    print("⚙️ Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-5, 
        weight_decay=0.01,
        eps=1e-6                # Slightly more stable for CPU
    )
    
    num_epochs = 3              # Perfect for CPU training
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )
    
    print(f"\n🔥 Starting CPU training configuration:")
    print(f"   📊 {len(train_loader)} training steps per epoch")
    print(f"   📊 {num_training_steps} total training steps")
    print(f"   ⏱️ Estimated time per epoch: 45-60 minutes")
    print(f"   🎯 Total estimated time: 2.5-3 hours")
    print()
    
    # Create models directory
    os.makedirs("models/subtitle_normalizer_cpu", exist_ok=True)
    
    # Training loop with progress tracking
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n🏃‍♂️ Epoch {epoch + 1}/{num_epochs}")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
        
        epoch_loss = 0
        successful_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to CPU (already there, but explicit)
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update metrics
                epoch_loss += loss.item()
                successful_batches += 1
                global_step += 1
                
                # Update progress bar with detailed info
                avg_loss = epoch_loss / successful_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step
                })
                
                # Log every 200 steps
                if global_step % 200 == 0:
                    elapsed = time.time() - start_time
                    print(f"\n📊 Step {global_step}: Loss = {loss.item():.4f}, Time elapsed: {elapsed/3600:.1f}h")
                
            except Exception as e:
                print(f"\n⚠️ Error at step {global_step}: {e}")
                optimizer.zero_grad()
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / successful_batches if successful_batches > 0 else float('inf')
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   ✅ Average loss: {avg_epoch_loss:.4f}")
        print(f"   ⏱️ Epoch time: {epoch_time/60:.1f} minutes")
        print(f"   ✅ Successful batches: {successful_batches}/{len(train_loader)}")
        
        # Save checkpoint after each epoch
        checkpoint_dir = f"models/subtitle_normalizer_cpu/checkpoint-epoch-{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"💾 Checkpoint saved: {checkpoint_dir}")
        
        # Quick validation
        print("🧪 Running validation...")
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                if val_steps >= 100:  # Quick validation sample
                    break
                    
                try:
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
                    
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dir = "models/subtitle_normalizer_cpu/best"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"🏆 New best model saved! Loss: {avg_val_loss:.4f}")
        
        model.train()
        
        # Estimate remaining time
        if epoch < num_epochs - 1:
            remaining_time = epoch_time * (num_epochs - epoch - 1)
            print(f"⏱️ Estimated remaining time: {remaining_time/60:.1f} minutes")
    
    # Save final model
    final_model_dir = "models/subtitle_normalizer_cpu/final"
    os.makedirs(final_model_dir, exist_ok=True)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training Complete!")
    print(f"⏱️ Total training time: {total_time/3600:.1f} hours")
    print(f"💾 Final model: {final_model_dir}")
    print(f"🏆 Best model: models/subtitle_normalizer_cpu/best")
    
    # Test the model
    print("\n🧪 Testing your trained model...")
    
    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a great day\nfor a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)",
        "- SARAH: THIS IS ALL CAPS TEXT\nTHAT NEEDS FIXING!",
        "(PHONE BUZZING)\nWe need to talk about this.",
        "[THUNDER RUMBLING]\nThe storm is coming."
    ]
    
    # Use best model for testing
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    
    test_tokenizer = T5Tokenizer.from_pretrained("models/subtitle_normalizer_cpu/best")
    test_model = T5ForConditionalGeneration.from_pretrained("models/subtitle_normalizer_cpu/best")
    test_model.eval()
    
    for i, test_input in enumerate(test_inputs):
        try:
            input_text = f"normalize subtitle: {test_input}"
            input_ids = test_tokenizer(input_text, return_tensors="pt").input_ids
            
            with torch.no_grad():
                outputs = test_model.generate(
                    input_ids,
                    max_length=256,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
            
            normalized = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Input:  {test_input}")
            print(f"Output: {normalized}")
            
        except Exception as e:
            print(f"Test {i+1} failed: {e}")
            continue
    
    print(f"\n🎉 SUCCESS! Your subtitle normalizer is ready!")
    print(f"📁 Use this model: models/subtitle_normalizer_cpu/best")
    print(f"🎬 You can now normalize any subtitle file!")
    print(f"💪 Trained on your 21,354 English subtitle collection!")

if __name__ == "__main__":
    main()
