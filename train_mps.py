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
    print("🚀 Apple Metal (MPS) Training - Optimized for Your Mac!")
    print("📦 Model: T5-base (good quality on MPS); adjust below if needed")
    print("📊 Training data: from training_data.json")
    print("🎯 Epochs: 3")
    print("📏 Batch size: 4 (safe default for MPS)")
    print()

    # Select best available device: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'mps'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'cuda'
    else:
        device = torch.device('cpu')
        device_name = 'cpu'

    print(f"🔥 Using device: {device_name}")

    # Initialize trainer and model
    model_name = os.environ.get("SDH_MODEL_NAME", "t5-base")
    print(f"📦 Loading model: {model_name}")
    trainer = SubtitleNormalizationTrainer(model_name=model_name)
    model = trainer.model.to(device)
    tokenizer = trainer.tokenizer

    # Load training data
    data_path = os.environ.get("SDH_TRAIN_DATA", "training_data.json")
    if not os.path.exists(data_path):
        print(f"❌ Training data file not found: {data_path}")
        print("Please generate it first.")
        return

    print(f"📊 Loading training data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = data['train']
    val_data = data['validation']

    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(val_data)} validation samples")

    # Create datasets
    print("🔧 Preparing datasets...")
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)

    # Data loaders
    per_device_batch_size = int(os.environ.get("SDH_BATCH_SIZE", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Optimizer and scheduler
    print("⚙️ Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(os.environ.get("SDH_LR", 5e-5)),
        weight_decay=0.01,
        eps=1e-6,
    )

    num_epochs = int(os.environ.get("SDH_EPOCHS", 3))
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(500, max(0, num_training_steps // 20)),
        num_training_steps=num_training_steps,
    )

    # Prepare output dirs
    out_root = "models/subtitle_normalizer_mps"
    os.makedirs(out_root, exist_ok=True)

    print("\n🔥 Starting training...")
    print(f"   📊 Steps per epoch: {len(train_loader)}")
    print(f"   📊 Total steps: {num_training_steps}")

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n🏃‍♂️ Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        successful_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Metrics
                epoch_loss += loss.item()
                successful_batches += 1
                global_step += 1

                avg_loss = epoch_loss / successful_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step,
                })
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

        # Save checkpoint
        checkpoint_dir = os.path.join(out_root, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"💾 Checkpoint saved: {checkpoint_dir}")

        # Quick validation
        print("🧪 Running validation...")
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                try:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    val_outputs = model(
                        input_ids=val_input_ids,
                        attention_mask=val_attention_mask,
                        labels=val_labels,
                    )
                    val_loss += val_outputs.loss.item()
                    val_steps += 1

                    # Limit validation steps for speed
                    if val_steps >= 100:
                        break
                except Exception:
                    continue

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dir = os.path.join(out_root, "best")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"🏆 New best model saved! Loss: {avg_val_loss:.4f}")

        model.train()

    # Save final model
    final_model_dir = os.path.join(out_root, "final")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete!")
    print(f"⏱️ Total training time: {total_time/3600:.1f} hours")
    print(f"💾 Final model: {final_model_dir}")
    print(f"🏆 Best model: {os.path.join(out_root, 'best')}")

    # Quick tests
    print("\n🧪 Testing your trained model...")
    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a great day\nfor a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)",
        "- SARAH: THIS IS ALL CAPS TEXT\nTHAT NEEDS FIXING!",
    ]

    from transformers import T5Tokenizer, T5ForConditionalGeneration
    test_tokenizer = T5Tokenizer.from_pretrained(os.path.join(out_root, "best"))
    test_model = T5ForConditionalGeneration.from_pretrained(os.path.join(out_root, "best")).to(device)
    test_model.eval()

    for i, test_input in enumerate(test_inputs):
        try:
            input_text = f"normalize subtitle: {test_input}"
            input_ids = test_tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                outputs = test_model.generate(
                    input_ids,
                    max_length=256,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True,
                )

            normalized = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n--- MPS Test {i+1} ---")
            print(f"Input:  {test_input}")
            print(f"Output: {normalized}")
        except Exception as e:
            print(f"Test {i+1} failed: {e}")
            continue

    print("\n🎉 SUCCESS! Your MPS-trained subtitle normalizer is ready!")
    print(f"📁 Use this model: {os.path.join(out_root, 'best')}")


if __name__ == "__main__":
    main()

