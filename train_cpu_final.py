#!/usr/bin/env python3
import sys
sys.path.append('src')
import os
import json
import time
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from train_model import SubtitleNormalizationTrainer


def find_latest_checkpoint(base_dir: str) -> str:
    """Find the latest epoch checkpoint directory in base_dir."""
    if not os.path.isdir(base_dir):
        return ""
    candidates = []
    for name in os.listdir(base_dir):
        if name.startswith("checkpoint-epoch-"):
            try:
                epoch_num = int(name.split("checkpoint-epoch-")[-1])
                candidates.append((epoch_num, os.path.join(base_dir, name)))
            except ValueError:
                continue
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description="CPU training with resume and multithreading")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for CPU training")
    parser.add_argument("--threads", type=str, default="auto", help="Number of CPU threads to use (auto or integer)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--model_dir", type=str, default="models/subtitle_normalizer_cpu", help="Directory to save checkpoints and models")
    args = parser.parse_args()

    print("🚀 CPU Training - Optimized for Quality!")
    print("💻 Device: CPU (no GPU compatibility issues)")
    print("📦 Model: T5-small (excellent quality, CPU-friendly)")
    print("📊 Training data: 50,000 samples from your collection")
    print(f"🎯 Epochs: {args.epochs} (optimal for CPU)")
    print(f"📏 Batch size: {args.batch_size} (CPU memory optimized)")
    print("🏆 Expected result: Excellent quality model")
    print()

    # Configure CPU threading
    if args.threads == "auto":
        num_threads = max(1, os.cpu_count() or 1)
    else:
        try:
            num_threads = max(1, int(args.threads))
        except Exception:
            num_threads = max(1, os.cpu_count() or 1)
    # Set env hints for BLAS/OpenMP backends
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(min(num_threads, 8))
    except Exception:
        pass
    print(f"🧵 CPU threads: {num_threads}")

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
    print(f"📈 Real SDH pairs: {data['metadata'].get('real_sdh_pairs', 'n/a')}")
    print(f"📈 Case normalization pairs: {data['metadata'].get('case_normalization_pairs', 'n/a')}")

    # Create datasets
    print("🔧 Preparing datasets...")
    train_dataset, val_dataset = trainer.create_datasets(train_data, val_data)

    # CPU-optimized data loaders
    # Use a conservative number of workers on Windows; allow slight parallelism
    num_workers = max(0, min(4, num_threads - 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    # Setup optimizer and scheduler
    print("⚙️ Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-6
    )

    num_epochs = args.epochs
    num_training_steps_total = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps_total
    )

    print(f"\n🔥 CPU training configuration:")
    print(f"   📊 {len(train_loader)} training steps per epoch")
    print(f"   📊 {num_training_steps_total} total training steps")

    # Create models directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Resume logic
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    state_file = os.path.join(args.model_dir, "training_state.json")

    if args.resume:
        latest_ckpt = find_latest_checkpoint(args.model_dir)
        if latest_ckpt:
            print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            # Load model/tokenizer
            model.from_pretrained(latest_ckpt)
            tokenizer.from_pretrained(latest_ckpt)
            # Load optimizer/scheduler state if present
            opt_path = os.path.join(latest_ckpt, "optimizer.pt")
            sch_path = os.path.join(latest_ckpt, "scheduler.pt")
            if os.path.isfile(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            if os.path.isfile(sch_path):
                scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
            # Load training state (epoch/global_step/best)
            if os.path.isfile(state_file):
                try:
                    with open(state_file, 'r', encoding='utf-8') as sf:
                        st = json.load(sf)
                        start_epoch = int(st.get("last_completed_epoch", 0))
                        global_step = int(st.get("global_step", 0))
                        best_val_loss = float(st.get("best_val_loss", float('inf')))
                except Exception:
                    pass
            # Start after the last completed epoch
            print(f"➡️  Resuming at epoch {start_epoch + 1}, global_step {global_step}, best_val_loss {best_val_loss}")
        else:
            print("ℹ️  No checkpoint found to resume; starting fresh.")

    # Training loop with progress tracking
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        print(f"\n🏃‍♂️ Epoch {epoch + 1}/{num_epochs}")
        print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")

        epoch_loss = 0.0
        successful_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                successful_batches += 1
                global_step += 1

                avg_loss = epoch_loss / successful_batches
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step
                })

            except Exception as e:
                print(f"\n⚠️ Error at step {global_step}: {e}")
                optimizer.zero_grad()
                continue

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = (epoch_loss / successful_batches) if successful_batches else float('inf')
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   ✅ Average loss: {avg_epoch_loss:.4f}")
        print(f"   ⏱️ Epoch time: {epoch_time/60:.1f} minutes")
        print(f"   ✅ Successful batches: {successful_batches}/{len(train_loader)}")

        # Save checkpoint after each epoch (model, tokenizer, optimizer, scheduler, state)
        checkpoint_dir = os.path.join(args.model_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        # Persist training state (completed epoch, global_step, best_val_loss)
        with open(state_file, 'w', encoding='utf-8') as sf:
            json.dump({
                "last_completed_epoch": epoch + 0,  # current epoch completed index (0-based)
                "global_step": global_step,
                "best_val_loss": best_val_loss
            }, sf, indent=2)
        print(f"💾 Checkpoint saved: {checkpoint_dir}")

        # Quick validation
        print("🧪 Running validation...")
        model.eval()
        val_loss = 0.0
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
                except Exception:
                    continue

        avg_val_loss = (val_loss / val_steps) if val_steps else float('inf')
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dir = os.path.join(args.model_dir, "best")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"🏆 New best model saved! Loss: {avg_val_loss:.4f}")
            # Update state with best
            with open(state_file, 'w', encoding='utf-8') as sf:
                json.dump({
                    "last_completed_epoch": epoch + 0,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss
                }, sf, indent=2)

        model.train()

    # Save final model
    final_model_dir = os.path.join(args.model_dir, "final")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    total_time = time.time() - (os.path.getctime(state_file) if os.path.isfile(state_file) else time.time())
    print(f"\n✅ Training Complete!")
    print(f"💾 Final model: {final_model_dir}")
    print(f"🏆 Best model: {os.path.join(args.model_dir, 'best')}")

    # Test the model
    print("\n🧪 Testing your trained model...")
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    test_model_dir = os.path.join(args.model_dir, "best")
    test_tokenizer = T5Tokenizer.from_pretrained(test_model_dir)
    test_model = T5ForConditionalGeneration.from_pretrained(test_model_dir)
    test_model.eval()

    test_inputs = [
        "[MUSIC PLAYING]\nHello, how are you today?",
        "JOHN: I think we should go now.\n[door slams]",
        "♪ upbeat music ♪\nThis is a great day\nfor a picnic.",
        ">> NARRATOR: The story begins here.\n(sighs heavily)",
        "- SARAH: THIS IS ALL CAPS TEXT\nTHAT NEEDS FIXING!",
        "(PHONE BUZZING)\nWe need to talk about this.",
        "[THUNDER RUMBLING]\nThe storm is coming."
    ]

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
    print(f"📁 Use this model: {os.path.join(args.model_dir, 'best')}")
    print(f"🎬 You can now normalize any subtitle file!")

if __name__ == "__main__":
    main()
