#!/usr/bin/env python3
"""
Style Transfer Subtitle Normalizer
Applies learned clean subtitle style to dirty subtitle files
"""

import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .srt_analyzer import SRTAnalyzer
import glob
import argparse

class StyleTransferNormalizer:
    def __init__(self, model_path):
        """Initialize with trained style transfer model"""
        print(f"Loading style transfer model from {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        
        # Load learned clean patterns
        patterns_file = os.path.join(model_path, 'clean_patterns.json')
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                self.clean_patterns = json.load(f)
            print(f"Loaded clean patterns: {self.clean_patterns['stats']['total_clean_examples']} examples")
        else:
            self.clean_patterns = None
            print("No clean patterns file found")
        
        self.analyzer = SRTAnalyzer()
        print("Style transfer normalizer ready!")
    
    def apply_clean_style(self, text):
        """Apply clean subtitle style to input text"""
        # Use style transfer prompt
        input_text = f"clean subtitle style: {text}"
        
        # Tokenize
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids
        
        # Generate with style-aware parameters
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=256,  # Clean subtitles are typically shorter
                num_beams=5,     # Better quality
                length_penalty=1.2,  # Prefer shorter, cleaner output
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        # Decode
        cleaned = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return cleaned.strip()
    
    def should_apply_style_transfer(self, text):
        """Determine if text needs style transfer based on learned patterns"""
        if not self.clean_patterns:
            # If no patterns, apply to everything
            return True
        
        # Check if text has SDH content
        sdh_content = self.analyzer.detect_sdh_content(text)
        if sdh_content:
            return True
        
        # Check against learned clean patterns
        stats = self.clean_patterns['stats']
        
        # Check line length (dirty subtitles often have longer lines due to SDH)
        avg_line_length = sum(len(line.strip()) for line in text.split('\n') if line.strip()) / max(1, len([l for l in text.split('\n') if l.strip()]))
        
        if avg_line_length > stats['avg_line_length'] * 1.5:
            return True
        
        # Check for uncommon patterns
        text_lower = text.lower()
        vocabulary = set(self.clean_patterns['vocabulary'])
        text_words = set(text_lower.replace('\n', ' ').split())
        
        # If many unknown words, might be SDH
        unknown_ratio = len(text_words - vocabulary) / max(1, len(text_words))
        if unknown_ratio > 0.3:
            return True
        
        return False
    
    def normalize_srt_file(self, input_file, output_file=None):
        """Apply style transfer to entire SRT file"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_style_normalized.srt"
        
        print(f"Applying style transfer to: {input_file}")
        
        # Parse SRT file
        entries = self.analyzer.parse_srt_file(input_file)
        if not entries:
            print(f"No subtitle entries found in {input_file}")
            return
        
        print(f"Processing {len(entries)} subtitle entries...")
        
        # Apply style transfer
        processed_entries = []
        style_transfers_applied = 0
        
        for i, entry in enumerate(entries):
            if i % 50 == 0:
                print(f"Processing entry {i+1}/{len(entries)}")
            
            original_text = entry.text.strip()
            
            if self.should_apply_style_transfer(original_text):
                # Apply style transfer
                try:
                    cleaned_text = self.apply_clean_style(original_text)
                    entry.text = cleaned_text
                    style_transfers_applied += 1
                except Exception as e:
                    print(f"Error processing entry {i+1}: {e}")
                    # Keep original if error
                    entry.text = original_text
            else:
                # Text already appears clean, keep as is
                entry.text = original_text
            
            processed_entries.append(entry)
        
        # Write output file
        self.write_srt_file(processed_entries, output_file)
        
        print(f"Style transfer complete!")
        print(f"Entries processed: {len(processed_entries)}")
        print(f"Style transfers applied: {style_transfers_applied}")
        print(f"Output saved to: {output_file}")
        
        return output_file
    
    def write_srt_file(self, entries, output_file):
        """Write subtitle entries to SRT file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.index}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.text}\n")
                f.write("\n")
    
    def batch_normalize(self, input_directory, output_directory=None):
        """Apply style transfer to all SRT files in directory"""
        if output_directory is None:
            output_directory = f"{input_directory}_style_normalized"
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Find all SRT files
        srt_files = glob.glob(os.path.join(input_directory, "**/*.srt"), recursive=True)
        print(f"Found {len(srt_files)} SRT files to process")
        
        for i, srt_file in enumerate(srt_files):
            print(f"\n--- Processing {i+1}/{len(srt_files)} ---")
            
            # Create output path maintaining directory structure
            relative_path = os.path.relpath(srt_file, input_directory)
            output_file = os.path.join(output_directory, relative_path)
            
            try:
                self.normalize_srt_file(srt_file, output_file)
            except Exception as e:
                print(f"Error processing {srt_file}: {e}")
                continue
        
        print(f"\nBatch style transfer complete!")
        print(f"Processed files saved to: {output_directory}")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\nStyle Transfer Interactive Test Mode")
        print("Enter subtitle text to clean (press Enter twice to finish):")
        
        while True:
            print("\n" + "="*50)
            print("Enter text to clean:")
            
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            
            if not lines or (len(lines) == 1 and lines[0] == ""):
                break
            
            test_text = "\n".join(lines[:-1])  # Remove last empty line
            
            if test_text.strip():
                print(f"\nOriginal: {test_text}")
                
                # Check if style transfer would be applied
                should_apply = self.should_apply_style_transfer(test_text)
                print(f"Needs cleaning: {should_apply}")
                
                # Apply style transfer
                cleaned = self.apply_clean_style(test_text)
                print(f"Cleaned: {cleaned}")
                
                # Show difference
                if test_text.strip() != cleaned.strip():
                    print(f"Changed: Yes")
                else:
                    print(f"Changed: No")
            
            continue_test = input("\nTest another? (y/n): ").strip().lower()
            if continue_test != 'y':
                break

def main():
    parser = argparse.ArgumentParser(description="Apply the style transfer normalizer to SRT files")
    parser.add_argument("--model-path", dest="model_path", type=str, default="models/style_transfer_normalizer", help="Path to trained style transfer model directory")
    subparsers = parser.add_subparsers(dest="mode")

    # Single-file mode
    single_parser = subparsers.add_parser("file", help="Normalize a single SRT file")
    single_parser.add_argument("--input", dest="input_file", type=str, required=True, help="Path to input SRT file")
    single_parser.add_argument("--output", dest="output_file", type=str, default=None, help="Path to output SRT file")

    # Batch mode
    batch_parser = subparsers.add_parser("dir", help="Normalize all SRT files in a directory")
    batch_parser.add_argument("--input-dir", dest="input_dir", type=str, required=True, help="Directory containing SRT files")
    batch_parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="Output directory for normalized files")

    # Interactive mode
    subparsers.add_parser("interactive", help="Interactive test mode")

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model directory {model_path} does not exist!")
        print("Please train a style transfer model first.")
        return

    try:
        normalizer = StyleTransferNormalizer(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.mode == "file":
        if not os.path.exists(args.input_file):
            print(f"File {args.input_file} does not exist!")
            return
        normalizer.normalize_srt_file(args.input_file, args.output_file)
    elif args.mode == "dir":
        if not os.path.exists(args.input_dir):
            print(f"Directory {args.input_dir} does not exist!")
            return
        normalizer.batch_normalize(args.input_dir, args.output_dir)
    elif args.mode == "interactive" or args.mode is None:
        # Default to interactive if no mode provided
        normalizer.interactive_test()
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()
