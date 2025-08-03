#!/usr/bin/env python3
"""
Subtitle Normalizer - Use trained model to normalize SDH subtitles
"""

import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .srt_analyzer import SRTAnalyzer
import re

class SubtitleNormalizer:
    def __init__(self, model_path):
        """Initialize the normalizer with a trained model"""
        print(f"Loading model from {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        self.analyzer = SRTAnalyzer()
        print("Model loaded successfully!")
    
    def normalize_text(self, text):
        """Normalize a single subtitle text using the trained model"""
        # Prepare input
        input_text = f"normalize subtitle: {text}"
        
        # Tokenize
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode
        normalized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalized.strip()
    
    def normalize_srt_file(self, input_file, output_file=None):
        """Normalize an entire SRT file"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_normalized.srt"
        
        print(f"Processing: {input_file}")
        
        # Parse the SRT file
        entries = self.analyzer.parse_srt_file(input_file)
        
        if not entries:
            print(f"No subtitle entries found in {input_file}")
            return
        
        print(f"Found {len(entries)} subtitle entries")
        
        # Normalize each entry
        normalized_entries = []
        sdh_found = 0
        
        for i, entry in enumerate(entries):
            if i % 50 == 0:
                print(f"Processing entry {i+1}/{len(entries)}")
            
            # Check if this entry has SDH content
            sdh_content = self.analyzer.detect_sdh_content(entry.text)
            
            if sdh_content:
                # Use AI model for normalization
                normalized_text = self.normalize_text(entry.text)
                sdh_found += 1
            else:
                # For clean subtitles, just do basic formatting cleanup
                normalized_text = self.analyzer.normalize_subtitle(entry.text)
            
            # Update entry with normalized text
            entry.text = normalized_text
            normalized_entries.append(entry)
        
        # Write normalized SRT file
        self.write_srt_file(normalized_entries, output_file)
        
        print(f"Normalization complete!")
        print(f"SDH entries found and normalized: {sdh_found}")
        print(f"Output saved to: {output_file}")
        
        return output_file
    
    def write_srt_file(self, entries, output_file):
        """Write subtitle entries to SRT file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(f"{entry.index}\n")
                f.write(f"{entry.start_time} --> {entry.end_time}\n")
                f.write(f"{entry.text}\n")
                f.write("\n")
    
    def batch_normalize(self, input_directory, output_directory=None):
        """Normalize all SRT files in a directory"""
        if output_directory is None:
            output_directory = f"{input_directory}_normalized"
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Find all SRT files
        import glob
        srt_files = glob.glob(os.path.join(input_directory, "**/*.srt"), recursive=True)
        
        print(f"Found {len(srt_files)} SRT files to normalize")
        
        for i, srt_file in enumerate(srt_files):
            print(f"\n--- Processing {i+1}/{len(srt_files)} ---")
            
            # Create relative output path
            relative_path = os.path.relpath(srt_file, input_directory)
            output_file = os.path.join(output_directory, relative_path)
            
            # Create output subdirectory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            try:
                self.normalize_srt_file(srt_file, output_file)
            except Exception as e:
                print(f"Error processing {srt_file}: {e}")
                continue
        
        print(f"\nBatch normalization complete!")
        print(f"Normalized files saved to: {output_directory}")

def main():
    # Get model path
    model_path = input("Enter path to trained model directory: ").strip()
    if not model_path:
        model_path = "./subtitle_normalizer"
    
    if not os.path.exists(model_path):
        print(f"Model directory {model_path} does not exist!")
        print("Please train a model first using train_model.py")
        return
    
    # Initialize normalizer
    try:
        normalizer = SubtitleNormalizer(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get operation mode
    print("\\nChoose operation mode:")
    print("1. Normalize single SRT file")
    print("2. Normalize all SRT files in a directory")
    print("3. Test on sample text")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Single file mode
        input_file = input("Enter path to SRT file: ").strip()
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist!")
            return
        
        output_file = input("Enter output file path (optional): ").strip()
        if not output_file:
            output_file = None
        
        normalizer.normalize_srt_file(input_file, output_file)
    
    elif choice == "2":
        # Directory mode
        input_dir = input("Enter directory containing SRT files: ").strip()
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist!")
            return
        
        output_dir = input("Enter output directory (optional): ").strip()
        if not output_dir:
            output_dir = None
        
        normalizer.batch_normalize(input_dir, output_dir)
    
    elif choice == "3":
        # Test mode
        print("\\nEnter subtitle text to normalize (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        
        test_text = "\\n".join(lines[:-1])  # Remove the last empty line
        
        if test_text.strip():
            print(f"\\nOriginal: {test_text}")
            normalized = normalizer.normalize_text(test_text)
            print(f"Normalized: {normalized}")
        else:
            print("No text entered!")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
