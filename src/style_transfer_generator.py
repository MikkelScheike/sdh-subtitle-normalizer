#!/usr/bin/env python3
"""
Style Transfer Training Data Generator
Learns clean subtitle style from good examples and applies to dirty subtitles
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple
from .srt_analyzer import SRTAnalyzer, SubtitleEntry

class StyleTransferGenerator:
    def __init__(self):
        self.analyzer = SRTAnalyzer()
        
    def extract_clean_patterns(self, clean_files_dir: str) -> Dict:
        """Analyze clean files to learn good subtitle patterns"""
        print(f"Learning clean subtitle patterns from: {clean_files_dir}")
        
        import glob
        srt_files = glob.glob(os.path.join(clean_files_dir, "**/*.srt"), recursive=True)
        print(f"Found {len(srt_files)} clean reference files")
        
        clean_patterns = {
            'sentence_structures': [],
            'line_lengths': [],
            'punctuation_patterns': [],
            'vocabulary': set(),
            'timing_patterns': [],
            'clean_examples': []
        }
        
        for filepath in srt_files:
            print(f"Analyzing: {os.path.basename(filepath)}")
            entries = self.analyzer.parse_srt_file(filepath)
            
            for entry in entries:
                # Skip entries that might have SDH content
                if self.analyzer.detect_sdh_content(entry.text):
                    continue
                    
                text = entry.text.strip()
                if not text:
                    continue
                
                # Store clean examples
                clean_patterns['clean_examples'].append(text)
                
                # Analyze sentence structure
                sentences = text.replace('\n', ' ').split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        clean_patterns['sentence_structures'].append(len(sentence.split()))
                
                # Line length patterns
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        clean_patterns['line_lengths'].append(len(line.strip()))
                
                # Vocabulary building
                words = text.lower().replace('\n', ' ').split()
                clean_patterns['vocabulary'].update(words)
                
                # Timing patterns (duration)
                start_parts = entry.start_time.split(':')
                end_parts = entry.end_time.split(':')
                
                start_ms = (int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + 
                           float(start_parts[2].replace(',', '.'))) * 1000
                end_ms = (int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + 
                         float(end_parts[2].replace(',', '.'))) * 1000
                
                duration = end_ms - start_ms
                clean_patterns['timing_patterns'].append(duration / len(text))  # ms per character
        
        # Convert set to list for JSON serialization
        clean_patterns['vocabulary'] = list(clean_patterns['vocabulary'])
        
        # Calculate statistics
        if clean_patterns['line_lengths']:
            clean_patterns['stats'] = {
                'avg_line_length': sum(clean_patterns['line_lengths']) / len(clean_patterns['line_lengths']),
                'avg_sentence_length': sum(clean_patterns['sentence_structures']) / len(clean_patterns['sentence_structures']) if clean_patterns['sentence_structures'] else 0,
                'total_clean_examples': len(clean_patterns['clean_examples']),
                'vocabulary_size': len(clean_patterns['vocabulary'])
            }
        
        print(f"Learned patterns from {len(clean_patterns['clean_examples'])} clean examples")
        print(f"Average line length: {clean_patterns['stats']['avg_line_length']:.1f} characters")
        print(f"Vocabulary size: {clean_patterns['stats']['vocabulary_size']} words")
        
        return clean_patterns
    
    def generate_style_transfer_data(self, clean_dir: str, output_file: str, max_examples: int = 5000):
        """Generate training data for style transfer model"""
        
        # Step 1: Learn clean patterns
        clean_patterns = self.extract_clean_patterns(clean_dir)
        
        # Step 2: Create training examples
        training_data = []
        
        # Use clean examples as positive examples (clean → clean)
        clean_examples = clean_patterns['clean_examples'][:max_examples//2]
        
        for example in clean_examples:
            training_data.append({
                'input': example,
                'output': example,
                'type': 'clean_reference',
                'source': 'reference'
            })
        
        # Step 3: Create contrastive examples (what NOT to do)
        # We'll create these by taking clean examples and artificially making them "dirty"
        # Then the model learns: dirty_version → clean_version
        
        synthetic_dirty_examples = self.create_contrastive_examples(
            clean_examples[:max_examples//2], 
            max_examples - len(training_data)
        )
        
        training_data.extend(synthetic_dirty_examples)
        
        # Step 4: Shuffle and split
        random.shuffle(training_data)
        split_idx = int(len(training_data) * 0.9)
        
        output_data = {
            'train': training_data[:split_idx],
            'validation': training_data[split_idx:],
            'clean_patterns': clean_patterns,
            'metadata': {
                'total_examples': len(training_data),
                'train_examples': split_idx,
                'val_examples': len(training_data) - split_idx,
                'clean_references': len([d for d in training_data if d['type'] == 'clean_reference']),
                'contrastive_examples': len([d for d in training_data if d['type'] == 'contrastive'])
            }
        }
        
        # Save training data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nStyle transfer training data saved to: {output_file}")
        print(f"Total examples: {len(training_data)}")
        print(f"Clean references: {output_data['metadata']['clean_references']}")
        print(f"Contrastive examples: {output_data['metadata']['contrastive_examples']}")
        
        # Show examples
        print(f"\nSample Training Examples:")
        for i, example in enumerate(training_data[:3]):
            print(f"\n--- Example {i+1} ({example['type']}) ---")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")
        
        return output_data
    
    def create_contrastive_examples(self, clean_examples: List[str], max_examples: int) -> List[Dict]:
        """Create dirty versions of clean examples for contrastive learning"""
        contrastive_data = []
        
        # SDH patterns to add to clean examples
        sdh_patterns = {
            'sound_effects': [
                '[MUSIC PLAYING]', '[DOOR SLAMS]', '[PHONE RINGING]', '[FOOTSTEPS]',
                '(sighs)', '(laughs)', '(whispers)', '(crying)', '♪ music ♪'
            ],
            'speaker_labels': [
                'JOHN:', 'MARY:', 'NARRATOR:', '>> SARAH:', '- TOM:', 'DOCTOR:'
            ],
            'action_tags': [
                '[sighs heavily]', '[door creaks]', '[wind howling]', '[phone buzzing]'
            ]
        }
        
        for clean_text in clean_examples:
            if len(contrastive_data) >= max_examples:
                break
            
            # Create multiple dirty versions of each clean example
            for variation in range(3):
                if len(contrastive_data) >= max_examples:
                    break
                
                dirty_version = clean_text
                
                # Randomly add SDH elements
                if random.random() < 0.4:
                    sound = random.choice(sdh_patterns['sound_effects'])
                    if random.random() < 0.5:
                        dirty_version = f"{sound}\n{dirty_version}"
                    else:
                        dirty_version = f"{dirty_version}\n{sound}"
                
                if random.random() < 0.3:
                    speaker = random.choice(sdh_patterns['speaker_labels'])
                    dirty_version = f"{speaker} {dirty_version}"
                
                if random.random() < 0.2:
                    action = random.choice(sdh_patterns['action_tags'])
                    words = dirty_version.split()
                    if len(words) > 2:
                        insert_pos = random.randint(1, len(words) - 1)
                        words.insert(insert_pos, action)
                        dirty_version = ' '.join(words)
                
                # Add bad line breaks
                if random.random() < 0.4 and '\n' not in dirty_version:
                    words = dirty_version.split()
                    if len(words) > 4:
                        break_pos = random.randint(2, len(words) - 2)
                        dirty_version = ' '.join(words[:break_pos]) + '\n' + ' '.join(words[break_pos:])
                
                # Only add if it's actually different
                if dirty_version.strip() != clean_text.strip():
                    contrastive_data.append({
                        'input': dirty_version,
                        'output': clean_text,
                        'type': 'contrastive',
                        'source': 'synthetic'
                    })
        
        return contrastive_data

def main():
    parser = argparse.ArgumentParser(description="Generate style transfer training data")
    parser.add_argument("--clean-dir", dest="clean_dir", type=str, default=None, help="Directory with CLEAN reference SRT files")
    parser.add_argument("--output", dest="output_file", type=str, default=None, help="Output JSON file for training data")
    parser.add_argument("--max-examples", dest="max_examples", type=int, default=5000, help="Maximum number of training examples to generate")
    args = parser.parse_args()

    generator = StyleTransferGenerator()

    # Resolve inputs (fallback to interactive if missing)
    clean_dir = args.clean_dir or input("Enter directory containing CLEAN reference SRT files: ").strip() or "data/raw_srt"
    if not os.path.exists(clean_dir):
        print(f"Directory {clean_dir} does not exist!")
        return

    output_file = args.output_file or input("Enter output file name (default: style_transfer_data.json): ").strip() or "data/style_transfer_data.json"

    max_examples = args.max_examples

    # Generate training data
    generator.generate_style_transfer_data(clean_dir, output_file, max_examples)

if __name__ == "__main__":
    main()
