#!/usr/bin/env python3
"""
Training Data Generator for SDH Subtitle Normalization
Creates input-output pairs for training a text-to-text model
"""

import os
import json
import random
from typing import List, Tuple, Dict
from .srt_analyzer import SRTAnalyzer, SubtitleEntry

class TrainingDataGenerator:
    def __init__(self):
        self.analyzer = SRTAnalyzer()
        
        # Synthetic SDH patterns to augment training data
        self.synthetic_sdh = {
            'sound_effects': [
                '[MUSIC PLAYING]', '[DOOR SLAMS]', '[PHONE RINGING]', '[CAR ENGINE]',
                '[THUNDER RUMBLING]', '[FOOTSTEPS]', '[APPLAUSE]', '[GUNSHOT]',
                '(sighs)', '(laughs)', '(whispers)', '(shouting)', '(crying)',
                '♪ upbeat music ♪', '♪ sad melody ♪', '♪ classical music ♪'
            ],
            'speaker_labels': [
                'JOHN:', 'MARY:', 'NARRATOR:', '>> SARAH:', '- TOM:', 'ALICE:',
                'DOCTOR:', 'TEACHER:', 'STUDENT:', 'MOTHER:'
            ],
            'action_tags': [
                '[sighs heavily]', '[door creaks]', '[glass breaks]', '[wind howling]',
                '[typing on keyboard]', '[phone buzzing]', '[car honks]', '[dog barks]'
            ]
        }
    
    def create_training_pair(self, entry: SubtitleEntry) -> Tuple[str, str]:
        """Create input-output training pair from subtitle entry"""
        original = entry.text
        normalized = self.analyzer.normalize_subtitle(original)
        
        return original, normalized
    
    def augment_clean_subtitle(self, clean_text: str) -> List[Tuple[str, str]]:
        """Take clean subtitle and add synthetic SDH to create training pairs"""
        training_pairs = []
        
        # Original clean version
        training_pairs.append((clean_text, clean_text))
        
        # Add various SDH elements
        for _ in range(3):  # Generate 3 variations per clean subtitle
            augmented = clean_text
            
            # Randomly add sound effects
            if random.random() < 0.3:
                sound_effect = random.choice(self.synthetic_sdh['sound_effects'])
                if random.random() < 0.5:
                    augmented = f"{sound_effect}\n{augmented}"
                else:
                    augmented = f"{augmented}\n{sound_effect}"
            
            # Randomly add speaker labels
            if random.random() < 0.4:
                speaker = random.choice(self.synthetic_sdh['speaker_labels'])
                augmented = f"{speaker} {augmented}"
            
            # Randomly add action tags
            if random.random() < 0.2:
                action = random.choice(self.synthetic_sdh['action_tags'])
                words = augmented.split()
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, action)
                augmented = ' '.join(words)
            
            # Randomly mess up line breaks
            if random.random() < 0.5 and '\n' not in augmented:
                words = augmented.split()
                if len(words) > 4:
                    break_pos = random.randint(2, len(words) - 2)
                    augmented = ' '.join(words[:break_pos]) + '\n' + ' '.join(words[break_pos:])
            
            training_pairs.append((augmented, clean_text))
        
        return training_pairs
    
    def generate_training_data(self, srt_directory: str, output_file: str, max_pairs: int = 10000):
        """Generate training dataset from SRT files"""
        print(f"Generating training data from: {srt_directory}")
        
        # Get all SRT files
        import glob
        srt_files = glob.glob(os.path.join(srt_directory, "**/*.srt"), recursive=True)
        print(f"Found {len(srt_files)} SRT files")
        
        training_pairs = []
        clean_subtitles = []
        
        # Process SRT files
        for i, filepath in enumerate(srt_files):
            if len(training_pairs) >= max_pairs:
                break
                
            print(f"Processing {i+1}/{len(srt_files)}: {os.path.basename(filepath)}")
            entries = self.analyzer.parse_srt_file(filepath)
            
            for entry in entries:
                if len(training_pairs) >= max_pairs:
                    break
                
                # Create training pair from existing SDH content
                sdh_content = self.analyzer.detect_sdh_content(entry.text)
                if sdh_content:
                    original, normalized = self.create_training_pair(entry)
                    if original.strip() != normalized.strip():  # Only if there's actual change
                        training_pairs.append({
                            'input': original,
                            'output': normalized,
                            'source': 'real_sdh',
                            'file': os.path.basename(filepath)
                        })
                else:
                    # Store clean subtitles for augmentation
                    clean_subtitles.append(entry.text.strip())
        
        print(f"Generated {len(training_pairs)} pairs from real SDH content")
        
        # Augment with synthetic SDH
        synthetic_pairs_needed = min(max_pairs - len(training_pairs), len(clean_subtitles) * 3)
        print(f"Generating {synthetic_pairs_needed} synthetic pairs...")
        
        for clean_text in clean_subtitles[:synthetic_pairs_needed//3]:
            if len(training_pairs) >= max_pairs:
                break
                
            augmented_pairs = self.augment_clean_subtitle(clean_text)
            for aug_input, aug_output in augmented_pairs:
                if len(training_pairs) >= max_pairs:
                    break
                if aug_input.strip() != aug_output.strip():
                    training_pairs.append({
                        'input': aug_input,
                        'output': aug_output,
                        'source': 'synthetic',
                        'file': 'augmented'
                    })
        
        # Shuffle training pairs
        random.shuffle(training_pairs)
        
        # Split into train/validation
        split_idx = int(len(training_pairs) * 0.9)
        train_data = training_pairs[:split_idx]
        val_data = training_pairs[split_idx:]
        
        # Save training data
        output_data = {
            'train': train_data,
            'validation': val_data,
            'metadata': {
                'total_pairs': len(training_pairs),
                'train_pairs': len(train_data),
                'val_pairs': len(val_data),
                'real_sdh_pairs': len([p for p in training_pairs if p['source'] == 'real_sdh']),
                'synthetic_pairs': len([p for p in training_pairs if p['source'] == 'synthetic'])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nTraining data saved to: {output_file}")
        print(f"Total pairs: {len(training_pairs)}")
        print(f"Train pairs: {len(train_data)}")
        print(f"Validation pairs: {len(val_data)}")
        print(f"Real SDH pairs: {output_data['metadata']['real_sdh_pairs']}")
        print(f"Synthetic pairs: {output_data['metadata']['synthetic_pairs']}")
        
        # Show some examples
        print(f"\nSample Training Pairs:")
        for i, pair in enumerate(train_data[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Input: {pair['input']}")
            print(f"Output: {pair['output']}")
            print(f"Source: {pair['source']}")
        
        return output_data

def main():
    generator = TrainingDataGenerator()
    
    # Get input directory
    srt_dir = input("Enter directory containing SRT files: ").strip()
    if not srt_dir:
        srt_dir = "."
    
    if not os.path.exists(srt_dir):
        print(f"Directory {srt_dir} does not exist!")
        return
    
    # Get output file
    output_file = input("Enter output file name (default: training_data.json): ").strip()
    if not output_file:
        output_file = "training_data.json"
    
    # Get max pairs
    max_pairs_input = input("Enter maximum training pairs to generate (default: 10000): ").strip()
    max_pairs = int(max_pairs_input) if max_pairs_input else 10000
    
    # Generate training data
    generator.generate_training_data(srt_dir, output_file, max_pairs)

if __name__ == "__main__":
    main()
