#!/usr/bin/env python3
"""
Training Data Generator for SDH Subtitle Normalization
Creates input-output pairs for training a text-to-text model
"""

import os
import json
import random
import re
from typing import List, Tuple, Dict
from srt_analyzer import SRTAnalyzer, SubtitleEntry

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
    
    def filter_by_language(self, file_list, target_language):
        """Filter files by ISO 639 language code in file names"""
        import re
        
        filtered_files = []
        language_patterns = [
            rf'[._-]{target_language}[._-]',  # movie_en.srt, movie.en.srt
            rf'[._-]{target_language}$',      # movie_en.srt (end of basename)
            rf'^{target_language}[._-]',     # en_movie.srt
        ]
        
        for file_path in file_list:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Check all patterns
            for pattern in language_patterns:
                if re.search(pattern, basename, re.IGNORECASE):
                    filtered_files.append(file_path)
                    break
            else:
                # Fallback: check if ends with target language
                if basename.lower().endswith(f'_{target_language.lower()}') or basename.lower().endswith(f'.{target_language.lower()}'):
                    filtered_files.append(file_path)
        
        return filtered_files
    
    def analyze_file_languages(self, file_list):
        """Analyze and show detected languages in file names"""
        import re
        
        detected_languages = {}
        iso_639_1_codes = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
            'ar': 'Arabic', 'hi': 'Hindi', 'nl': 'Dutch', 'sv': 'Swedish', 'no': 'Norwegian',
            'da': 'Danish', 'fi': 'Finnish', 'pl': 'Polish', 'tr': 'Turkish', 'he': 'Hebrew'
        }
        
        for file_path in file_list:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Look for 2-letter language codes
            for code, name in iso_639_1_codes.items():
                if re.search(rf'[._-]{code}[._-]|[._-]{code}$|^{code}[._-]', basename, re.IGNORECASE):
                    if code not in detected_languages:
                        detected_languages[code] = {'name': name, 'count': 0, 'files': []}
                    detected_languages[code]['count'] += 1
                    detected_languages[code]['files'].append(os.path.basename(file_path))
                    break
        
        return detected_languages
    
    def load_commercial_filters(self, filter_file="commercial_filters.json"):
        """Load commercial text patterns to be removed"""
        if os.path.exists(filter_file):
            with open(filter_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def is_commercial_text(self, text, commercial_filters):
        """Check if text contains commercial content that should be removed"""
        if not commercial_filters:
            return False
        
        text_lower = text.lower().strip()
        
        # Check against commercial patterns
        for pattern in commercial_filters.get('commercial_patterns', []):
            pattern_check = pattern.lower() if not commercial_filters['removal_rules']['case_sensitive'] else pattern
            
            if pattern_check in text_lower:
                return True
        
        # Check against custom removals
        for removal in commercial_filters.get('custom_removals', []):
            removal_text = removal['text']
            if not commercial_filters['removal_rules']['case_sensitive']:
                removal_text = removal_text.lower()
            
            if removal_text in text_lower:
                return True
        
        return False
    
    def create_commercial_removal_pairs(self, text, commercial_filters):
        """Create training pairs for commercial text removal"""
        pairs = []
        
        if not commercial_filters:
            return pairs
        
        # If this text contains commercial content, create removal training pairs
        if self.is_commercial_text(text, commercial_filters):
            # Create multiple training examples with different weights
            weight = commercial_filters['removal_rules'].get('training_weight', 3)
            
            for _ in range(weight):
                pairs.append({
                    'input': text,
                    'output': '',  # Remove completely
                    'source': 'commercial_removal',
                    'file': 'commercial_filter'
                })
        
        return pairs
    
    def create_comprehensive_training_pairs(self, entry: SubtitleEntry) -> List[Dict]:
        """Create comprehensive training pairs including manual clean versions"""
        pairs = []
        original = entry.text.strip()
        
        # Create manually cleaned version (what we want the model to learn)
        cleaned = self.create_manual_clean_version(original)
        
        if original != cleaned:
            pairs.append({
                'input': original,
                'output': cleaned,
                'source': 'comprehensive_sdh',
                'file': 'manual_clean'
            })
        
        return pairs
    
    def create_manual_clean_version(self, text: str) -> str:
        """Create manually cleaned version - what we want the model to learn"""
        import re
        
        cleaned = text
        
        # Remove SDH patterns - let training data teach these transformations
        # [MUSIC PLAYING] -> removed
        cleaned = re.sub(r'\[([^\]]*)\]', '', cleaned, flags=re.IGNORECASE)
        
        # (sound effects) -> removed
        cleaned = re.sub(r'\(([^)]*)\)', '', cleaned, flags=re.IGNORECASE)
        
        # ♪ music ♪ -> removed  
        cleaned = re.sub(r'♪([^♪]*)♪', '', cleaned)
        
        # Speaker labels: "JOHN: Hello" -> "Hello"
        cleaned = re.sub(r'^([-–>]*\s*[A-Z][A-Z\s]*:|[A-Z][A-Z\s]*:)\s*', '', cleaned, flags=re.MULTILINE)
        
        # Narrator tags: ">> NARRATOR:" -> removed
        cleaned = re.sub(r'>>?\s*[A-Z][A-Z\s]*:?\s*', '', cleaned)
        
        # Fix line breaks - merge broken sentences
        lines = cleaned.split('\n')
        merged_lines = []
        current_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if current_line and not re.search(r'[.!?]$', current_line.strip()):
                # Previous line doesn't end with punctuation, merge
                current_line += " " + line
            else:
                if current_line:
                    merged_lines.append(current_line)
                current_line = line
        
        if current_line:
            merged_lines.append(current_line)
        
        cleaned = '\n'.join(merged_lines)
        
        # Clean up formatting
        cleaned = re.sub(r'>>+\s*', '', cleaned)  # Remove >> markers
        cleaned = re.sub(r'^[-–]\s*', '', cleaned, flags=re.MULTILINE)  # Remove leading dashes
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Remove extra blank lines
        
        return cleaned.strip()
    
    def create_training_pair(self, entry: SubtitleEntry) -> Tuple[str, str]:
        """Create basic input-output training pair - legacy method"""
        original = entry.text
        cleaned = self.create_manual_clean_version(original)
        
        return original, cleaned
    
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
    
    def is_all_caps(self, text: str) -> bool:
        """Check if text is predominantly in ALL CAPS"""
        import re
        letters_only = re.sub(r'[^a-zA-Z]', '', text)
        if len(letters_only) < 5:
            return False
        uppercase_count = sum(1 for c in letters_only if c.isupper())
        return uppercase_count / len(letters_only) > 0.8
    
    def normalize_case_for_training(self, text: str) -> str:
        """Create proper case version for training data"""
        if not self.is_all_caps(text):
            return text
        
        # Simple sentence case normalization for training
        normalized = text.lower()
        # Capitalize sentence beginnings
        normalized = '. '.join(s.strip().capitalize() for s in normalized.split('.') if s.strip())
        normalized = '! '.join(s.strip().capitalize() for s in normalized.split('!') if s.strip())
        normalized = '? '.join(s.strip().capitalize() for s in normalized.split('?') if s.strip())
        # Fix "I" 
        normalized = re.sub(r'\bi\b', 'I', normalized)
        return normalized
    
    def create_case_normalization_pairs(self, text: str) -> List[Dict]:
        """Create training pairs for case normalization"""
        pairs = []
        
        if self.is_all_caps(text):
            normalized_case = self.normalize_case_for_training(text)
            if normalized_case != text:
                pairs.append({
                    'input': text,
                    'output': normalized_case,
                    'source': 'case_normalization',
                    'file': 'case_normalizer'
                })
        
        return pairs
    
    def generate_training_data(self, srt_directory: str, output_file: str, max_pairs: int = 10000):
        """Generate training dataset from SRT files"""
        print(f"Generating training data from: {srt_directory}")
        
        # Load commercial filters
        commercial_filters = self.load_commercial_filters()
        if commercial_filters:
            print(f"Loaded commercial filters with {len(commercial_filters['commercial_patterns'])} patterns")
        else:
            print("No commercial filters found (commercial_filters.json)")
        
        # Get all SRT files
        import glob
        srt_files = glob.glob(os.path.join(srt_directory, "**/*.srt"), recursive=True)
        print(f"Found {len(srt_files)} total SRT files")
        
        # Analyze languages in file names
        detected_languages = self.analyze_file_languages(srt_files)
        if detected_languages:
            print("\nDetected languages in file names:")
            for code, info in detected_languages.items():
                print(f"  {code}: {info['name']} ({info['count']} files)")
                if info['count'] <= 5:  # Show file names if few files
                    for filename in info['files'][:3]:
                        print(f"    - {filename}")
                    if len(info['files']) > 3:
                        print(f"    ... and {len(info['files'])-3} more")
        
        # Filter by language code if specified
        target_language = input("\nEnter target ISO 639 language code (e.g., 'en' for English, or press Enter to process all): ").strip()
        if target_language:
            original_count = len(srt_files)
            srt_files = self.filter_by_language(srt_files, target_language)
            print(f"Filtered to {len(srt_files)} files with language code '{target_language}' (from {original_count} total)")
        else:
            print(f"Processing all {len(srt_files)} SRT files")
        
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
                
                # Check for commercial content first
                if commercial_filters and self.is_commercial_text(entry.text, commercial_filters):
                    # Create commercial removal training pairs
                    commercial_pairs = self.create_commercial_removal_pairs(entry.text, commercial_filters)
                    training_pairs.extend(commercial_pairs)
                    continue
                
                # Check for ALL CAPS text and create case normalization pairs
                case_pairs = self.create_case_normalization_pairs(entry.text)
                training_pairs.extend(case_pairs)
                
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
                'synthetic_pairs': len([p for p in training_pairs if p['source'] == 'synthetic']),
                'commercial_removal_pairs': len([p for p in training_pairs if p['source'] == 'commercial_removal']),
                'case_normalization_pairs': len([p for p in training_pairs if p['source'] == 'case_normalization'])
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
