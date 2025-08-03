#!/usr/bin/env python3
"""
SRT Subtitle Analyzer and Normalizer
Analyzes SRT files to identify SDH patterns and prepare training data
"""

import re
import os
import glob
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry"""
    index: int
    start_time: str
    end_time: str
    text: str
    original_text: str = ""

class SRTAnalyzer:
    def __init__(self):
        # SDH patterns to identify and remove
        self.sdh_patterns = {
            'sound_effects_brackets': r'\[([^]]*)\]',
            'sound_effects_parens': r'\(([^)]*)\)',
            'music_notes': r'♪([^♪]*)♪',
            'speaker_labels': r'^([-–>]*\s*[A-Z][A-Z\s]*:|[A-Z][A-Z\s]*:)',
            'action_descriptions': r'\[[^\]]*\]',
            'environmental_sounds': r'\([^)]*\)',
            'narrator_tags': r'>>?\s*[A-Z][A-Z\s]*:?',
        }
        
        # Common SDH keywords to detect
        self.sdh_keywords = [
            'music playing', 'door slams', 'phone ringing', 'laughs', 'sighs',
            'thunder', 'engine', 'footsteps', 'applause', 'crying', 'breathing',
            'gunshot', 'explosion', 'narrator', 'whispers', 'shouting'
        ]
    
    def parse_srt_file(self, filepath: str) -> List[SubtitleEntry]:
        """Parse an SRT file and return list of subtitle entries"""
        entries = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except:
                    continue
            else:
                print(f"Could not read file: {filepath}")
                return []
        
        # Split by double newlines to get individual subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0].strip())
                    time_line = lines[1].strip()
                    text_lines = lines[2:]
                    
                    # Parse timing
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
                    if time_match:
                        start_time, end_time = time_match.groups()
                        text = '\n'.join(text_lines)
                        
                        entry = SubtitleEntry(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            text=text,
                            original_text=text
                        )
                        entries.append(entry)
                except (ValueError, IndexError):
                    continue
        
        return entries
    
    def detect_sdh_content(self, text: str) -> Dict[str, List[str]]:
        """Detect SDH content in subtitle text"""
        sdh_found = {}
        
        for pattern_name, pattern in self.sdh_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                sdh_found[pattern_name] = matches
        
        # Check for SDH keywords
        text_lower = text.lower()
        keyword_matches = [kw for kw in self.sdh_keywords if kw in text_lower]
        if keyword_matches:
            sdh_found['keywords'] = keyword_matches
        
        return sdh_found
    
    def normalize_subtitle(self, text: str) -> str:
        """Apply minimal normalization rules - most logic moved to training data"""
        # Only basic cleanup - let the T5 model handle complex transformations
        normalized = text.strip()
        
        # Basic whitespace cleanup only
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize multiple spaces
        normalized = re.sub(r'\n\s*\n+', '\n', normalized)  # Remove extra blank lines
        
        return normalized
    
    def analyze_directory(self, directory: str) -> Dict:
        """Analyze all SRT files in a directory"""
        srt_files = glob.glob(os.path.join(directory, "**/*.srt"), recursive=True)
        
        analysis = {
            'total_files': len(srt_files),
            'total_entries': 0,
            'sdh_entries': 0,
            'sdh_patterns_found': {},
            'sample_sdh_content': [],
            'files_analyzed': []
        }
        
        print(f"Found {len(srt_files)} SRT files in {directory}")
        
        for filepath in srt_files[:10]:  # Analyze first 10 files as sample
            print(f"Analyzing: {os.path.basename(filepath)}")
            entries = self.parse_srt_file(filepath)
            
            file_analysis = {
                'filepath': filepath,
                'entries': len(entries),
                'sdh_entries': 0,
                'sdh_samples': []
            }
            
            for entry in entries:
                analysis['total_entries'] += 1
                
                sdh_content = self.detect_sdh_content(entry.text)
                if sdh_content:
                    analysis['sdh_entries'] += 1
                    file_analysis['sdh_entries'] += 1
                    
                    # Track SDH patterns
                    for pattern_type, matches in sdh_content.items():
                        if pattern_type not in analysis['sdh_patterns_found']:
                            analysis['sdh_patterns_found'][pattern_type] = []
                        analysis['sdh_patterns_found'][pattern_type].extend(matches)
                    
                    # Store sample for review
                    if len(analysis['sample_sdh_content']) < 20:
                        normalized = self.normalize_subtitle(entry.text)
                        analysis['sample_sdh_content'].append({
                            'original': entry.text,
                            'normalized': normalized,
                            'sdh_detected': sdh_content
                        })
            
            analysis['files_analyzed'].append(file_analysis)
        
        return analysis

def main():
    analyzer = SRTAnalyzer()
    
    # Get directory from user
    directory = input("Enter directory path containing SRT files (or press Enter for current directory): ").strip()
    if not directory:
        directory = "."
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return
    
    print(f"\nAnalyzing SRT files in: {os.path.abspath(directory)}")
    analysis = analyzer.analyze_directory(directory)
    
    # Print summary
    print(f"\n{'='*50}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Total files found: {analysis['total_files']}")
    print(f"Total subtitle entries: {analysis['total_entries']}")
    print(f"Entries with SDH content: {analysis['sdh_entries']}")
    if analysis['total_entries'] > 0:
        print(f"SDH percentage: {analysis['sdh_entries']/analysis['total_entries']*100:.1f}%")
    
    print(f"\nSDH Patterns Found:")
    for pattern_type, matches in analysis['sdh_patterns_found'].items():
        unique_matches = list(set(matches))[:10]  # Show up to 10 unique examples
        print(f"  {pattern_type}: {len(matches)} total, examples: {unique_matches}")
    
    print(f"\nSample Normalizations:")
    for i, sample in enumerate(analysis['sample_sdh_content'][:5]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original: {sample['original']}")
        print(f"Normalized: {sample['normalized']}")
        print(f"SDH detected: {list(sample['sdh_detected'].keys())}")
    
    # Save detailed analysis
    output_file = "srt_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
