# Data Directory

This directory contains all your subtitle files organized by purpose.

## Directory Structure
```
data/
├── clean_training_data/      # Your 20,000+ clean SRT files (for training)
├── files_to_be_cleaned/      # New/dirty SRT files you want to normalize  
│   └── 28 Years Later (2025).en.srt
├── cleaned_output/           # Results after running the model
├── sample_data/              # Sample files for testing
├── training_data.json        # Generated training pairs
└── analysis_results.json     # Analysis output
```

## Folder Purposes

### 📚 `clean_training_data/`
- **Purpose**: Your high-quality, manually cleaned SRT files
- **Contents**: 20,000+ clean subtitle files
- **Use**: Training data generation (synthetic SDH added to these)

### 🔧 `files_to_be_cleaned/`  
- **Purpose**: New/dirty SRT files that need normalization
- **Contents**: Files with SDH, commercials, formatting issues
- **Use**: Input for your trained model

### ✨ `cleaned_output/`
- **Purpose**: Results from the trained model
- **Contents**: Normalized, cleaned subtitle files
- **Use**: Your final, processed subtitles

## Workflow

1. **Training**: Put clean files in `clean_training_data/`
2. **Processing**: Put dirty files in `files_to_be_cleaned/`
3. **Results**: Get cleaned files from `cleaned_output/`

## Next Steps

1. Move your 20,000+ clean SRT files to `clean_training_data/`
2. Keep dirty files (like 28 Years Later) in `files_to_be_cleaned/`  
3. Run training data generation pointing to `clean_training_data/`
