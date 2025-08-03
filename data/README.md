# Data Directory

This directory is used to store:

## Input Data
- `raw_srt/` - Place your original SRT files here
- `sample_data/` - Sample SRT files for testing

## Generated Data
- `training_data.json` - Generated training pairs
- `analysis_results.json` - Analysis results from SRT files

## Directory Structure
```
data/
├── raw_srt/           # Your original SRT files
│   ├── movie1.srt
│   ├── tv_show/
│   │   ├── episode1.srt
│   │   └── episode2.srt
│   └── ...
├── sample_data/       # Sample files for testing
├── training_data.json # Generated training pairs
└── analysis_results.json # Analysis output
```

## Usage

1. **Place your SRT files** in the `raw_srt/` directory
2. **Run analysis**: `python src/srt_analyzer.py` and point to `data/raw_srt/`
3. **Generate training data**: `python src/training_data_generator.py`
4. **Train model**: Training data will be automatically loaded from here
