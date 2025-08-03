# SDH Subtitle Normalization with AI

This project trains an AI model to normalize SDH (Signs and Descriptions for the Hearing Impaired) subtitles by removing sound effects, speaker labels, and fixing formatting issues.

## What it does

- **Removes SDH elements**: `[MUSIC PLAYING]`, `(door slams)`, `♪ music ♪`
- **Removes speaker labels**: `JOHN:`, `>> NARRATOR:`, `- MARY:`
- **Fixes line breaks**: Merges artificially split sentences
- **Cleans formatting**: Removes special characters and normalizes spacing

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your SRT Files
```bash
# Place your SRT files in the data directory
cp /path/to/your/srt/files/* data/raw_srt/
```

### 3. Analyze Your SRT Files
```bash
python src/srt_analyzer.py
# Or use the Makefile
make analyze
```
This analyzes your SRT files to identify SDH patterns and shows examples.

### 4. Generate Training Data
```bash
python src/training_data_generator.py
# Or use the Makefile
make generate-data
```
Creates training pairs from your SRT files (original SDH → normalized text).

### 5. Train the Model
```bash
python src/train_model.py
# Or use the Makefile
make train
```
Trains a T5 model on your data. Uses `t5-small` by default (faster) or `t5-base` for better quality.

### 6. Normalize Subtitles
```bash
python src/normalize_subtitles.py
# Or use the Makefile
make normalize
```
Use your trained model to normalize new SRT files.

## Example

**Input (SDH):**
```
[MUSIC PLAYING]
JOHN: Hello, how are you
today?
[door slams]
```

**Output (Normalized):**
```
Hello, how are you today?
```

## Project Structure

```
sdh-subtitle-normalizer/
├── src/                          # Source code
│   ├── srt_analyzer.py          # Analyze SRT files for SDH patterns
│   ├── training_data_generator.py # Generate training data
│   ├── train_model.py           # Train the T5 model
│   ├── normalize_subtitles.py   # Normalize subtitles with trained model
│   └── __init__.py
├── data/                         # Data storage
│   ├── raw_srt/                 # Place your original SRT files here
│   ├── sample_data/             # Sample files for testing
│   ├── training_data.json       # Generated training pairs
│   └── README.md
├── models/                       # Trained models
│   ├── subtitle_normalizer/     # Main trained model
│   └── README.md
├── output/                       # Normalized results
│   ├── normalized_srt/          # Normalized SRT files
│   ├── batch_results/           # Batch processing results
│   └── README.md
├── config.json                   # Configuration settings
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Makefile                      # Common tasks
└── README.md                     # This file
```

## Training Tips

1. **More data = Better results**: Collect diverse SRT files with various SDH patterns
2. **Model size**: Use `t5-base` or `t5-large` for better quality (but slower training)
3. **Training time**: 3-5 epochs usually sufficient, watch validation accuracy
4. **GPU recommended**: Training will be much faster with CUDA

## Two Approaches Available

### 🎯 **Style Transfer (Recommended)**
Train on clean examples and apply that style to dirty subtitles:
1. **Learn from clean files**: Model studies your perfect subtitles
2. **Apply to dirty files**: Transforms messy subtitles to match clean style
3. **Commands**: `make style-generate`, `make style-train`, `make style-normalize`

### 🔧 **Pattern Removal (Alternative)**
Train on dirty→clean pairs to learn specific removal patterns:
1. **Learn removal patterns**: Model learns to remove `[MUSIC]`, `JOHN:`, etc.
2. **Apply transformations**: Removes known SDH patterns
3. **Commands**: `make generate-data`, `make train`, `make normalize`

## Model Architecture

Uses **T5 (Text-to-Text Transfer Transformer)** with different prompts:
- **Style Transfer**: "clean subtitle style: [MUSIC PLAYING] Hello there" → "Hello there"
- **Pattern Removal**: "normalize subtitle: [MUSIC PLAYING] Hello there" → "Hello there"

## Advanced Usage

### Custom SDH Patterns
Edit the `sdh_patterns` dictionary in `srt_analyzer.py` to add custom patterns.

### Batch Processing
Use the batch mode in `normalize_subtitles.py` to process entire directories.

### Model Fine-tuning
Adjust training parameters in `train_model.py`:
- Learning rate: `5e-5` (default) to `1e-4`
- Batch size: `8` (default) or `16` if you have more memory
- Epochs: `3` (default) to `5-10` for better results

## Troubleshooting

**Out of Memory**: Reduce batch size or use `t5-small` instead of `t5-base`

**Poor Results**: Need more diverse training data or longer training

**Encoding Issues**: The analyzer handles common encodings automatically

**Model Not Found**: Make sure you've trained a model first with `train_model.py`

## Performance

- **t5-small**: Fast training, good results for common SDH patterns
- **t5-base**: Better quality, slower training, handles complex cases
- **Training time**: ~30min for 10k samples on GPU, ~2-3 hours on CPU

## Next Steps

1. **Collect more SRT files** with diverse SDH patterns
2. **Experiment with model sizes** (t5-base, t5-large)
3. **Add evaluation metrics** (BLEU, ROUGE scores)
4. **Create web interface** for easy file uploads
5. **Support other subtitle formats** (VTT, ASS)
