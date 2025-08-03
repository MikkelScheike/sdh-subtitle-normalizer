# Training Data Workflow

## Proper Training Setup

For best results, you need **paired data**: raw files and your cleaned versions.

### Directory Structure:
```
data/
├── raw_srt/           # Original, uncleaned SRT files
│   └── 28 Years Later (2025).en.srt
├── clean_srt/         # Your manually cleaned versions
│   └── 28 Years Later (2025).en.srt
└── training_data.json # Generated training pairs
```

### Workflow:

1. **Add raw file** to `data/raw_srt/`
2. **Clean it manually** and save to `data/clean_srt/` (same filename)
3. **Run training generator** - it will create pairs automatically
4. **Train model** on your actual cleaning standards

### What to Clean:
- Remove: YTS.MX commercials, [MUSIC], speaker labels, etc.
- Fix: Case, line breaks, formatting
- Keep: Actual dialogue content

### Benefits:
✅ Model learns YOUR specific cleaning style
✅ Consistent quality standards
✅ Real-world training pairs
✅ Reproducible results

## Alternative: Start with Clean Files Only

If you have clean files, use synthetic SDH generation:
- Put clean files in `data/clean_srt/`
- Generator adds synthetic SDH → creates training pairs
- Model learns to remove common SDH patterns
