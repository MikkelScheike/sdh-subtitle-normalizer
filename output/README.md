# Output Directory

This directory contains normalized subtitle files and processing results.

## Structure
```
output/
├── normalized_srt/        # Normalized SRT files
├── batch_results/         # Batch processing results
├── logs/                  # Processing logs
└── reports/               # Analysis reports
```

## Contents

- **normalized_srt/**: Individual normalized subtitle files
- **batch_results/**: Results from batch processing entire directories
- **logs/**: Processing logs and error reports
- **reports/**: Analysis reports and statistics

## File Naming

Normalized files follow this pattern:
- Original: `movie.srt`
- Normalized: `movie_normalized.srt`

## Batch Processing

When processing directories, the folder structure is preserved:
```
input/
├── movies/
│   ├── movie1.srt
│   └── movie2.srt
└── tv_shows/
    └── episode1.srt

output/normalized_srt/
├── movies/
│   ├── movie1.srt
│   └── movie2.srt
└── tv_shows/
    └── episode1.srt
```
