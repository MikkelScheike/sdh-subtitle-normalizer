# Models Directory

This directory stores trained models and checkpoints.

## Structure
```
models/
├── subtitle_normalizer/    # Main trained model
│   ├── config.json         # Model configuration
│   ├── pytorch_model.bin   # Model weights
│   ├── tokenizer.json      # Tokenizer
│   └── ...
├── checkpoints/            # Training checkpoints
└── experiments/            # Different model experiments
```

## Usage

- **Training**: Models are automatically saved here after training
- **Inference**: Load models from this directory for normalization
- **Backup**: Keep backups of your best models

## Model Files

When training completes, you'll find these files:
- `config.json` - Model configuration
- `pytorch_model.bin` - Trained weights  
- `tokenizer.json` - Tokenizer settings
- `training_args.bin` - Training arguments
- `trainer_state.json` - Training state
