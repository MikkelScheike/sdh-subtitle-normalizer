# SDH Subtitle Normalizer Makefile

.PHONY: install setup analyze generate-data train normalize style-generate style-train style-train-base style-normalize clean help

help:
	@echo "SDH Subtitle Normalizer - Available commands:"
	@echo "  install           - Install dependencies"
	@echo "  setup             - Setup development environment"
	@echo "  analyze           - Analyze SRT files"
	@echo "  generate-data     - Generate training data (pattern removal)"
	@echo "  train             - Train the model (pattern removal)"
	@echo "  normalize         - Normalize subtitles (pattern removal)"
	@echo "  style-generate    - Generate style transfer training data"
	@echo "  style-train       - Train style transfer model (defaults)"
	@echo "  style-train-base  - Train style transfer with t5-base, batch-size=4"
	@echo "  style-normalize   - Apply style transfer normalization"
	@echo "  clean             - Clean temporary files"

install:
	pip install -r requirements.txt

setup:
	pip install -e .

analyze:
	python src/srt_analyzer.py

generate-data:
	python src/training_data_generator.py

train:
	python src/train_model.py

normalize:
	python src/normalize_subtitles.py

style-generate:
	python src/style_transfer_generator.py

style-train:
	python src/style_transfer_trainer.py

# One-liner for training t5-base on the pre-generated style transfer data
style-train-base:
	python3 -m src.style_transfer_trainer \
		--data-file="data/style_transfer_data.json" \
		--output-dir="models/style_transfer_normalizer_base" \
		--epochs=5 \
		--batch-size=4 \
		--model-name="t5-base" \
		--learning-rate=3e-5

style-normalize:
	python src/style_transfer_normalizer.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
