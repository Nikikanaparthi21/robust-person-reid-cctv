# Reproducibility entry points. Run from the repo root.
PY ?= python

.PHONY: setup test synthetic mars errors clean

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	$(PY) -m pytest -q

# Pipeline sanity check on the synthetic benchmark (no data needed; NOT a research result).
synthetic:
	$(PY) scripts/run_synthetic.py

# Real MARS baseline. Usage: make mars ZIP="/path/to/bbox_test.zip" DEVICE=mps
mars:
	bash scripts/run_mars_baseline.sh "$(ZIP)" $(or $(DEVICE),mps)

# Error taxonomy on extracted MARS embeddings.
errors:
	$(PY) scripts/error_analysis.py --features data/embeddings/mars_test.npy \
	  --names data/embeddings/mars_test_names.txt --pooling mean --out results/mars_resnet50_frozen/errors

clean:
	rm -rf .pytest_cache **/__pycache__
