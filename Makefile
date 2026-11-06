# Reproducibility entry points. Run from the repo root.
PY ?= python

.PHONY: setup test synthetic mars finetune demo errors clean

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

# OSNet fine-tuning on MARS train (needs torch + torchreid). Usage: make finetune TRAIN_ZIP=/path/bbox_train.zip DEVICE=mps
finetune:
	$(PY) scripts/finetune_osnet.py --zip "$(TRAIN_ZIP)" --out models/osnet_ft --device $(or $(DEVICE),mps)

# Review viewer data from the fine-tuned embeddings (thumbnails need the MARS zip). Usage: make demo ZIP=/path/bbox_test.zip
demo:
	$(PY) scripts/build_demo.py --features data/embeddings/mars_test_osnet_ft.npy --names data/embeddings/mars_test_osnet_ft_names.txt --zip "$(ZIP)" --out demo/data

# Error taxonomy on extracted MARS embeddings.
errors:
	$(PY) scripts/error_analysis.py --features data/embeddings/mars_test.npy \
	  --names data/embeddings/mars_test_names.txt --pooling mean --out results/mars_resnet50_frozen/errors

clean:
	rm -rf .pytest_cache **/__pycache__
