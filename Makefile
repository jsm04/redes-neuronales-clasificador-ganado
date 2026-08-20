.PHONY: install preprocess train evaluate test clean all

install:
	pip install -r requirements.txt

preprocess:
	python src/dataset.py --mode preprocess

train:
	python src/train.py

evaluate:
	python src/evaluate.py

test:
	pytest tests/ -v --cov=src

clean:
	rm -rf output/* models/*.keras

all: preprocess train evaluate
	@echo "Pipeline completo ejecutado"
