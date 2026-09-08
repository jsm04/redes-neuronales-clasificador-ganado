# AGENTS.md — AI Agent Instructions

## Project Overview

This is a university machine-learning project for **image classification of cattle**.

The current task is image classification using a convolutional neural network (CNN). Do not introduce support for unrelated problem types such as text classification, time series, speech recognition, or sequence models unless explicitly requested.

The project uses:

* Python
* `uv` for environment and dependency management
* TensorFlow / Keras for the neural network
* `tf.data` for dataset loading
* scikit-learn for evaluation metrics
* Matplotlib for plots
* PyYAML for configuration

The project is intended to remain small and understandable. Prefer simple, explicit implementations over unnecessary abstractions.

---

## Project Structure

```text
.
├── AGENTS.md
├── README.md
├── config.yaml
├── pyproject.toml
├── uv.lock
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│
├── data/
│   └── raw/
│       ├── cattle/
│       └── not_cattle/
│
└── output/
```

Responsibilities:

| File              | Responsibility                                                     |
| ----------------- | ------------------------------------------------------------------ |
| `config.yaml`     | Central project configuration                                      |
| `src/dataset.py`  | Dataset discovery, splitting, loading, preprocessing, augmentation |
| `src/model.py`    | CNN architecture                                                   |
| `src/train.py`    | Training and training callbacks                                    |
| `src/evaluate.py` | Test evaluation and reports                                        |
| `tests/`          | Automated tests                                                    |
| `output/`         | Generated models, metrics, plots, and reports                      |

Do not create additional modules unless there is a clear reason to do so.

---

## Dataset

The dataset uses directory-based image classification:

```text
data/raw/
├── cattle/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
└── not_cattle/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

Each immediate subdirectory of `data/raw/` represents one class.

Classes should be discovered from the filesystem rather than hardcoded in the Python source.

Do not move or modify the original images during preprocessing.

The pipeline should load images through TensorFlow datasets rather than loading the entire dataset into NumPy arrays.

---

## Configuration

`config.yaml` is the source of truth for experiment configuration.

Read it before modifying or executing training/evaluation code.

Important configuration areas include:

```yaml
data:
  raw_dir:
  batch_size:
  train_split:
  validation_split:
  test_split:
  random_seed:
  augmentation:

model:
  architecture:
  input_shape:
  num_classes:
  regularization:

training:
  epochs:
  learning_rate:
  optimizer:
  early_stopping:
  checkpointing:
  reduce_lr:

environment:
  output_dir:
```

Do not hardcode configurable hyperparameters in `train.py`, `dataset.py`, or `model.py`.

`num_classes` may be derived from the discovered dataset classes. Do not assume that the configured value is authoritative if the dataset contains a different number of classes.

---

## Data Splitting

The dataset must be divided into:

```text
Train      70%
Validation 15%
Test       15%
```

using the values defined in `config.yaml`.

The split must be:

* deterministic when using the configured random seed;
* performed at the image/file level;
* mutually exclusive;
* reproducible between runs.

The test set must remain isolated from training and model-selection decisions.

Do not use the test set to tune hyperparameters.

Do not apply training augmentation to validation or test data.

---

## Model

The current model is a CNN for image classification.

The model should:

* accept the configured `input_shape`;
* produce `num_classes` outputs;
* use an appropriate classification loss;
* use the configured optimizer and learning rate;
* use regularization only where configured.

Prefer `GlobalAveragePooling2D` over unnecessarily large `Flatten` layers when appropriate.

Do not introduce a more complex architecture such as ResNet, VGG, Transformer, LSTM, or transfer learning unless explicitly requested.

Keep the architecture understandable enough for a university project.

---

## Training

Training must use the datasets produced by `dataset.py`.

The training pipeline should:

1. Load `config.yaml`.
2. Discover the dataset classes.
3. Create train, validation, and test datasets.
4. Apply augmentation only to the training dataset.
5. Build the configured model.
6. Train using the configured hyperparameters.
7. Apply configured callbacks.
8. Save the best model.
9. Save the final model.
10. Save training metrics and a summary.

Use:

```bash
uv run python src/train.py
```

Do not use:

```bash
python src/train.py
```

when documenting or recommending project commands.

Do not use `pip install` for normal project setup. Dependencies are managed through `uv`.

---

## Evaluation

Evaluation must use the test dataset.

Use:

```bash
uv run python src/evaluate.py
```

or explicitly specify a model:

```bash
uv run python src/evaluate.py --model output/best_model.keras
```

Evaluation should report, where applicable:

* loss;
* accuracy;
* precision;
* recall;
* F1-score;
* per-class metrics;
* confusion matrix.

Generated evaluation artifacts belong in `output/`.

Never silently substitute the validation set for the test set.

---

## Reproducibility

Preserve reproducibility whenever possible.

Use the configured random seed.

Do not randomly change:

* dataset splits;
* hyperparameters;
* architecture;
* augmentation;
* dependency versions

without an explicit reason.

The files:

```text
pyproject.toml
uv.lock
config.yaml
```

are important to reproducibility and should be kept synchronized with the implementation.

---

## Dependencies

Use `uv` for dependency management.

Install/synchronize the environment:

```bash
uv sync
```

Add a runtime dependency:

```bash
uv add <package>
```

Add a development dependency:

```bash
uv add --dev <package>
```

Remove a dependency:

```bash
uv remove <package>
```

Do not introduce a second dependency-management system.

`requirements.txt`, if present, is compatibility-only and is not the authoritative dependency definition.

---

## Commands

From the project root:

```bash
# Synchronize the environment
uv sync

# Inspect/validate the dataset
uv run python src/dataset.py

# Train
uv run python src/train.py

# Evaluate
uv run python src/evaluate.py

# Run tests
uv run pytest -v
```

Do not rely on the old Makefile workflow.

There is no separate preprocessing stage that generates `.npy` files. Dataset loading and preprocessing are handled by `dataset.py` as part of the TensorFlow input pipeline.

---

## Output Files

Generated artifacts belong in:

```text
output/
```

Typical artifacts include:

```text
output/
├── best_model.keras
├── final_model.keras
├── class_names.json
├── training_metrics.csv
├── training_summary.json
├── confusion_matrix.csv
├── training_history.png
└── evaluation_*.json
```

Do not place generated models, metrics, plots, or temporary files inside `src/`.

Do not overwrite an existing experiment artifact without a reason.

---

## Code Changes

When modifying the project:

1. Read `config.yaml`.
2. Inspect the relevant existing implementation.
3. Preserve the existing architecture unless the task requires changing it.
4. Make the smallest change that correctly solves the problem.
5. Keep configuration in `config.yaml`.
6. Run relevant tests.
7. Run the affected script when practical.
8. Check that generated output is correct.

Do not rewrite unrelated files.

Do not add abstractions merely for theoretical extensibility.

Do not add dependencies when the existing stack can solve the problem.

---

## Experimentation

Do not autonomously start repeated training runs or modify hyperparameters based solely on evaluation results.

If a model performs poorly:

1. Inspect the training and validation metrics.
2. Determine whether the evidence suggests underfitting, overfitting, data-quality problems, or another issue.
3. Explain the proposed change.
4. Apply the change only when requested or when the task explicitly authorizes experimentation.
5. Keep experiments reproducible.

Avoid simplistic rules such as:

```text
accuracy < 0.70 = underfitting
```

Accuracy alone is not sufficient to diagnose model behavior.

Use training/validation loss and metrics, class distribution, confusion matrices, and the dataset itself when diagnosing problems.

---

## GPU

The project should use an available NVIDIA GPU automatically when TensorFlow has GPU support configured.

Check GPU visibility with:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Do not modify CUDA, TensorFlow, or driver configuration automatically.

If TensorFlow cannot access the GPU, report the issue and diagnose the environment before changing dependencies.

---

## Safety and Data Integrity

Never:

* delete the raw dataset;
* modify original images;
* use test data for training;
* leak test data into training or validation;
* silently change experiment configuration;
* overwrite important model artifacts without warning;
* install global Python dependencies;
* replace `uv` with another environment manager.

When a requested change conflicts with reproducibility or data isolation, point out the conflict before proceeding.

---

## Definition of Done

A change is complete when:

* the implementation matches `config.yaml`;
* train/validation/test isolation is preserved;
* the code remains consistent with the current project architecture;
* relevant tests pass;
* training/evaluation commands still work;
* generated artifacts are written to `output/`;
* no obsolete workflow has been reintroduced.

```

The biggest improvement is that this `AGENTS.md` is now a **constraint document**, not a description of an imaginary autonomous ML platform. It tells the agent what the project actually is, what it must preserve, which commands to use, and—importantly—what it must **not** do autonomously.

I would also **not create `MEMORY.md`/`memoria.md` yet**. For a small university project, Git history + `config.yaml` + `uv.lock` + generated experiment metrics are enough until you actually have multiple experiments that need a formal experiment log.
```

