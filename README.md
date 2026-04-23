# Enabling safe AI deployment: an automated Fitzpatrick skin type guardrail for out-of-distribution dermatology

This repository implements an automated Fitzpatrick skin type guardrail intended to help detect out-of-distribution (OOD) dermatology images and allow safe deployment of AI models.

## Contents

- **preprocessing**: scripts for extracting masks and patches from images. Key modules: `preprocessing.extract_masks.py`, `preprocessing.extract_patches.py`.
- **train**: training and evaluation scripts. Key modules: `train.train.py`, `train.train_bencevic.py`, `train.validation.py`, `train.statistical.py`.
- **test**: evaluation on the test set: `test.test.py`, `test.test_bencevic.py`, `test.test_kinyanjui.py`
- **results/** and **results_test/**: experiment outputs, logs, and metrics.
- `config.py`: global configuration used by scripts and experiments.

## Prerequisites

- Python 3.10+
- Typical workflow assumes standard scientific packages (numpy, pandas, torch, torchvision, scikit-learn, tqdm).

- Edit `config.py` to adjust global settings (paths, hyperparameters, dataset options) used by preprocessing, training, and evaluation scripts.


## Preprocessing

- Extract masks:

```
python -m preprocessing.extract_masks
```

- Extract patches:

```
python -m preprocessing.extract_patches
```

## Cross-Validation

```
python -m train.validation
```

Experiment outputs (models, logs, metrics) are written to the `results/` directory.

### Statistical Tests

```
python -m train.statistical
```

## Training SOTA methods

### Bencevic (2024)
```
python -m train.train_bencevic
```

## Testing

### Ours

```
python -m test.test
```
Results are saved under results_test/ours.


### Bencevic (2024)
```
python -m test.test_bencevic 
```

Results are saved under results_test/bencevic.


### Kinyanjui (2019)
```
python -m test.test_kinyanjui
```

Results are saved under results_test/kinyanjui.
