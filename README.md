# CVAE-ScratchGAN Password guesser

A password guessing tool based on a hybrid CVAE-ScratchGAN architecture.

## File structure

```
├── config.yaml                  # Configuration file with hyperparameters
├── train.py                     # Script for training the model
├── generate.py                  # Script for generating passwords
├── src/
│   ├── config.py                # Loads configuration from config.yaml
│   ├── dataset.py               # Password dataset class
│   ├── generator.py             # Password generation
│   ├── model.py                 # Definition of CVAE-ScratchGAN architecture
│   ├── pcfg.py                  # PCFG utilities
│   ├── tokenizer.py             # Tokenizers
│   ├── trainer.py               # Training loop
│   └── vocab.py                 # Vocabulary utilities
├── src_datasets/
│   ├── dataset_filter.py        # Filters password datasets
│   └── dataset_splitter.py      # Splits dataset into train and test sets
├── pyproject.toml
├── uv.lock
├── LICENSE
└── models/                      # Pre-trained models on RockYou and LinkedIn datasets
```

---

## Requirements and installation

- **Python 3.12** or newer
- **uv**

Install dependencies before first use:

```
uv sync
```

---

## Quick start

A complete example from training to generating passwords:

```bash
# 1. Install dependencies
uv sync

# 2. Filter the dataset
uv run python src_datasets/dataset_filter.py \
    --input-dataset-path rockyou.txt

# 3. Split dataset into train/test set
uv run python src_datasets/dataset_splitter.py \
    --input-dataset-path passwords_cleaned.txt

# 4. Train the model
uv run python train.py --dataset-path train_passwords.txt

# 5. Build PCFG structures
uv run python generate.py --mode build-pcfg \
    --dataset-path train_passwords.txt

# 6. Generate passwords
uv run python generate.py --mode generate \
    --pcfg-path pcfg/pcfg_structures.json \
    --model-path checkpoints/trained_model_epoch_30.pt
```

---

## Using pre-trained models

The `models/` directory contains models trained on the RockYou and LinkedIn datasets.

Each model directory contains:
- `trained_model_epoch_30.pt` — trained model
- `vocab.json` — vocabulary
- `pcfg_structures.json` — PCFG structures from the train set

To generate passwords with a pre-trained model:

```bash
uv run python generate.py --mode generate \
    --pcfg-path models/rockyou/pcfg_structures.json \
    --model-path models/rockyou/trained_model_epoch_30.pt
```

---

## Dataset preparation

Dataset preparation for training consists of two phases.

### Phase 1: Filtering

```
uv run python src_datasets/dataset_filter.py --input-dataset-path <path>
```

**Required arguments:**
- `--input-dataset-path <path>` — Path to the dataset.

**Optional arguments:**
- `--output-dataset-path <file>` — Output path for the filtered dataset (default: `./passwords_cleaned.txt`).
- `--min-len <N>` — Minimum password length (default: `4`).
- `--max-len <N>` — Maximum password length (default: `15`).
- `--split-colon` — If set, extracts only the part after the first colon (for datasets in `hash:password` format).

### Phase 2: Splitting

```
uv run python src_datasets/dataset_splitter.py --input-dataset-path <path>
```

**Required arguments:**
- `--input-dataset-path <path>` — Path to the dataset.

**Optional arguments:**
- `--train-output-path <file>` — Output path for the train set (default: `./train_passwords.txt`).
- `--test-output-path <file>` — Output path for the test set (default: `./test_passwords.txt`).
- `--train-ratio <X>` — Train/test split ratio (default: `0.8`).

---

## Training model

```
uv run python train.py --dataset-path <path>
```

**Required arguments:**
- `--dataset-path <path>` — Path to the train dataset.

**Optional arguments:**
- `--checkpoint-dir <path>` — Directory for saving model checkpoints (default: `checkpoints/`).
- `--vocab-path <file>` — Path for saving the vocabulary (default: `<checkpoint-dir>/vocab.json`).
- `--epochs <N>` — Number of training epochs (default: `30`).
- `--batch-size <N>` — Training batch size (default: `1024`).
- `--seed <N>` — Random seed (default: `42`).
- `--num-workers <N>` — Number of DataLoader worker processes (default: `1`).
- `--bf16` — Enable bfloat16 mixed precision.

---

## Generating passwords

Generation consists of two phases.

### Phase 1: Build PCFG

Before generating for the first time, build the PCFG structure from a train dataset:

```
uv run python generate.py --mode build-pcfg --dataset-path <path>
```

**Required arguments:**
- `--mode build-pcfg` — Mode for building the PCFG.
- `--dataset-path <path>` — Path to the dataset.

**Optional arguments:**
- `--pcfg-save-path <path>` — Directory to save the PCFG (default: `pcfg/`).

### Phase 2: Generate

Two modes are available: `generate` writes a fixed number of passwords to a file; `infinite` streams passwords to stdout indefinitely.

```
uv run python generate.py --mode {generate|infinite} \
    --pcfg-path <file> --model-path <file>
```

**Required arguments:**
- `--mode {generate|infinite}` — Generation mode.
- `--pcfg-path <file>` — Path to the PCFG JSON file.
- `--model-path <file>` — Path to the trained model.

**Optional arguments:**
- `--output-path <file>` — Output file (default: `generated_passwords.txt`).
- `--vocab-path <file>` — Path to the vocabulary (default: `vocab.json` in the same directory as the model).
- `--num-gen-samples <N>` — Number of passwords to generate (default: `1 000 000`).
- `--batch-size <N>` — Generation batch size (default: `10 000`).
- `--temp <X>` — Sampling temperature (default: `0.85`).
- `--top-p <X>` — Top-P sampling parameter (default: `0.85`).
- `--top-k <N>` — Top-K sampling parameter (default: `35`).
- `--max-gen-pass-len <N>` — Maximum generated password length (default: `15`).
- `--bf16` — Enable bfloat16 mixed precision.
- `--seed <N>` — Random seed (default: `42`).

**Policy filtering arguments** (for guided generation):
- `--policy-min-len <N>` — Minimum length of sampled structures.
- `--policy-max-len <N>` — Maximum length of sampled structures.
- `--policy-min-lower <N>` — Minimum number of lowercase letters.
- `--policy-min-upper <N>` — Minimum number of uppercase letters.
- `--policy-min-digit <N>` — Minimum number of digits.
- `--policy-min-special <N>` — Minimum number of special characters.

---

## Output file structure

### Training model

After training completes, the following files are created in `checkpoints/` (default):

| File | Description |
|------|-------------|
| `vocab.json` | Character vocabulary. |
| `trained_model_epoch_N.pt` | Trained model after epoch N. |
| `training_metrics_epoch.csv` | Per-epoch averaged losses. Columns: `epoch` — epoch number; `total` — total generator loss; `ce` — cross-entropy reconstruction loss; `kl` — KL divergence; `aux` — auxiliary head losses; `d_loss` — discriminator loss; `g_adv` — generator adversarial loss; `d_real` — discriminator score on real samples; `d_fake` — discriminator score on generated samples. |
| `training_metrics_batch.csv` | Same columns as above plus `batch`, logged every `LOG_EVERY_BATCHES` (config.yaml) batches as a rolling window average. |

Training has completed successfully when the console outputs:
```
Training complete.
```

### Building PCFG

After running `--mode build-pcfg`, the following file is created in `pcfg/` (default):

| File | Description |
|------|-------------|
| `pcfg_structures.json` | Password structures and their probabilities. |

### Generating passwords

After running `--mode generate`, the following file is created:

| File | Description |
|------|-------------|
| `generated_passwords.txt` | Plain text file with generated passwords, one per line. |

---

## Example output files

### `vocab.json`

Maps characters to their IDs.

```json
{
  "<PAD>": 0,
  "<BOS>": 1,
  "<EOS>": 2,
  "<UNK>": 3,
  "!": 4,
  "\"": 5,
  "#": 6,
  "$": 7,
  "%": 8,
  "&": 9,
  ...
}
```

### `training_metrics_epoch.csv`

```
epoch,total,ce,kl,aux,d_loss,g_adv,d_real,d_fake
1,2.0158,1.5479,1022.9977,0.4679,0.0,0.0,0.0,0.0
2,1.7166,1.1597,62.2738,0.1418,0.0,0.0,0.0,0.0
...
30,3.1386,1.8492,10.4047,0.2974,1.2670,-0.1944,0.5532,0.4467
```

### `training_metrics_batch.csv`

```
epoch,batch,total,ce,kl,aux,d_loss,g_adv,d_real,d_fake
1,100,19.9031,3.6705,291.3710,16.2326,0.0,0.0,0.0,0.0
1,200,5.5486,3.0965,504.3787,2.4521,0.0,0.0,0.0,0.0
...
30,10900,3.1351,1.8454,10.4170,0.2967,1.2667,-0.1949,0.5532,0.4467
```

### `pcfg_structures.json`

Maps password structures to their probabilities. (`L` = lowercase, `U` = uppercase, `N` = digit, `S` = special).

```json
{
  "LLLLLLLL": 0.08312,
  "LLLLLLLLN": 0.05241,
  "LLLLNNNN": 0.04187,
  "LLLLLNNN": 0.03902,
  "ULLLLLLLL": 0.02740,
  ...
}
```

### `generated_passwords.txt`

One password per line:

```
Monkie12
Rornie88
Casper11
Crawks17
...
```

---

## Acknowledgements

GitHub Copilot was used as a code completion and refactoring assistant during development.

---

## License

This project is licensed under the MIT License.
