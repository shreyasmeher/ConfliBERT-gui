# ConfliBERT GUI

[ConfliBERT](https://github.com/eventdata/ConfliBERT) is a pretrained language model built specifically for analyzing conflict and political violence text. This application provides a browser-based interface for running inference with ConfliBERT's pretrained models, fine-tuning custom classifiers on your own data, and comparing model performance across architectures.

## Screenshots

### Home

The landing page shows your system configuration (GPU/CPU, RAM, platform) and an overview of everything the app can do.

<!-- Take a screenshot of the Home tab and save as screenshots/home.png -->
![Home](./screenshots/home.png)

### Named Entity Recognition

Identifies persons, organizations, locations, weapons, and other entity types. Results are color-coded. Supports single text and CSV batch processing.

<!-- Take a screenshot of the NER tab with sample output and save as screenshots/ner.png -->
![NER](./screenshots/ner.png)

### Binary Classification

Classifies text as conflict-related or not. Uses the pretrained ConfliBERT classifier by default, or load your own fine-tuned model.

<!-- Take a screenshot of the Classification tab and save as screenshots/classification.png -->
![Classification](./screenshots/classification.png)

### Multilabel Classification

Scores text against four event categories (Armed Assault, Bombing/Explosion, Kidnapping, Other). Each category is scored independently.

<!-- Take a screenshot of the Multilabel tab and save as screenshots/multilabel.png -->
![Multilabel](./screenshots/multilabel.png)

### Question Answering

Provide a context passage and a question. The model extracts the most relevant answer span.

<!-- Take a screenshot of the QA tab and save as screenshots/qa.png -->
![QA](./screenshots/qa.png)

### Fine-tuning

Train your own binary or multiclass classifier directly in the browser. Upload data (or load a built-in example), pick a base model, configure training, and go. Supports **LoRA** and **QLoRA** for parameter-efficient training with lower VRAM usage. After training, results and a "Try Your Model" panel appear side by side. You can also save the model and run batch predictions.

### Model Comparison

Compare multiple base model architectures on the same dataset. The comparison produces a metrics table, a grouped bar chart, and ROC-AUC curves.

<!-- Take a screenshot of the Fine-tune tab and save as screenshots/finetune.png -->
![Fine-tune](./screenshots/finetune.png)

### Active Learning

Iteratively build a strong classifier with fewer labels. Start with a small labeled seed set and a pool of unlabeled text. The model identifies the most uncertain samples for you to label, retrains, and repeats. Supports entropy, margin, and least-confidence query strategies.

## Supported Models

### Pretrained (Inference)

| Task | HuggingFace Model |
|------|-------------------|
| NER | `eventdata-utd/conflibert-named-entity-recognition` |
| Binary Classification | `eventdata-utd/conflibert-binary-classification` |
| Multilabel Classification | `eventdata-utd/conflibert-satp-relevant-multilabel` |
| Question Answering | `salsarra/ConfliBERT-QA` |

### Fine-tuning (Base Models)

| Model | HuggingFace ID | Notes |
|-------|----------------|-------|
| ConfliBERT | `snowood1/ConfliBERT-scr-uncased` | Best for conflict/political text |
| BERT Base Uncased | `bert-base-uncased` | General-purpose baseline |
| BERT Base Cased | `bert-base-cased` | Case-sensitive variant |
| RoBERTa Base | `roberta-base` | Improved BERT training |
| ModernBERT Base | `answerdotai/ModernBERT-base` | Up to 8K token context |
| DeBERTa v3 Base | `microsoft/deberta-v3-base` | Strong on benchmarks |
| DistilBERT Base | `distilbert-base-uncased` | Faster, smaller |

## Installation

### Requirements

- Python 3.8+
- Git

### Steps

1. Clone the repository:

```bash
git clone https://github.com/shreyasmeher/conflibert-gui.git
cd conflibert-gui
```

2. Create and activate a virtual environment:

```bash
python -m venv env

# Mac/Linux:
source env/bin/activate

# Windows:
env\Scripts\activate
```

On Windows, if you get a permission error, run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

3. Install PyTorch:

```bash
# CPU only (Mac, or no NVIDIA GPU):
pip install torch

# NVIDIA GPU (Windows/Linux):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

4. Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the application:

```bash
python app.py
```

Opens at `http://localhost:7860` and generates a public shareable link. The first launch takes a minute or two while it downloads the pretrained models.

### Tabs

| Tab | What it does |
|-----|-------------|
| Home | System info, feature overview, citation |
| Named Entity Recognition | Identify entities in text or CSV |
| Binary Classification | Conflict vs. non-conflict, supports custom models |
| Multilabel Classification | Multi-event-type scoring |
| Question Answering | Extract answers from a context passage |
| Fine-tune | Train classifiers with optional LoRA/QLoRA, compare models, ROC curves |
| Active Learning | Iterative uncertainty-based labeling and retraining |

### Fine-tuning Quick Start

1. Go to the **Fine-tune** tab
2. Click **"Load Example: Binary"** to load sample data
3. Leave defaults and click **"Start Training"**
4. Review metrics and try your model on new text
5. Save the model and load it in the **Binary Classification** tab

### LoRA / QLoRA Fine-tuning

1. Go to the **Fine-tune** tab
2. Open **Advanced Settings** and check **Use LoRA** (optionally enable **QLoRA** for 4-bit quantization on CUDA GPUs)
3. Adjust LoRA rank and alpha as needed (defaults of r=8, alpha=16 work well)
4. Train as usual — LoRA weights are merged back automatically so the saved model works like any other

### Model Comparison Quick Start

1. Upload data (or load an example) in the **Fine-tune** tab
2. Scroll down and open **"Compare Multiple Models"**
3. Check 2 or more models to compare
4. Click **"Compare Models"**
5. View the metrics table, bar chart, and ROC-AUC curves

### Active Learning Quick Start

1. Go to the **Active Learning** tab
2. Click **"Load Example: Binary Active Learning"** (or upload your own seed + pool)
3. Configure the query strategy and samples per round
4. Click **"Initialize Active Learning"**
5. Label the uncertain samples shown in the table (fill in 0 or 1)
6. Click **"Submit Labels & Next Round"** to retrain and get the next batch
7. Repeat until satisfied, then save the model

### Data Format

Tab-separated values (TSV), no header row. Each line: `text<TAB>label`

Binary example:
```
The bomb exploded near the market	1
It was a sunny day at the park	0
```

Multiclass example (integer labels starting from 0):
```
The president signed the peace treaty	0
Militants attacked the military base	1
Thousands marched in the capital	2
Aid workers delivered food supplies	3
```

### CSV Batch Processing

Prepare a CSV with a `text` column:

```csv
text
"The soldiers advanced toward the border."
"The festival attracted thousands of visitors."
```

Upload it in the Batch Processing section of any inference tab.

## Project Structure

```
conflibert-gui/
  app.py                 # Main application
  requirements.txt       # Dependencies
  README.md
  screenshots/           # UI screenshots for documentation
  examples/
    binary/              # Example binary dataset (conflict vs non-conflict)
      train.tsv
      dev.tsv
      test.tsv
    multiclass/          # Example multiclass dataset (4 event types)
      train.tsv          #   0=Diplomacy, 1=Armed Conflict,
      dev.tsv            #   2=Protest, 3=Humanitarian
      test.tsv
    active_learning/     # Example active learning dataset
      seed.tsv           #   20 labeled seed samples
      pool.txt           #   61 unlabeled pool texts
      pool_with_labels.tsv  # Ground truth for pool (cheat sheet)
```

## Training Features

- **LoRA / QLoRA** parameter-efficient fine-tuning (via [PEFT](https://github.com/huggingface/peft))
- **Active learning** with entropy, margin, and least-confidence query strategies
- Early stopping with configurable patience
- Learning rate schedulers: linear, cosine, constant, constant with warmup
- Mixed precision training (FP16) on CUDA GPUs
- Gradient accumulation for larger effective batch sizes
- Weight decay regularization
- Automatic system detection (NVIDIA GPU, Apple Silicon MPS, CPU)
- Model comparison with grouped bar charts and ROC-AUC curves

## Citation

If you use ConfliBERT in your research, please cite:

Brandt, P.T., Alsarra, S., D'Orazio, V., Heintze, D., Khan, L., Meher, S., Osorio, J. and Sianan, M., 2025. Extractive versus Generative Language Models for Political Conflict Text Classification. *Political Analysis*, pp.1-29.

```bibtex
@article{brandt2025extractive,
  title={Extractive versus Generative Language Models for Political Conflict Text Classification},
  author={Brandt, Patrick T and Alsarra, Sultan and D'Orazio, Vito and Heintze, Dagmar and Khan, Latifur and Meher, Shreyas and Osorio, Javier and Sianan, Marcus},
  journal={Political Analysis},
  pages={1--29},
  year={2025},
  publisher={Cambridge University Press}
}
```

## License

MIT License. See LICENSE for details.
