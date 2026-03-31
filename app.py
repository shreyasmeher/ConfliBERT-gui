# ============================================================================
# ConfliBERT - Conflict & Political Violence NLP Toolkit
# University of Texas at Dallas | Event Data Lab
# ============================================================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import tensorflow as tf
import tf_keras  # noqa: F401 - needed for TF model loading
import keras     # noqa: F401 - needed for TF model loading
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TFAutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
import gradio as gr
import numpy as np
import pandas as pd
import re
import csv
import tempfile
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
    roc_curve,
    auc as sk_auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset as TorchDataset
import gc


# ============================================================================
# CONFIGURATION
# ============================================================================

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

MAX_TOKEN_LENGTH = 512


def get_system_info():
    """Build an HTML string describing the user's compute environment."""
    import platform
    lines = []

    # Device
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        lines.append(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        lines.append("FP16 training: supported")
    elif device.type == 'mps':
        lines.append("GPU: Apple Silicon (MPS)")
        lines.append("FP16 training: not supported on MPS")
    else:
        lines.append("GPU: None detected (using CPU)")
        lines.append("FP16 training: not supported on CPU")

    # CPU / RAM
    import os
    cpu_count = os.cpu_count() or 1
    lines.append(f"CPU cores: {cpu_count}")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        lines.append(f"RAM: {ram_gb:.1f} GB")
    except ImportError:
        pass

    lines.append(f"Platform: {platform.system()} {platform.machine()}")
    lines.append(f"PyTorch: {torch.__version__}")

    return " · ".join(lines)

FINETUNE_MODELS = {
    "ConfliBERT (recommended for conflict/political text)": "snowood1/ConfliBERT-scr-uncased",
    "BERT Base Uncased": "bert-base-uncased",
    "BERT Base Cased": "bert-base-cased",
    "RoBERTa Base": "roberta-base",
    "ModernBERT Base": "answerdotai/ModernBERT-base",
    "DeBERTa v3 Base": "microsoft/deberta-v3-base",
    "DistilBERT Base Uncased": "distilbert-base-uncased",
}

NER_LABELS = {
    'Organisation': '#3b82f6',
    'Person': '#ef4444',
    'Location': '#10b981',
    'Quantity': '#ff6b35',
    'Weapon': '#8b5cf6',
    'Nationality': '#06b6d4',
    'Temporal': '#ec4899',
    'DocumentReference': '#92400e',
    'MilitaryPlatform': '#f59e0b',
    'Money': '#f472b6',
}

CLASS_NAMES = ['Negative', 'Positive']
MULTI_CLASS_NAMES = ["Armed Assault", "Bombing or Explosion", "Kidnapping", "Other"]


# ============================================================================
# PRETRAINED MODEL LOADING
# ============================================================================

qa_model_name = 'salsarra/ConfliBERT-QA'
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

ner_model_name = 'eventdata-utd/conflibert-named-entity-recognition'
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

clf_model_name = 'eventdata-utd/conflibert-binary-classification'
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_name).to(device)
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)

multi_clf_model_name = 'eventdata-utd/conflibert-satp-relevant-multilabel'
multi_clf_model = AutoModelForSequenceClassification.from_pretrained(multi_clf_model_name).to(device)
multi_clf_tokenizer = AutoTokenizer.from_pretrained(multi_clf_model_name)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_path(f):
    """Get file path from Gradio file component output."""
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, 'name', str(f))


def truncate_text(text, tokenizer, max_length=MAX_TOKEN_LENGTH):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length - 1] + [tokenizer.sep_token_id]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def info_callout(text):
    """Wrap markdown text in a styled callout div to avoid Gradio double-border."""
    return (
        "<div class='info-callout-inner' style='"
        "background: #fff7f3; border-left: 3px solid #ff6b35; "
        "padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; "
        "font-size: 0.9rem;'>\n\n"
        f"{text}\n\n</div>"
    )


def handle_error(e, default_limit=512):
    msg = str(e)
    match = re.search(
        r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", msg
    )
    if match:
        return (
            f"<span style='color: #ef4444; font-weight: 600;'>"
            f"Error: Input ({match.group(1)} tokens) exceeds model limit ({match.group(2)})</span>"
        )
    match_qa = re.search(r"indices\[0,(\d+)\] = \d+ is not in \[0, (\d+)\)", msg)
    if match_qa:
        return (
            f"<span style='color: #ef4444; font-weight: 600;'>"
            f"Error: Input too long for model (limit: {match_qa.group(2)} tokens)</span>"
        )
    return f"<span style='color: #ef4444; font-weight: 600;'>Error: {msg}</span>"


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def question_answering(context, question):
    if not context or not question:
        return "Please provide both context and question."
    try:
        inputs = qa_tokenizer(question, context, return_tensors='tf', truncation=True)
        outputs = qa_model(inputs)
        start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
        end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1
        tokens = qa_tokenizer.convert_ids_to_tokens(
            inputs['input_ids'].numpy()[0][start:end]
        )
        answer = qa_tokenizer.convert_tokens_to_string(tokens)
        return f"<span style='color: #10b981; font-weight: 600;'>{answer}</span>"
    except Exception as e:
        return handle_error(e)


def named_entity_recognition(text, output_format='html'):
    if not text:
        return "Please provide text for analysis."
    try:
        inputs = ner_tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = ner_model(**inputs)
        results = outputs.logits.argmax(dim=2).squeeze().tolist()
        tokens = ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        tokens = [t.replace('[UNK]', "'") for t in tokens]

        entities = []
        seen_labels = set()
        current_entity = []
        current_label = None

        for i in range(len(tokens)):
            token = tokens[i]
            label = ner_model.config.id2label[results[i]].split('-')[-1]

            if token.startswith('##'):
                if entities:
                    if output_format == 'html':
                        entities[-1][0] += token[2:]
                    elif current_entity:
                        current_entity[-1] = current_entity[-1] + token[2:]
            else:
                if output_format == 'csv':
                    if label != 'O':
                        if label == current_label:
                            current_entity.append(token)
                        else:
                            if current_entity:
                                entities.append([' '.join(current_entity), current_label])
                            current_entity = [token]
                            current_label = label
                    else:
                        if current_entity:
                            entities.append([' '.join(current_entity), current_label])
                            current_entity = []
                            current_label = None
                else:
                    entities.append([token, label])

                if label != 'O':
                    seen_labels.add(label)

        if output_format == 'csv' and current_entity:
            entities.append([' '.join(current_entity), current_label])

        if output_format == 'csv':
            grouped = {}
            for token, label in entities:
                if label != 'O':
                    grouped.setdefault(label, []).append(token)
            parts = []
            for label, toks in grouped.items():
                unique = list(dict.fromkeys(toks))
                parts.append(f"{label}: {' | '.join(unique)}")
            return ' || '.join(parts)

        # HTML output
        highlighted = ""
        for token, label in entities:
            color = NER_LABELS.get(label, 'inherit')
            if label != 'O':
                highlighted += (
                    f"<span style='color: {color}; font-weight: 600;'>{token}</span> "
                )
            else:
                highlighted += f"{token} "

        if seen_labels:
            legend_items = ""
            for label in sorted(seen_labels):
                color = NER_LABELS.get(label, '#666')
                legend_items += (
                    f"<li style='color: {color}; font-weight: 600; "
                    f"background: {color}15; padding: 2px 8px; border-radius: 4px; "
                    f"font-size: 0.85rem;'>{label}</li>"
                )
            legend = (
                f"<div style='margin-top: 1rem; padding-top: 0.75rem; "
                f"border-top: 1px solid #e5e7eb;'>"
                f"<strong>Entities found:</strong>"
                f"<ul style='list-style: none; padding: 0; display: flex; "
                f"flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;'>"
                f"{legend_items}</ul></div>"
            )
            return f"<div style='line-height: 1.8;'>{highlighted}</div>{legend}"
        else:
            return (
                f"<div style='line-height: 1.8;'>{highlighted}</div>"
                f"<div style='color: #888; margin-top: 0.5rem;'>No entities detected.</div>"
            )

    except Exception as e:
        return handle_error(e)


def predict_with_model(text, model, tokenizer):
    """Run inference with an arbitrary classification model."""
    model.eval()
    dev = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted = torch.argmax(probs).item()
    num_classes = probs.shape[0] if probs.dim() > 0 else 1

    lines = []
    for i in range(num_classes):
        p = probs[i].item() * 100 if probs.dim() > 0 else probs.item() * 100
        if i == predicted:
            lines.append(
                f"<span style='color: #10b981; font-weight: 600;'>"
                f"Class {i}: {p:.2f}% (predicted)</span>"
            )
        else:
            lines.append(f"<span style='color: #9ca3af;'>Class {i}: {p:.2f}%</span>")
    return "<br>".join(lines)


def text_classification(text, custom_model=None, custom_tokenizer=None):
    if not text:
        return "Please provide text for classification."
    try:
        # Use custom model if loaded
        if custom_model is not None and custom_tokenizer is not None:
            return predict_with_model(text, custom_model, custom_tokenizer)

        # Pretrained binary classifier
        inputs = clf_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = clf_model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item() * 100

        if predicted == 1:
            return (
                f"<span style='color: #10b981; font-weight: 600;'>"
                f"Positive -- Related to conflict, violence, or politics. "
                f"(Confidence: {confidence:.1f}%)</span>"
            )
        else:
            return (
                f"<span style='color: #ef4444; font-weight: 600;'>"
                f"Negative -- Not related to conflict, violence, or politics. "
                f"(Confidence: {confidence:.1f}%)</span>"
            )
    except Exception as e:
        return handle_error(e)


def multilabel_classification(text):
    if not text:
        return "Please provide text for classification."
    try:
        inputs = multi_clf_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = multi_clf_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        results = []
        for i in range(len(probs)):
            conf = probs[i] * 100
            if probs[i] >= 0.5:
                results.append(
                    f"<span style='color: #10b981; font-weight: 600;'>"
                    f"{MULTI_CLASS_NAMES[i]}: {conf:.1f}%</span>"
                )
            else:
                results.append(
                    f"<span style='color: #9ca3af;'>"
                    f"{MULTI_CLASS_NAMES[i]}: {conf:.1f}%</span>"
                )
        return "<br>".join(results)
    except Exception as e:
        return handle_error(e)


# ============================================================================
# CSV BATCH PROCESSING
# ============================================================================

def process_csv_ner(file):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    entities = []
    for text in df['text']:
        if pd.isna(text):
            entities.append("")
        else:
            entities.append(named_entity_recognition(str(text), output_format='csv'))
    df['entities'] = entities

    out = tempfile.NamedTemporaryFile(suffix='_ner_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


def process_csv_binary(file, custom_model=None, custom_tokenizer=None):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    results = []
    for text in df['text']:
        if pd.isna(text):
            results.append("")
        else:
            html = text_classification(str(text), custom_model, custom_tokenizer)
            results.append(re.sub(r'<[^>]+>', '', html).strip())
    df['classification_results'] = results

    out = tempfile.NamedTemporaryFile(suffix='_classification_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


def process_csv_multilabel(file):
    path = get_path(file)
    if path is None:
        return None
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    results = []
    for text in df['text']:
        if pd.isna(text):
            results.append("")
        else:
            html = multilabel_classification(str(text))
            results.append(re.sub(r'<[^>]+>', '', html).strip())
    df['multilabel_results'] = results

    out = tempfile.NamedTemporaryFile(suffix='_multilabel_results.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


# ============================================================================
# FINETUNING
# ============================================================================

class TextClassificationDataset(TorchDataset):
    """PyTorch Dataset for text classification with HuggingFace tokenizers."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors=None,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def parse_data_file(file_path):
    """Parse a TSV/CSV data file. Expected format: text<separator>label (no header).
    Labels must be integers. Returns (texts, labels, num_labels)."""
    path = get_path(file_path)
    texts, labels = [], []

    # Detect delimiter from first line
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"')
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(row[-1].strip())
                text = row[0].strip() if len(row) == 2 else delimiter.join(row[:-1]).strip()
                if text:
                    texts.append(text)
                    labels.append(label)
            except (ValueError, IndexError):
                continue  # skip header or malformed rows

    if not texts:
        raise ValueError(
            "No valid data rows found. Expected format: text<tab>label (no header row)"
        )

    num_labels = max(labels) + 1
    return texts, labels, num_labels


class LogCallback(TrainerCallback):
    """Captures training logs for display in the UI."""

    def __init__(self):
        self.entries = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.entries.append({**logs})

    def format(self):
        lines = []
        skip_keys = {
            'total_flos', 'train_runtime', 'train_samples_per_second',
            'train_steps_per_second', 'train_loss',
        }
        for entry in self.entries:
            parts = []
            for k, v in sorted(entry.items()):
                if k in skip_keys:
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                elif isinstance(v, (int, np.integer)):
                    parts.append(f"{k}: {v}")
            if parts:
                lines.append("  ".join(parts))
        return "\n".join(lines)


def make_compute_metrics(task_type):
    """Factory for compute_metrics function based on task type."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = sk_accuracy(labels, preds)

        if task_type == "Binary":
            return {
                'accuracy': acc,
                'precision': sk_precision(labels, preds, zero_division=0),
                'recall': sk_recall(labels, preds, zero_division=0),
                'f1': sk_f1(labels, preds, zero_division=0),
            }
        else:
            return {
                'accuracy': acc,
                'f1_macro': sk_f1(labels, preds, average='macro', zero_division=0),
                'f1_micro': sk_f1(labels, preds, average='micro', zero_division=0),
                'precision_macro': sk_precision(
                    labels, preds, average='macro', zero_division=0
                ),
                'precision_micro': sk_precision(
                    labels, preds, average='micro', zero_division=0
                ),
                'recall_macro': sk_recall(
                    labels, preds, average='macro', zero_division=0
                ),
                'recall_micro': sk_recall(
                    labels, preds, average='micro', zero_division=0
                ),
            }

    return compute_metrics


def run_finetuning(
    train_file, dev_file, test_file, task_type, model_display_name,
    epochs, batch_size, lr, weight_decay, warmup_ratio, max_seq_len,
    grad_accum, fp16, patience, scheduler,
    progress=gr.Progress(track_tqdm=True),
):
    """Main finetuning function. Returns logs, metrics, model state, and visibility updates."""
    try:
        # Validate inputs
        if train_file is None or dev_file is None or test_file is None:
            raise ValueError("Please upload all three data files (train, dev, test).")

        epochs = int(epochs)
        batch_size = int(batch_size)
        max_seq_len = int(max_seq_len)
        grad_accum = int(grad_accum)
        patience = int(patience)

        # Parse data files
        train_texts, train_labels, n_train = parse_data_file(train_file)
        dev_texts, dev_labels, n_dev = parse_data_file(dev_file)
        test_texts, test_labels, n_test = parse_data_file(test_file)

        num_labels = max(n_train, n_dev, n_test)
        if task_type == "Binary" and num_labels > 2:
            raise ValueError(
                f"Binary task selected but found {num_labels} label classes in data. "
                f"Use Multiclass instead."
            )
        if task_type == "Binary":
            num_labels = 2

        # Load model and tokenizer
        model_id = FINETUNE_MODELS[model_display_name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )

        # Create datasets
        train_ds = TextClassificationDataset(
            train_texts, train_labels, tokenizer, max_seq_len
        )
        dev_ds = TextClassificationDataset(
            dev_texts, dev_labels, tokenizer, max_seq_len
        )
        test_ds = TextClassificationDataset(
            test_texts, test_labels, tokenizer, max_seq_len
        )

        # Output directory
        output_dir = tempfile.mkdtemp(prefix='conflibert_ft_')

        # Training arguments
        best_metric = 'f1' if task_type == 'Binary' else 'f1_macro'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            gradient_accumulation_steps=grad_accum,
            fp16=fp16 and torch.cuda.is_available(),
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=best_metric,
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=2,
            lr_scheduler_type=scheduler,
            report_to='none',
            seed=42,
        )

        # Callbacks
        log_callback = LogCallback()
        callbacks = [log_callback]
        if patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=make_compute_metrics(task_type),
            callbacks=callbacks,
        )

        # Train
        train_result = trainer.train()

        # Evaluate on test set
        test_results = trainer.evaluate(test_ds, metric_key_prefix='test')

        # Build log text
        header = (
            f"=== Configuration ===\n"
            f"Model: {model_display_name}\n"
            f"       {model_id}\n"
            f"Task:  {task_type} Classification ({num_labels} classes)\n"
            f"Data:  {len(train_texts)} train / {len(dev_texts)} dev / {len(test_texts)} test\n"
            f"Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  Scheduler: {scheduler}\n"
            f"\n=== Training Log ===\n"
        )
        runtime = train_result.metrics.get('train_runtime', 0)
        footer = (
            f"\n=== Training Complete ===\n"
            f"Time: {runtime:.1f}s ({runtime / 60:.1f} min)\n"
        )
        log_text = header + log_callback.format() + footer

        # Build metrics DataFrame
        metrics_data = []
        for k, v in sorted(test_results.items()):
            if isinstance(v, (int, float, np.floating, np.integer)) and k != 'test_epoch':
                name = k.replace('test_', '').replace('_', ' ').title()
                metrics_data.append([name, f"{float(v):.4f}"])
        metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Score'])

        # Move trained model to CPU for inference
        trained_model = trainer.model.cpu()
        trained_model.eval()

        return (
            log_text, metrics_df, trained_model, tokenizer, num_labels,
            gr.Column(visible=True), gr.Column(visible=True),
        )

    except Exception as e:
        error_log = f"Training failed:\n{str(e)}"
        empty_df = pd.DataFrame(columns=['Metric', 'Score'])
        return (
            error_log, empty_df, None, None, None,
            gr.Column(visible=False), gr.Column(visible=False),
        )


# ============================================================================
# MODEL MANAGEMENT (predict, save, load)
# ============================================================================

def predict_finetuned(text, model_state, tokenizer_state, num_labels_state):
    """Run prediction with the finetuned model stored in gr.State."""
    if not text:
        return "Please enter some text."
    if model_state is None:
        return "No model available. Please train a model first."
    return predict_with_model(text, model_state, tokenizer_state)


def save_finetuned_model(save_path, model_state, tokenizer_state):
    """Save the finetuned model and tokenizer to disk."""
    if model_state is None:
        return "No model to save. Please train a model first."
    if not save_path:
        return "Please specify a save directory."
    try:
        os.makedirs(save_path, exist_ok=True)
        model_state.save_pretrained(save_path)
        tokenizer_state.save_pretrained(save_path)
        return f"Model saved successfully to: {save_path}"
    except Exception as e:
        return f"Error saving model: {str(e)}"


def load_custom_model(path):
    """Load a finetuned classification model from disk."""
    if not path or not os.path.isdir(path):
        return None, None, "Invalid path. Please enter a valid model directory."
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        n = model.config.num_labels
        return model, tokenizer, f"Loaded model with {n} classes from: {path}"
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def reset_custom_model():
    """Reset to the pretrained ConfliBERT binary classifier."""
    return None, None, "Reset to pretrained ConfliBERT binary classifier."


def batch_predict_finetuned(file, model_state, tokenizer_state, num_labels_state):
    """Run batch predictions on a CSV using the finetuned model."""
    if model_state is None:
        return None
    path = get_path(file)
    if path is None:
        return None

    df = pd.read_csv(path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    model_state.eval()
    dev = next(model_state.parameters()).device

    predictions, confidences = [], []
    for text in df['text']:
        if pd.isna(text):
            predictions.append("")
            confidences.append("")
            continue

        inputs = tokenizer_state(
            str(text), return_tensors='pt', truncation=True,
            padding=True, max_length=512,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_state(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        conf = probs[pred].item() * 100
        predictions.append(str(pred))
        confidences.append(f"{conf:.1f}%")

    df['predicted_class'] = predictions
    df['confidence'] = confidences

    out = tempfile.NamedTemporaryFile(suffix='_predictions.csv', delete=False)
    df.to_csv(out.name, index=False)
    return out.name


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def load_example_binary():
    """Load the binary classification example dataset."""
    return (
        os.path.join(EXAMPLES_DIR, "binary", "train.tsv"),
        os.path.join(EXAMPLES_DIR, "binary", "dev.tsv"),
        os.path.join(EXAMPLES_DIR, "binary", "test.tsv"),
        "Binary",
    )


def load_example_multiclass():
    """Load the multiclass classification example dataset."""
    return (
        os.path.join(EXAMPLES_DIR, "multiclass", "train.tsv"),
        os.path.join(EXAMPLES_DIR, "multiclass", "dev.tsv"),
        os.path.join(EXAMPLES_DIR, "multiclass", "test.tsv"),
        "Multiclass",
    )


def run_comparison(
    train_file, dev_file, test_file, task_type, selected_models,
    epochs, batch_size, lr,
    progress=gr.Progress(track_tqdm=True),
):
    """Train multiple models on the same data and compare performance + ROC curves."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    empty = ("", None, None, None, gr.Column(visible=False))
    try:
        if not selected_models or len(selected_models) < 2:
            return ("Select at least 2 models to compare.",) + empty[1:]
        if train_file is None or dev_file is None or test_file is None:
            return ("Upload all 3 data files first.",) + empty[1:]

        epochs = int(epochs)
        batch_size = int(batch_size)

        train_texts, train_labels, n_train = parse_data_file(train_file)
        dev_texts, dev_labels, n_dev = parse_data_file(dev_file)
        test_texts, test_labels, n_test = parse_data_file(test_file)
        num_labels = max(n_train, n_dev, n_test)
        if task_type == "Binary":
            num_labels = 2

        # Only keep these metrics for the table and bar chart
        if task_type == "Binary":
            keep_metrics = {'Accuracy', 'Precision', 'Recall', 'F1'}
        else:
            keep_metrics = {
                'Accuracy', 'F1 Macro', 'F1 Micro',
                'Precision Macro', 'Recall Macro',
            }

        results = []
        roc_data = {}  # model_name -> (true_labels, probabilities)
        log_lines = []

        for i, model_display_name in enumerate(selected_models):
            model_id = FINETUNE_MODELS[model_display_name]
            short_name = model_display_name.split(" (")[0]
            log_lines.append(f"[{i + 1}/{len(selected_models)}] Training {short_name}...")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, num_labels=num_labels,
                )
                train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer, 512)
                dev_ds = TextClassificationDataset(dev_texts, dev_labels, tokenizer, 512)
                test_ds = TextClassificationDataset(test_texts, test_labels, tokenizer, 512)

                output_dir = tempfile.mkdtemp(prefix='conflibert_cmp_')
                best_metric = 'f1' if task_type == 'Binary' else 'f1_macro'

                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size * 2,
                    learning_rate=lr,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    eval_strategy='epoch',
                    save_strategy='epoch',
                    load_best_model_at_end=True,
                    metric_for_best_model=best_metric,
                    greater_is_better=True,
                    logging_steps=50,
                    save_total_limit=1,
                    report_to='none',
                    seed=42,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=dev_ds,
                    compute_metrics=make_compute_metrics(task_type),
                )

                train_result = trainer.train()

                # Get predictions for ROC curves
                pred_output = trainer.predict(test_ds)
                logits = pred_output.predictions
                true_labels = pred_output.label_ids
                probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                roc_data[short_name] = (true_labels, probs)

                # Collect classification metrics only
                test_results = trainer.evaluate(test_ds, metric_key_prefix='test')
                row = {'Model': short_name}
                for k, v in sorted(test_results.items()):
                    if not isinstance(v, (int, float, np.floating, np.integer)):
                        continue
                    name = k.replace('test_', '').replace('_', ' ').title()
                    if name in keep_metrics:
                        row[name] = round(float(v), 4)
                results.append(row)

                runtime = train_result.metrics.get('train_runtime', 0)
                log_lines.append(f"    Done in {runtime:.1f}s")

                del model, trainer, tokenizer, train_ds, dev_ds, test_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                log_lines.append(f"    Failed: {str(e)}")

        log_lines.append(f"\nComparison complete. {len(results)} models evaluated.")
        log_text = "\n".join(log_lines)

        if not results:
            return log_text, None, None, None, gr.Column(visible=False)

        comparison_df = pd.DataFrame(results)

        # --- Bar chart: classification metrics only ---
        metric_cols = [c for c in comparison_df.columns if c in keep_metrics]
        colors = ['#ff6b35', '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b']
        fig_bar = go.Figure()
        for j, metric in enumerate(metric_cols):
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].apply(
                    lambda x: f'{x:.3f}' if isinstance(x, float) else ''
                ),
                textposition='auto',
                marker_color=colors[j % len(colors)],
            ))
        fig_bar.update_layout(
            barmode='group',
            yaxis_title='Score', yaxis_range=[0, 1.05],
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400, margin=dict(t=40, b=40),
        )

        # --- ROC curves ---
        model_colors = ['#ff6b35', '#3b82f6', '#10b981', '#8b5cf6',
                        '#f59e0b', '#ec4899', '#06b6d4']
        fig_roc = go.Figure()
        for j, (model_name, (labels, probs)) in enumerate(roc_data.items()):
            color = model_colors[j % len(model_colors)]
            if num_labels == 2:
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                roc_auc_val = sk_auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{model_name} (AUC = {roc_auc_val:.3f})',
                    line=dict(color=color, width=2),
                ))
            else:
                # Macro-average ROC for multiclass
                labels_bin = label_binarize(labels, classes=list(range(num_labels)))
                all_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.zeros_like(all_fpr)
                for c in range(num_labels):
                    fpr_c, tpr_c, _ = roc_curve(labels_bin[:, c], probs[:, c])
                    mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
                mean_tpr /= num_labels
                roc_auc_val = sk_auc(all_fpr, mean_tpr)
                fig_roc.add_trace(go.Scatter(
                    x=all_fpr, y=mean_tpr, mode='lines',
                    name=f'{model_name} (macro AUC = {roc_auc_val:.3f})',
                    line=dict(color=color, width=2),
                ))

        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='#ccc', width=1),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400, margin=dict(t=40, b=40),
        )

        return log_text, comparison_df, fig_bar, fig_roc, gr.Column(visible=True)

    except Exception as e:
        return f"Comparison failed: {str(e)}", None, None, None, gr.Column(visible=False)


# ============================================================================
# THEME & CSS
# ============================================================================

utd_orange = gr.themes.Color(
    c50="#fff7f3", c100="#ffead9", c200="#ffd4b3", c300="#ffb380",
    c400="#ff8c52", c500="#ff6b35", c600="#e8551f", c700="#c2410c",
    c800="#9a3412", c900="#7c2d12", c950="#431407",
)

theme = gr.themes.Soft(
    primary_hue=utd_orange,
    secondary_hue="neutral",
    font=gr.themes.GoogleFont("Inter"),
)

custom_css = """
/* Top accent bar */
.gradio-container::before {
    content: '';
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #ff6b35, #ff9f40, #ff6b35);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
}

/* Active tab styling */
.tab-nav button.selected {
    border-bottom-color: #ff6b35 !important;
    color: #ff6b35 !important;
    font-weight: 600 !important;
}

/* Log output - monospace */
.log-output textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* Dark mode: info callout adjustment */
.dark .info-callout-inner {
    background: rgba(255, 107, 53, 0.1) !important;
    color: #ffead9 !important;
}

/* Clean container width */
.gradio-container {
    max-width: 1200px !important;
}

/* Smooth transitions */
.gradio-container * {
    transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
"""


# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(theme=theme, css=custom_css, title="ConfliBERT") as demo:

    # ---- HEADER ----
    gr.Markdown(
        "<div style='text-align: center; padding: 1.5rem 0 0.5rem;'>"
        "<h1 style='font-size: 2.5rem; font-weight: 800; margin: 0;'>"
        "<a href='https://eventdata.utdallas.edu/conflibert/' target='_blank' "
        "style='color: #ff6b35; text-decoration: none;'>ConfliBERT</a></h1>"
        "<p style='color: #888; font-size: 0.95rem; margin: 0.25rem 0 0;'>"
        "A Pretrained Language Model for Conflict and Political Violence</p></div>"
    )

    with gr.Tabs():

        # ================================================================
        # HOME TAB
        # ================================================================
        with gr.Tab("Home"):
            gr.Markdown(
                "## Welcome to ConfliBERT\n\n"
                "ConfliBERT is a pretrained language model built specifically for "
                "conflict and political violence text. This application lets you "
                "run inference with ConfliBERT's pretrained models and fine-tune "
                "your own classifiers on custom data. Use the tabs above to get started."
            )

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown(
                        "### Inference\n\n"
                        "Run pretrained ConfliBERT models on your text. "
                        "Each task has its own tab with single-text analysis "
                        "and CSV batch processing.\n\n"
                        "**Named Entity Recognition**\n"
                        "Identify persons, organizations, locations, weapons, "
                        "and other entities in text. Results are color-coded "
                        "by entity type.\n\n"
                        "**Binary Classification**\n"
                        "Determine whether text is related to conflict, violence, "
                        "or politics (positive) or not (negative). You can also "
                        "load a custom fine-tuned model here.\n\n"
                        "**Multilabel Classification**\n"
                        "Score text against four event categories: Armed Assault, "
                        "Bombing/Explosion, Kidnapping, and Other. Each category "
                        "is scored independently.\n\n"
                        "**Question Answering**\n"
                        "Provide a context passage and ask a question. The model "
                        "extracts the most relevant answer span from the text."
                    )
                with gr.Column():
                    gr.Markdown(
                        "### Fine-tuning\n\n"
                        "Train your own binary or multiclass text classifier "
                        "on custom labeled data, all within the browser.\n\n"
                        "**Workflow:**\n"
                        "1. Upload your training, validation, and test data as "
                        "TSV files (or load a built-in example dataset)\n"
                        "2. Pick a base model: ConfliBERT, BERT, RoBERTa, "
                        "ModernBERT, DeBERTa, or DistilBERT\n"
                        "3. Configure training parameters (sensible defaults "
                        "are provided)\n"
                        "4. Train and watch progress in real time\n"
                        "5. Review test-set metrics (accuracy, precision, "
                        "recall, F1)\n"
                        "6. Try your model on new text immediately\n"
                        "7. Run batch predictions on a CSV\n"
                        "8. Save the model and load it later in the "
                        "Classification tab\n\n"
                        "**Advanced features:**\n"
                        "- Early stopping with configurable patience\n"
                        "- Learning rate schedulers (linear, cosine, constant)\n"
                        "- Mixed precision training (FP16 on CUDA GPUs)\n"
                        "- Gradient accumulation for larger effective batch sizes\n"
                        "- Weight decay regularization"
                    )

            gr.Markdown(
                f"---\n\n"
                f"**Your system:** {get_system_info()}"
            )

            gr.Markdown(
                "**Citation:** Brandt, P.T., Alsarra, S., D'Orazio, V., "
                "Heintze, D., Khan, L., Meher, S., Osorio, J. and Sianan, M., "
                "2025. Extractive versus Generative Language Models for Political "
                "Conflict Text Classification. *Political Analysis*, pp.1-29."
            )

        # ================================================================
        # NER TAB
        # ================================================================
        with gr.Tab("Named Entity Recognition"):
            gr.Markdown(info_callout(
                "Identify entities in text such as **persons**, **organizations**, "
                "**locations**, **weapons**, and more. Results are color-coded by type."
            ))
            with gr.Row(equal_height=True):
                with gr.Column():
                    ner_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to analyze for entities...",
                        label="Input Text",
                    )
                    ner_btn = gr.Button("Analyze Entities", variant="primary")
                with gr.Column():
                    ner_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown(
                    "Upload a CSV file with a `text` column to process "
                    "multiple texts at once."
                )
                with gr.Row():
                    ner_csv_in = gr.File(
                        label="Upload CSV", file_types=[".csv"],
                    )
                    ner_csv_out = gr.File(label="Download Results")
                ner_csv_btn = gr.Button("Process CSV", variant="secondary")

        # ================================================================
        # BINARY CLASSIFICATION TAB
        # ================================================================
        with gr.Tab("Binary Classification"):
            gr.Markdown(info_callout(
                "Classify text as **conflict-related** (positive) or "
                "**not conflict-related** (negative). Uses the pretrained ConfliBERT "
                "binary classifier by default, or load your own finetuned model below."
            ))

            custom_clf_model = gr.State(None)
            custom_clf_tokenizer = gr.State(None)

            with gr.Row(equal_height=True):
                with gr.Column():
                    clf_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to classify...",
                        label="Input Text",
                    )
                    clf_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    clf_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown("Upload a CSV file with a `text` column.")
                with gr.Row():
                    clf_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    clf_csv_out = gr.File(label="Download Results")
                clf_csv_btn = gr.Button("Process CSV", variant="secondary")

            with gr.Accordion("Load Custom Model", open=False):
                gr.Markdown(
                    "Load a finetuned classification model from a local directory "
                    "to use instead of the default pretrained classifier."
                )
                clf_model_path = gr.Textbox(
                    label="Model directory path",
                    placeholder="e.g., ./finetuned_model",
                )
                with gr.Row():
                    clf_load_btn = gr.Button("Load Model", variant="secondary")
                    clf_reset_btn = gr.Button(
                        "Reset to Pretrained", variant="secondary",
                    )
                clf_status = gr.Markdown("")

        # ================================================================
        # MULTILABEL CLASSIFICATION TAB
        # ================================================================
        with gr.Tab("Multilabel Classification"):
            gr.Markdown(info_callout(
                "Identify multiple event types in text. Each category is scored "
                "independently: **Armed Assault**, **Bombing/Explosion**, "
                "**Kidnapping**, **Other**. Categories above 50% confidence "
                "are highlighted."
            ))
            with gr.Row(equal_height=True):
                with gr.Column():
                    multi_input = gr.Textbox(
                        lines=6,
                        placeholder="Paste or type text to classify...",
                        label="Input Text",
                    )
                    multi_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    multi_output = gr.HTML(label="Results")

            with gr.Accordion("Batch Processing (CSV)", open=False):
                gr.Markdown("Upload a CSV file with a `text` column.")
                with gr.Row():
                    multi_csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    multi_csv_out = gr.File(label="Download Results")
                multi_csv_btn = gr.Button("Process CSV", variant="secondary")

        # ================================================================
        # QUESTION ANSWERING TAB
        # ================================================================
        with gr.Tab("Question Answering"):
            gr.Markdown(info_callout(
                "Extract answers from a context passage. Provide a paragraph of "
                "text and ask a question about it. The model will highlight the "
                "most relevant span."
            ))
            with gr.Row(equal_height=True):
                with gr.Column():
                    qa_context = gr.Textbox(
                        lines=6,
                        placeholder="Paste the context passage here...",
                        label="Context",
                    )
                    qa_question = gr.Textbox(
                        lines=2,
                        placeholder="What would you like to know?",
                        label="Question",
                    )
                    qa_btn = gr.Button("Get Answer", variant="primary")
                with gr.Column():
                    qa_output = gr.HTML(label="Answer")

        # ================================================================
        # FINE-TUNE TAB
        # ================================================================
        with gr.Tab("Fine-tune"):
            gr.Markdown(info_callout(
                "Fine-tune a binary or multiclass classifier on your own data. "
                "Upload labeled TSV files, pick a base model, and train. "
                "Or compare multiple models head-to-head on the same dataset."
            ))

            # -- Data --
            gr.Markdown("### Data")
            gr.Markdown(
                "TSV files, no header, format: `text[TAB]label` "
                "(binary: 0/1, multiclass: 0, 1, 2, ...)"
            )
            with gr.Row():
                ft_ex_binary_btn = gr.Button(
                    "Load Example: Binary", variant="secondary", size="sm",
                )
                ft_ex_multi_btn = gr.Button(
                    "Load Example: Multiclass (4 classes)", variant="secondary", size="sm",
                )
            with gr.Row():
                ft_train_file = gr.File(
                    label="Train", file_types=[".tsv", ".csv", ".txt"],
                )
                ft_dev_file = gr.File(
                    label="Validation", file_types=[".tsv", ".csv", ".txt"],
                )
                ft_test_file = gr.File(
                    label="Test", file_types=[".tsv", ".csv", ".txt"],
                )

            # -- Configuration --
            gr.Markdown("### Configuration")
            with gr.Row():
                ft_task = gr.Radio(
                    ["Binary", "Multiclass"],
                    label="Task Type", value="Binary",
                )
                ft_model = gr.Dropdown(
                    choices=list(FINETUNE_MODELS.keys()),
                    label="Base Model",
                    value=list(FINETUNE_MODELS.keys())[0],
                )
            with gr.Row():
                ft_epochs = gr.Number(
                    label="Epochs", value=3, minimum=1, maximum=100, precision=0,
                )
                ft_batch = gr.Number(
                    label="Batch Size", value=8, minimum=1, maximum=128, precision=0,
                )
                ft_lr = gr.Number(
                    label="Learning Rate", value=2e-5, minimum=1e-7, maximum=1e-2,
                )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    ft_weight_decay = gr.Number(
                        label="Weight Decay", value=0.01, minimum=0, maximum=1,
                    )
                    ft_warmup = gr.Number(
                        label="Warmup Ratio", value=0.1, minimum=0, maximum=0.5,
                    )
                    ft_max_len = gr.Number(
                        label="Max Sequence Length", value=512,
                        minimum=32, maximum=8192, precision=0,
                    )
                with gr.Row():
                    ft_grad_accum = gr.Number(
                        label="Gradient Accumulation", value=1,
                        minimum=1, maximum=64, precision=0,
                    )
                    ft_fp16 = gr.Checkbox(
                        label="Mixed Precision (FP16)", value=False,
                    )
                    ft_patience = gr.Number(
                        label="Early Stopping Patience", value=3,
                        minimum=0, maximum=20, precision=0,
                    )
                ft_scheduler = gr.Dropdown(
                    ["linear", "cosine", "constant", "constant_with_warmup"],
                    label="LR Scheduler", value="linear",
                )

            # -- Train --
            ft_train_btn = gr.Button(
                "Start Training", variant="primary", size="lg",
            )

            # State for the trained model
            ft_model_state = gr.State(None)
            ft_tokenizer_state = gr.State(None)
            ft_num_labels_state = gr.State(None)

            with gr.Accordion("Training Log", open=False) as ft_log_accordion:
                ft_log = gr.Textbox(
                    lines=12, interactive=False, elem_classes="log-output",
                    show_label=False,
                )

            # -- Results + Try Model (hidden until training completes) --
            with gr.Column(visible=False) as ft_results_col:
                gr.Markdown("### Results")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        ft_metrics = gr.Dataframe(
                            label="Test Set Metrics",
                            headers=["Metric", "Score"],
                            interactive=False,
                        )
                    with gr.Column(scale=3):
                        gr.Markdown("**Try your model**")
                        ft_try_input = gr.Textbox(
                            lines=2, label="Input Text",
                            placeholder="Type text to classify...",
                        )
                        with gr.Row():
                            ft_try_btn = gr.Button("Predict", variant="primary")
                        ft_try_output = gr.HTML(label="Prediction")

            # -- Save + Batch (hidden until training completes) --
            with gr.Column(visible=False) as ft_actions_col:
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("**Save model**")
                        ft_save_path = gr.Textbox(
                            label="Save Directory", value="./finetuned_model",
                        )
                        ft_save_btn = gr.Button("Save", variant="secondary")
                        ft_save_status = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("**Batch predictions**")
                        ft_batch_in = gr.File(
                            label="Upload CSV (needs 'text' column)",
                            file_types=[".csv"],
                        )
                        ft_batch_btn = gr.Button(
                            "Run Predictions", variant="secondary",
                        )
                        ft_batch_out = gr.File(label="Download Results")

            # -- Compare Models --
            gr.Markdown("---")
            with gr.Accordion("Compare Multiple Models", open=False):
                gr.Markdown(
                    "Train the same dataset on different base models and compare "
                    "performance side by side. Uses the data and task type above."
                )
                cmp_models = gr.CheckboxGroup(
                    choices=list(FINETUNE_MODELS.keys()),
                    label="Select models to compare (pick 2 or more)",
                )
                with gr.Row():
                    cmp_epochs = gr.Number(label="Epochs", value=3, minimum=1, precision=0)
                    cmp_batch = gr.Number(label="Batch Size", value=8, minimum=1, precision=0)
                    cmp_lr = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-7)
                cmp_btn = gr.Button("Compare Models", variant="primary")
                cmp_log = gr.Textbox(
                    label="Comparison Log", lines=8,
                    interactive=False, elem_classes="log-output",
                )
                with gr.Column(visible=False) as cmp_results_col:
                    cmp_table = gr.Dataframe(
                        label="Comparison Results", interactive=False,
                    )
                    cmp_plot = gr.Plot(label="Metrics Comparison")
                    cmp_roc = gr.Plot(label="ROC Curves")

    # ---- FOOTER ----
    gr.Markdown(
        "<div style='text-align: center; padding: 1rem 0; margin-top: 0.5rem; "
        "border-top: 1px solid #e5e7eb;'>"
        "<p style='color: #888; font-size: 0.85rem; margin: 0;'>"
        "Developed by "
        "<a href='http://shreyasmeher.com' target='_blank' "
        "style='color: #ff6b35; text-decoration: none;'>Shreyas Meher</a>"
        "</p>"
        "<p style='color: #999; font-size: 0.75rem; margin: 0.5rem 0 0; "
        "max-width: 700px; margin-left: auto; margin-right: auto; line-height: 1.4;'>"
        "If you use ConfliBERT in your research, please cite:<br>"
        "<em>Brandt, P.T., Alsarra, S., D'Orazio, V., Heintze, D., Khan, L., "
        "Meher, S., Osorio, J. and Sianan, M., 2025. Extractive versus Generative "
        "Language Models for Political Conflict Text Classification. "
        "Political Analysis, pp.1&ndash;29.</em>"
        "</p></div>"
    )

    # ====================================================================
    # EVENT HANDLERS
    # ====================================================================

    # NER
    ner_btn.click(
        fn=named_entity_recognition, inputs=[ner_input], outputs=[ner_output],
    )
    ner_csv_btn.click(
        fn=process_csv_ner, inputs=[ner_csv_in], outputs=[ner_csv_out],
    )

    # Binary Classification
    clf_btn.click(
        fn=text_classification,
        inputs=[clf_input, custom_clf_model, custom_clf_tokenizer],
        outputs=[clf_output],
    )
    clf_csv_btn.click(
        fn=process_csv_binary,
        inputs=[clf_csv_in, custom_clf_model, custom_clf_tokenizer],
        outputs=[clf_csv_out],
    )
    clf_load_btn.click(
        fn=load_custom_model,
        inputs=[clf_model_path],
        outputs=[custom_clf_model, custom_clf_tokenizer, clf_status],
    )
    clf_reset_btn.click(
        fn=reset_custom_model,
        outputs=[custom_clf_model, custom_clf_tokenizer, clf_status],
    )

    # Multilabel Classification
    multi_btn.click(
        fn=multilabel_classification, inputs=[multi_input], outputs=[multi_output],
    )
    multi_csv_btn.click(
        fn=process_csv_multilabel, inputs=[multi_csv_in], outputs=[multi_csv_out],
    )

    # Question Answering
    qa_btn.click(
        fn=question_answering,
        inputs=[qa_context, qa_question],
        outputs=[qa_output],
    )

    # Fine-tuning: example dataset loaders
    ft_ex_binary_btn.click(
        fn=load_example_binary,
        outputs=[ft_train_file, ft_dev_file, ft_test_file, ft_task],
    )
    ft_ex_multi_btn.click(
        fn=load_example_multiclass,
        outputs=[ft_train_file, ft_dev_file, ft_test_file, ft_task],
    )

    # Fine-tuning: training
    ft_train_btn.click(
        fn=run_finetuning,
        inputs=[
            ft_train_file, ft_dev_file, ft_test_file,
            ft_task, ft_model,
            ft_epochs, ft_batch, ft_lr,
            ft_weight_decay, ft_warmup, ft_max_len,
            ft_grad_accum, ft_fp16, ft_patience, ft_scheduler,
        ],
        outputs=[
            ft_log, ft_metrics,
            ft_model_state, ft_tokenizer_state, ft_num_labels_state,
            ft_results_col, ft_actions_col,
        ],
        concurrency_limit=1,
    )

    # Try finetuned model
    ft_try_btn.click(
        fn=predict_finetuned,
        inputs=[ft_try_input, ft_model_state, ft_tokenizer_state, ft_num_labels_state],
        outputs=[ft_try_output],
    )

    # Save finetuned model
    ft_save_btn.click(
        fn=save_finetuned_model,
        inputs=[ft_save_path, ft_model_state, ft_tokenizer_state],
        outputs=[ft_save_status],
    )

    # Batch predictions with finetuned model
    ft_batch_btn.click(
        fn=batch_predict_finetuned,
        inputs=[ft_batch_in, ft_model_state, ft_tokenizer_state, ft_num_labels_state],
        outputs=[ft_batch_out],
    )

    # Model comparison
    cmp_btn.click(
        fn=run_comparison,
        inputs=[
            ft_train_file, ft_dev_file, ft_test_file,
            ft_task, cmp_models, cmp_epochs, cmp_batch, cmp_lr,
        ],
        outputs=[cmp_log, cmp_table, cmp_plot, cmp_roc, cmp_results_col],
        concurrency_limit=1,
    )


# ============================================================================
# LAUNCH
# ============================================================================

demo.launch(share=True)
