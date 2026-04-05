import argparse
import json
import os
import random
import gc

from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
import spacy

from transformers.trainer_utils import EvalPrediction
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)

parser = argparse.ArgumentParser()
parser.add_argument("--eval-only", action="store_true",
                    help="Skip training, load model from ./money_detector_final")
args = parser.parse_args()

torch.cuda.empty_cache()

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


def transform_tags(row: dict) -> dict:
    """ Convert every other tag to 0, only keep B-MONEY (16) and I-MONEY (17)"""

    row['tags'] = [1 if t == 16 else 2 if t == 17 else 0 for t in row['tags']]

    return row

# Link to the dataset: https://huggingface.co/datasets/tner/ontonotes5
dataset = load_dataset("tner/ontonotes5", revision="refs/convert/parquet")
dataset = dataset.map(transform_tags)


# Label scheme: 0=O, 1=B-MONEY, 2=I-MONEY, -100=ignore (special tokens / padding)
LABEL_O = 0
LABEL_B = 1
LABEL_I = 2
IGNORE = -100

# The dataset uses PTB convention, let's handle that and convert dataset to actual sentences
PTB_REPLACEMENTS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"',
}

def tokens_to_text_and_spans(row: dict) -> tuple[str, list[tuple[int, int, int]]]:
    """Reconstructs raw text from PTB-style tokens and extracts MONEY entity spans."""

    tokens, tags = row["tokens"], row["tags"]
    clean_tokens = [PTB_REPLACEMENTS.get(t, t) for t in tokens]

    # Build text by joining with spaces, tracking each token's char offset
    char_offsets = []  # (start, end) per token
    pos = 0
    parts = []
    for i, tok in enumerate(clean_tokens):
        char_offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok) + 1  # +1 for space
    text = " ".join(parts)

    spans = []
    i = 0
    while i < len(tags):
        if tags[i] == LABEL_B:
            start = char_offsets[i][0]
            end = char_offsets[i][1]
            j = i + 1
            while j < len(tags) and tags[j] == LABEL_I:
                end = char_offsets[j][1]
                j += 1
            spans.append((start, end, LABEL_B))
            i = j
        else:
            i += 1

    return {"text": text, "spans": spans, "source": "ontonotes"}

onto_dataset = dataset.map(tokens_to_text_and_spans)

with open("data/synthetic_dataset.json") as f:
    synth_data = json.load(f)

# 1. Separate & Downsample Synthetic Data
synth_pos = [r for r in synth_data if r["spans"]]
synth_neg = [r for r in synth_data if not r["spans"]]

random.seed(42)
synth_pos = random.sample(synth_pos, min(800, len(synth_pos)))

pos_train, pos_val = train_test_split(synth_pos, test_size=0.15, random_state=42)
neg_train, neg_val = train_test_split(synth_neg, test_size=0.15, random_state=42)

# 2. Process OntoNotes & Target 10% Positive Rate
onto_pos = onto_dataset["train"].filter(lambda r: len(r["spans"]) > 0)
onto_neg_all = onto_dataset["train"].filter(lambda r: len(r["spans"]) == 0).shuffle(seed=42)

n_pos_train = len(onto_pos) + len(pos_train)
n_neg_target = (n_pos_train * 6) - len(neg_train)
onto_neg = onto_neg_all.select(range(min(n_neg_target, len(onto_neg_all))))

# 3. Build Final Splits
train_set = concatenate_datasets([onto_pos, onto_neg, Dataset.from_list(pos_train + neg_train)]).shuffle(seed=42)
val_set = concatenate_datasets([onto_dataset["validation"], Dataset.from_list(pos_val + neg_val)]).shuffle(seed=42)
test_set = onto_dataset["test"]

print(f"Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")


"""## Training"""

for name in list(globals()):
    obj = globals()[name]
    if hasattr(obj, 'parameters') or hasattr(obj, 'generate'):  # models
        del globals()[name]

for name in ['trainer', 'model', 'llm', 'predictions', 'labels',
             'outputs', 'enrich_outputs', 'neg_outputs']:
    if name in globals():
        exec(f'del {name}')

gc.collect()
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

model_name = "microsoft/deberta-v3-base"

# Label mapping for the task
id2label = {0: "O", 1: "B-MONEY", 2: "I-MONEY"}
label2id = {"O": 0, "B-MONEY": 1, "I-MONEY": 2}

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_dataset(row: dict) -> dict:
  """
  Tokenize text with model's tokenizer and assing BIO labels to each subword
  based on character-level spans.
  """

  text, spans = row["text"], row["spans"]
  encoding = tokenizer(text, max_length=256, truncation=True,
                       padding=False, return_offsets_mapping=True,
                       return_tensors=None)
  offset_mapping = encoding["offset_mapping"]

  labels = []

  for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == 0 and tok_end == 0:
            labels.append(IGNORE)
            continue

        label = LABEL_O
        for span_start, span_end, _ in spans:
            if tok_start >= span_start and tok_end <= span_end:
                if tok_start == span_start:
                    label = LABEL_B
                else:
                    label = LABEL_I
                break

        labels.append(label)

  encoding["labels"] = labels
  del encoding["offset_mapping"]

  return encoding

columns_to_remove = ["text", "spans", "tokens", "tags"]

def tokenize_and_clean(dataset, desc="Tokenizing"):
    existing = [c for c in columns_to_remove if c in dataset.column_names]
    return dataset.map(tokenize_dataset, remove_columns=existing, desc=desc)

train_tokenized = tokenize_and_clean(train_set, "Tokenizing train")
valid_tokenized = tokenize_and_clean(val_set, "Tokenizing valid")
test_tokenized = tokenize_and_clean(test_set, "Tokenizing test")

train_sources = train_set["source"]
valid_sources = val_set["source"]
test_sources = test_set["source"]

train_tokenized = train_tokenized.remove_columns(["source"])
valid_tokenized = valid_tokenized.remove_columns(["source"])
test_tokenized = test_tokenized.remove_columns(["source"])

def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reduce logits to predictions on GPU before they accumulate in memory."""

    return logits.argmax(dim=-1)

def compute_metrics(evaluation_prediction: EvalPrediction) -> dict:
    """Computes entity-level metrics for MONEY detection."""

    predictions, labels = evaluation_prediction

    true_labels = []
    predicted_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_sent = []
        pred_sent = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == IGNORE:
                continue
            true_sent.append(id2label[label_id])
            pred_sent.append(id2label[pred_id])
        true_labels.append(true_sent)
        predicted_labels.append(pred_sent)

    report = classification_report(true_labels, predicted_labels, output_dict=True)
    results = report.get("MONEY", {})

    return {
        "f1": results.get("f1-score", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
    }

if args.eval_only:
    model = AutoModelForTokenClassification.from_pretrained("./money_detector_final")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Loaded trained model from ./money_detector_final")
else:
    config = AutoConfig.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)

    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    # DeBERTa-v3 ships some weights as fp16; upcast to fp32 so AMP's GradScaler works
    for param in model.parameters():
        if param.data.dtype == torch.float16:
            param.data = param.data.float()

    training_args = TrainingArguments(
        output_dir="./checkpoints",

        learning_rate=2e-5,

        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        warmup_steps=550,
        weight_decay=0.02,

        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        max_grad_norm=1.0,

        fp16=True,
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=10,

        seed=42,
        data_seed=42,

        report_to="none",
        run_name="deberta-v3-money-detector"
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding="longest",
        label_pad_token_id=IGNORE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.005,
            ),
        ],
    )

    torch.cuda.empty_cache()
    train_result = trainer.train()
    print("Training Metrics:")
    print(pd.Series(train_result.metrics))

    eval_metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    print(pd.Series(eval_metrics))

    trainer.save_model("./money_detector_final")
    tokenizer.save_pretrained("./money_detector_final")

    model = trainer.model
    model.eval()
    device = next(model.parameters()).device


if args.eval_only:
    eval_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_eval_batch_size=4,
        report_to="none",
        fp16=True,
    )
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding="longest",
        label_pad_token_id=IGNORE,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

predictions, labels, _ = trainer.predict(test_tokenized)
metrics = compute_metrics((predictions, labels))

def detect_money(text: str) -> list:
    """Inference logic for the trained model."""

    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                         max_length=256, truncation=True)
    offsets = encoding.pop("offset_mapping")[0]
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)[0].tolist()

    spans = []
    i = 0
    while i < len(preds):
        s, e = offsets[i][0].item(), offsets[i][1].item()
        if s == 0 and e == 0:
            i += 1
            continue
        if preds[i] in (LABEL_B, LABEL_I):
            start = s
            end = e
            j = i + 1
            while j < len(preds):
                sj, ej = offsets[j][0].item(), offsets[j][1].item()
                if sj == 0 and ej == 0:
                    j += 1
                    continue
                if preds[j] in (LABEL_B, LABEL_I):
                    end = ej
                    j += 1
                else:
                    break
            spans.append((start, end))
            i = j
        else:
            i += 1
    return spans


nlp = spacy.load("en_core_web_lg")

spacy_true = []
spacy_pred = []

for row in test_set:
    text = row["text"]
    groundtruth = [(s, e) for s, e, _ in row["spans"]]

    doc = nlp(text)
    preds = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "MONEY"]

    true_labels = ["O"] * len(text)
    for s, e in groundtruth:
        true_labels[s] = "B-MONEY"
        for i in range(s + 1, e):
            true_labels[i] = "I-MONEY"

    pred_labels = ["O"] * len(text)
    for s, e in preds:
        pred_labels[s] = "B-MONEY"
        for i in range(s + 1, e):
            pred_labels[i] = "I-MONEY"

    spacy_true.append(true_labels)
    spacy_pred.append(pred_labels)

spacy_metrics = {}

spacy_metrics["f1"] = seqeval_f1(spacy_true, spacy_pred, average="micro")
spacy_metrics["precision"] = seqeval_precision(spacy_true, spacy_pred, average="micro")
spacy_metrics["recall"] = seqeval_recall(spacy_true, spacy_pred, average="micro")


"""## Final Results"""

print(
    pd.DataFrame({
        "Model": ["DeBERTaV3 (ours)", "spaCy en_core_web_lg"],
        "Precision": [metrics["precision"], spacy_metrics["precision"]],
        "Recall": [metrics["recall"], spacy_metrics["recall"]],
        "F1": [metrics["f1"], spacy_metrics["f1"]],
    }).to_string(index=False)
)
