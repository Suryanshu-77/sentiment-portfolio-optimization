"""Fine-tune a general transformer on financial sentiment (PhraseBank + Twitter financial news).

The held-out test set is the Twitter *validation* split: ProsusAI FinBERT was trained on
PhraseBank, so testing on PhraseBank would favour it.
"""

import logging
from pathlib import Path

import numpy as np

from sdpo import config
from sdpo.nlp.evaluate import classification_report_dict

log = logging.getLogger(__name__)

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
PHRASEBANK_MAP = {0: 0, 1: 1, 2: 2}  # negative, neutral, positive
TWITTER_MAP = {0: 0, 1: 2, 2: 1}  # Bearish -> neg, Bullish -> pos, Neutral -> neu


def _phrasebank():
    """PhraseBank sentences_allagree (2264 rows), all splits combined: it is training-only here.

    takala/financial_phrasebank is script-based, which `datasets` >= 4 no longer runs and the
    Hub has no parquet conversion for, so we use the gtfintechlab parquet mirror. Its labels
    were verified to be 0 negative / 1 neutral / 2 positive (class mix matches the paper)."""
    from datasets import concatenate_datasets, load_dataset

    d = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
    return concatenate_datasets([d[s].select_columns(["sentence", "label"]) for s in d])


def load_finetune_data():
    """DatasetDict(train=PhraseBank allagree + Twitter train, test=Twitter validation),
    columns text, label (unified ids)."""
    from datasets import DatasetDict, concatenate_datasets, load_dataset

    pb = _phrasebank().map(lambda b: {"text": b["sentence"],
                                      "label": [PHRASEBANK_MAP[x] for x in b["label"]]},
                           batched=True, remove_columns=["sentence"])
    tw = load_dataset("zeroshot/twitter-financial-news-sentiment")
    tw = tw.map(lambda b: {"label": [TWITTER_MAP[x] for x in b["label"]]}, batched=True)
    cols = ["text", "label"]
    train = concatenate_datasets([pb.select_columns(cols), tw["train"].select_columns(cols)])
    return DatasetDict(train=train.shuffle(seed=config.SEED),
                       test=tw["validation"].select_columns(cols))


def finetune(out_dir: Path, base: str = "microsoft/deberta-v3-base") -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    data = load_finetune_data()
    split = data["train"].train_test_split(test_size=0.1, seed=config.SEED)
    tok = AutoTokenizer.from_pretrained(base)
    enc = split.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()})

    def metrics(p):
        return classification_report_dict(p.label_ids, np.argmax(p.predictions, -1))

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=32, per_device_eval_batch_size=128, bf16=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", seed=config.SEED,
        report_to="none", logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=enc["train"],
                      eval_dataset=enc["test"], data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=lambda p: {k: v for k, v in metrics(p).items()
                                                 if k != "confusion"})
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    log.info("saved fine-tuned model to %s", out_dir)
