from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, EvalPrediction

from hate_speech_xai.config import MODEL_NAME, NUM_LABELS, TRAINING_ARGS, SAVED_MODELS_DIR


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}


def train_transformer(train: Dataset, val: Dataset, training_args: dict = TRAINING_ARGS, save_dir: Path = SAVED_MODELS_DIR) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    trainer.state.save_to_json(save_dir / "trainer_state.json")