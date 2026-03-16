import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

from hate_speech_xai.src.config import MODEL_NAME, NUM_LABELS, TRAINING_ARGS

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}


def train_transformer(train, val):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAINING_ARGS),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()