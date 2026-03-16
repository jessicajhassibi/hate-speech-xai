import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "hatexplain_classifier")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model and training configuration
MODEL_NAME = "bert-base-uncased"
TRUNCATION = True
PADDING = "max_length"
MAX_LENGTH = 128
NUM_LABELS = 3
TRAINING_ARGS = {
	"output_dir": CHECKPOINT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "logging_strategy": "steps",
    "logging_steps": 50,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "save_total_limit": 2
} # using standard huggingface arguments
