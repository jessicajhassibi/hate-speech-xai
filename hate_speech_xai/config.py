from pathlib import Path

BASE_DIR = Path(__file__).parent
SAVED_MODELS_DIR = Path(BASE_DIR) / "src/models/hatexplain_classifier"
CHECKPOINT_DIR = Path(BASE_DIR) / "checkpoints"
DATA_DIR = Path(BASE_DIR) / "data"

# Labels in HateXplain
LABELS = {
    0: "Hate speech",
    1: "Normal",
    2: "Offensive"
}

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
