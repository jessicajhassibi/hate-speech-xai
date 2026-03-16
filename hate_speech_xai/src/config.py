MODEL_NAME = "bert-base-uncased"
TRUNCATION = True
PADDING = "max_length"
MAX_LENGTH = 128
NUM_LABELS = 3
TRAINING_ARGS = {
	"output_dir": "./results",
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
