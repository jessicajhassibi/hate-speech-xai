from hate_speech_xai.src.data.preprocessing import load_preprocessed_dataset, tokenize_dataset
from hate_speech_xai.src.models.train import train_transformer


def main():
    train, val, _ = load_preprocessed_dataset()
    train, val = tokenize_dataset(train, val)

    # To train v2:
    # train_transformer(train, val, training_args=TRAINING_ARGS_V2, save_dir=SAVED_MODELS_V2_DIR)
    train_transformer(train, val)

if __name__ == "__main__":
    main()