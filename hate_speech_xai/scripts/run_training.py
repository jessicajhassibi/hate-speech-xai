from hate_speech_xai.src.data.preprocessing import load_preprocessed_dataset, tokenize_dataset
from hate_speech_xai.src.models.train import train_transformer


def main():
    train, val, _ = load_preprocessed_dataset()
    train, val = tokenize_dataset(train, val)

    train_transformer(train, val)

if __name__ == "__main__":
    main()