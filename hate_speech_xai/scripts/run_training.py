from hate_speech_xai.src.data.preprocessing import preprocess_dataset
from hate_speech_xai.src.models.train import train_transformer


def main():
    train, val, test = preprocess_dataset()
    train_transformer(train, val)

if __name__ == "__main__":
    main()