from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.data.preprocess import preprocess_dataset


def main():
    train, val, test = load_hatexplain_dataset()
    preprocess_dataset(train, val, test)


if __name__ == "__main__":
    main()