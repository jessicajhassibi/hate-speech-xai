from hate_speech_xai.config import PREPROCESSED_DATA_DIR
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.data.preprocessing import preprocess_dataset


def main():
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_hatexplain_dataset()
    train, val, test = preprocess_dataset(train, val, test)

    train.save_to_disk(str(PREPROCESSED_DATA_DIR / "train"))
    val.save_to_disk(str(PREPROCESSED_DATA_DIR / "val"))
    test.save_to_disk(str(PREPROCESSED_DATA_DIR / "test"))

    print(f"Saved preprocessed dataset to {PREPROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()