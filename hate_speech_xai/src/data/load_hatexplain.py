from datasets import load_dataset

def load_hatexplain_dataset():
    """
    Load the HateXplain dataset from Hugging Face.
    Returns train, validation, and test splits.
    It will be stored in the huggingface_hub cache to be re-used,
    so calling the function again will not re-download it again
    https://huggingface.co/docs/datasets/cache
    """
    dataset = load_dataset("Hate-speech-CNERG/hatexplain")

    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    return train_ds, val_ds, test_ds