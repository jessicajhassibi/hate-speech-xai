import json
from hate_speech_xai.config import SAVED_MODELS_DIR
from hate_speech_xai.src.data.preprocessing import load_preprocessed_dataset
from hate_speech_xai.src.models.predict import predict_label


def main():
    _, _, test_ds = load_preprocessed_dataset()

    y_true = []
    y_pred = []
    for i in range(len(test_ds)):
        example = test_ds[i]
        pred_label = predict_label(example["text"])
        y_true.append(example["label"]) # we are using the majority label from the annotators as the true label
        y_pred.append(pred_label)

    results = {"y_true": y_true, "y_pred": y_pred}
    output_path = SAVED_MODELS_DIR / "test_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()