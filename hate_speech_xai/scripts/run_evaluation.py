import json
from hate_speech_xai.config import SAVED_MODELS_DIR, SAVED_MODELS_V2_DIR
from hate_speech_xai.src.data.preprocess import load_preprocessed_dataset
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
    # to evaluate v2:
    # output_path = SAVED_MODELS_V2_DIR / "test_evaluation_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()