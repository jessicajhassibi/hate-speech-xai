import json

from hate_speech_xai.config import SAVED_MODELS_DIR
from hate_speech_xai.src.data.load_hatexplain import load_hatexplain_dataset
from hate_speech_xai.src.models.evaluate_xai import evaluate_all_xai_methods


def main():
    _, _, test_ds = load_hatexplain_dataset()

    # its slow due to the number of forward passes per sample in the Integrated Gradients method
    # also SHAP takes a long time to run
    # if you want to reproduce that it works and speed it up, you can use:
    # results = evaluate_all_xai_methods(test_ds, max_samples=200)
    results = evaluate_all_xai_methods(test_ds)

    # to evaluate v2:
    # output_path = SAVED_MODELS_V2_DIR / "test_evaluation_v2.json"
    output_path = SAVED_MODELS_DIR / "xai_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()