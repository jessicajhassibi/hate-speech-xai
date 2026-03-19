# hate-speech-xai
# Explainabile AI on Hate Speech Detection

This repository contains the code and experiments for an Explainable AI (XAI) study on a Hate Speech dataset. I
The report is currently being written under [this link](https://www.overleaf.com/read/bvdccnjzcstc#da316e)

## What are we doing here?
We are interested in understanding the decision-making process of a hate speech detection model. 
We will be using the HateXplain dataset [1], which contains social media posts labeled for hate speech, 
along with rationales that indicate why exactly a post is considered toxic or not. 
We will train a machine learning model on this dataset and then apply various XAI techniques 
to understand which features are most influential in the models decision of whether a post is classified as hate speech or not.

### References
[1] [HateXplain Dataset](https://github.com/hate-alert/HateXplain).
Used in the following paper:
@inproceedings{mathew2021hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}

## Running the project
> **Note:** The preprocessed data and trained model are already provided, you don't need to run the whole pipeline again. Only need to install the dependencies and start the app. The preprocessing, training, and evaluation steps can be skipped.

### Option A: using Makefile
#### Installation of dependencies
`make install`
#### Accessing the data
The dataset is accessed via Hugging Face: [Link to data](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain).
There is no need to download it manually. The `datasets` library will handle the download and caching of the dataset when you run the app for the first time.
#### Preprocessing the data
`make preprocess`
#### Training the model
`make train`
#### Evaluating the model
`make evaluate`
#### Starting the app
`make app`

### Option B: manual
```
pip install -r requirements.txt
pip install -e .
python -m hate_speech_xai.scripts.run_preprocessing
python -m hate_speech_xai.scripts.run_training
python -m hate_speech_xai.scripts.run_evaluation
streamlit run hate_speech_xai/app/app.py
```

## Visualizations


