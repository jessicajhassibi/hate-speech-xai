# hate-speech-xai
# Explainabile AI on Hate Speech Detection

This repository contains the code and experiments for an Explainable AI (XAI) study on a Hate Speech dataset. 

Practical Project made by Jessica Hassibi, Winter Semester 2025/26, 
for the Practical Course AI and Security by TU Darmstadt and the Fraunhofer SIT of Darmstadt.
This project is based on and builds upon the research of Mathew et al. [1].

The report can be found under [this link](https://sharelatex.tu-darmstadt.de/read/ygjkwnvqbytr#9f0b95) (only accessible with this link).

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
The preprocessed data and the trained model are already provided if the repository is shared with you as a zip file.
Also the Dockerfile is provided in the zip file, so no need to publish it to Docker Hub.

### Option A: Docker
> **Note:** If you just want to run the app, you can do this with Docker without having to install the dependencies.
Needs [Docker](https://www.docker.com/) to be installed on your system. 
```
docker build -t hate-speech-xai .
docker run -p 8501:8501 hate-speech-xai
```
The app starts at http://localhost:8501 in your browser.

### Option B: using Makefile
> **Note:** With Option B & C you only need to install the dependencies and start the app (steps 1 and 6). You could also run the whole pipeline again if you want with the following commands.
#### 1) Installation of dependencies
`make install`
#### 2) Accessing the data
The dataset is accessed via Hugging Face: [Link to data](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain).
There is no need to download it manually. The `datasets` library will handle the download and caching of the dataset when you run the app for the first time.
#### 3) Preprocessing the data
`make preprocess`
#### 4) Training the model
`make train`
#### 5) Evaluating the model
`make evaluate`
#### 6) Starting the app
`make app`

### Option C: manual
```
pip install -r requirements.txt
pip install -e .
python -m hate_speech_xai.scripts.run_preprocessing
python -m hate_speech_xai.scripts.run_training
python -m hate_speech_xai.scripts.run_evaluation
streamlit run hate_speech_xai/app/app.py
```

## Preview of the App


