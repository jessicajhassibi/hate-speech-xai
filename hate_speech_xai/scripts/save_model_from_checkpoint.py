from hate_speech_xai.config import CHECKPOINT_DIR, SAVED_MODELS_DIR
from hate_speech_xai.src.models.predict import load_model

# script to load model from checkpoints if saving the model was missed (not part of the usual workflow)
if __name__=="__main__":
	checkpoint = CHECKPOINT_DIR / "checkpoint-1924"  # use the checkpoint mentioned under "best_model_checkpoint" key in trainer_state.json of checkpoints
	tokenizer, model = load_model(source=checkpoint)
	model.save_pretrained(SAVED_MODELS_DIR)
	tokenizer.save_pretrained(SAVED_MODELS_DIR)