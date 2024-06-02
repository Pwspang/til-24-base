from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline

class ASRManager:
    # To Do: Add noise reduction
    def __init__(self):
        device = "cuda:0" 
        model = WhisperForConditionalGeneration.from_pretrained("./ASR_Model")
        processor = WhisperProcessor.from_pretrained("./ASR_Model")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            device=device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

        print("Model is ready.")
    

    def transcribe(self, audio_bytes: bytes) -> str:
        
        return self.pipe(audio_bytes)['text']

if __name__ == "__main__":
    asr = ASRManager()
    with open("/home/jupyter/advanced/audio/audio_1670.wav", "rb") as file:
        print(asr.transcribe(file.read()))
    