from faster_whisper import WhisperModel

model = WhisperModel("base")

segments, info = model.transcribe("fraud_detection_sample.wav")

for segment in segments:
    print(segment.text)