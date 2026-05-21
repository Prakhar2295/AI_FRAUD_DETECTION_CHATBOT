"""Speech Emotion Recognition service."""

# from future import annotations
from __future__ import annotations
import logging
from typing import Any

import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

logger = logging.getLogger("AudioEmotionService")


class AudioEmotionService:

    _model = None

    def init(
        self,
        model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    ) -> None:

        self.model_name = model_name

        if AudioEmotionService._model is None:
            self._load_model()

    def _load_model(self) -> None:

        try:

            logger.info("Loading SER model: %s", self.model_name)

            AudioEmotionService._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir="pretrained_models/emotion_recognition",
            )

            logger.info("SER model loaded successfully")

        except Exception as exc:

            logger.exception("Failed to load SER model: %s", exc)

            AudioEmotionService._model = None

    def analyze_file(self, wav_path: str) -> dict[str, Any]:

        if AudioEmotionService._model is None:
            return self._fallback()

        try:

            classifier = AudioEmotionService._model

            signal, sample_rate = torchaudio.load(wav_path)

            # Convert stereo -> mono
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != 16000:

                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000,
                )

                signal = resampler(signal)

            prediction = classifier.classify_batch(signal)

            out_prob, score, index, label = prediction

            dominant_emotion = label[0]

            confidence = float(score.squeeze().item())

            emotions = {
                dominant_emotion: confidence
            }

            audio_behavioral_signals = {
                "high_stress": dominant_emotion in ["angry", "fearful"],
                "emotional_instability": dominant_emotion in ["sad", "fearful"],
                "aggressive_tone": dominant_emotion == "angry",
            }

            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_probabilities": emotions,
                "audio_behavioral_signals": audio_behavioral_signals,
            }

        except Exception as exc:

            logger.exception("SER inference failed: %s", exc)

            return self._fallback()

    def _fallback(self) -> dict[str, Any]:

        return {
            "dominant_emotion": "unknown",
            "confidence": 0.0,
            "emotion_probabilities": {},
            "audio_behavioral_signals": {
                "high_stress": False,
                "emotional_instability": False,
                "aggressive_tone": False,
            },
        }