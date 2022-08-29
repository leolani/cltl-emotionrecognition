import abc
import dataclasses
import random
from enum import Enum
from emissor.representation.scenario import TextSignal, ImageSignal, AudioSignal
import emotion_sentences as responses
from typing import Optional

class EmotionTypes(Enum):
    EKMAN = "ekman"
    GO = "go"
    SENTIMENT = "sentiment"

@dataclasses.dataclass
class Emotion:
    """
    Information about an Emotion.
    "Emotion type"
    "Emotion value"
    "confidence"
    "source"
    """
    # TODO switch to np.typing.ArrayLike
    type: Optional[EmotionTypes]
    value: Optional[str]
    confidence: Optional[float]
    source: Optional[str]


class EmotionExtractor(abc.ABC):

    def __init__(self):
        """
        Abstract EmotionExtraction Object: call any of the modality specific emotion extraction function

        Parameters
        ----------
        """

        self._go_emotions = [Emotion]
        self._ekman_emotions = [Emotion]
        self._sentiments = [Emotion]

    def _extract_text_emotions(self, utterance:str, source: str):
        raise NotImplementedError()

    # @TODO
    def _extract_audio_emotions(self, audioSignal: AudioSignal, source: str):
        raise NotImplementedError()
    # @TODO
    def _extract_face_emotions(self, imageSignal: ImageSignal, source: str):
        raise NotImplementedError()

    def _respond_to_emotions(self, speaker_name: str = None) -> str:
        results = []
        results.extend(responses.respond_to_go(self.go_emotions))
        results.extend(responses.respond_to_ekman(self.ekman_emotions))
        results.extend(responses.respond_to_sentiment(self.sentiments))
        if results:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker_name}," if speaker_name else "")
            for result in results:
                say += "," + result
            return say

