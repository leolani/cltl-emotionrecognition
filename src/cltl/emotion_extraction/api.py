import abc
import dataclasses
import random
from enum import Enum
from emissor.representation.scenario import ImageSignal, AudioSignal
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

