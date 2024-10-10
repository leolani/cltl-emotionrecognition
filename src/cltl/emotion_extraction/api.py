import abc
import dataclasses
from typing import Optional, Any, List, Union

from cltl.emotion_extraction.emotion_mappings import EmotionType, GoEmotion, EkmanEmotion, EmoticEmotion, Sentiment


@dataclasses.dataclass
class Emotion:
    """
    Information about an Emotion.
    "Emotion type"
    "Emotion value"
    "confidence"
    "source"
    """
    # TODO Collection[EmotionType]?
    type: EmotionType
    value: str
    confidence: Optional[float]

    def to_enum(self) -> Union[GoEmotion, EkmanEmotion, EmoticEmotion, Sentiment]:
        if self.type == EmotionType.GO:
            return GoEmotion[self.value.upper()]
        if self.type == EmotionType.EKMAN:
            return EkmanEmotion[self.value.upper()]
        if self.type == EmotionType.EMOTIC:
            return EmoticEmotion[self.value.upper()]
        if self.type == EmotionType.SENTIMENT:
            return Sentiment[self.value.upper()]

        raise ValueError("Unknown type: " + self.type)


class EmotionExtractor(abc.ABC):
    """Abstract EmotionExtraction Object
    Call any of the modality specific emotion extraction function.
    """

    def extract_text_emotions(self, utterance: str) -> List[Emotion]:
        """Recognize the emotions of a given utterance.

        Parameters
        ----------
        utterance : str
            The utterance to be analyzed.

        Returns
        -------
        List[Emotion]
            The Emotions extracted from the utterance.
        """
        raise NotImplementedError()

    # @TODO
    def extract_audio_emotions(self, audio: Any) -> List[Emotion]:
        raise NotImplementedError()

