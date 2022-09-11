import abc
from typing import List

from cltl.emotion_extraction.api import Emotion
from cltl.emotion_extraction.emotion_mappings import FaceEmotion



class FaceEmotionExtractor(abc.ABC):
    """Abstract EmotionExtraction Object
    Call face emotion extraction function.
    """

    def extract_face_emotions(self, utterance: str) -> List[FaceEmotion]:
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

