import abc
from typing import Collection

from cltl.emotion_extraction.api import Emotion


class EmotionResponder(abc.ABC):
    def respond(self, emotions: Collection[Emotion], speaker: str) -> str:
        """
        Create a text response to a collection of emotions.

        Parameters
        ----------
        emotions : Collection[Emotion]
            The emotions to respond to.
        speaker : str
            The addressee of the text response.
        Returns
        -------
        str
            The text response.
        """
        raise NotImplementedError("")
