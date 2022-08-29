import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, TextSignal, Annotation
from typing import Iterable

from cltl.emotion_extraction.api import Emotion


@dataclass
class EmotionRecognitionEvent(AnnotationEvent[Annotation[Emotion]]):
    @classmethod
    def create_text_mentions(cls, text_signal: TextSignal, emotions: Iterable[Emotion]):
        if emotions:
            mentions = [EmotionRecognitionEvent.to_mention(text_signal, emotion)
                        for emotion in emotions]
        else:
            mentions = [EmotionRecognitionEvent.to_mention(text_signal)]

        return cls(cls.__name__, mentions)

    @staticmethod
    def to_mention(text_signal: TextSignal, emotion: Emotion = None):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = text_signal.ruler
        ### Uses more specific emotion type
        annotation = Annotation(emotion.type, emotion.value, __name__, timestamp_now())

        ### Uses the general Emotion type
        #annotation = Annotation(Emotion.__name__, emotion.value, __name__, timestamp_now())
        return Mention(str(uuid.uuid4()), [segment], [annotation])

