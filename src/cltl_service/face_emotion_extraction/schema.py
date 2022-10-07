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
    def create_text_mentions(cls, face_mention: Mention, emotions: Iterable[Emotion]):
        return cls(cls.__name__, EmotionRecognitionEvent.to_mention(face_mention, emotions))

    @staticmethod
    def to_mention(face_mention: Mention, emotions: Iterable[Emotion]):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = face_mention.segment
        annotations = [Annotation(Emotion.__name__, emotion, __name__, timestamp_now())
                       for emotion in emotions]

        return Mention(str(uuid.uuid4()), segment, annotations)

