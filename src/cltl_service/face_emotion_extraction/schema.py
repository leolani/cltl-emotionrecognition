import uuid
from dataclasses import dataclass
from typing import Iterable

from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from emissor.representation.scenario import Mention, Annotation
from emissor.representation.scenario import class_type

from cltl.emotion_extraction.api import Emotion


@dataclass
class EmotionRecognitionEvent(AnnotationEvent[Annotation[Emotion]]):
    @classmethod
    def create_text_mentions(cls, face_mention: Mention, emotions: Iterable[Emotion], source: str):
        return cls(class_type(cls), [EmotionRecognitionEvent.to_mention(face_mention, emotions, source)])

    @staticmethod
    def to_mention(face_mention: Mention, emotions: Iterable[Emotion], source: str) -> Mention:
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = face_mention.segment
        annotations = [Annotation(class_type(Emotion), emotion, source, timestamp_now())
                       for emotion in emotions]

        return Mention(str(uuid.uuid4()), segment, annotations)

