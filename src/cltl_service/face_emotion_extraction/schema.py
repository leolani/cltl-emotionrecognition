import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, ImageSignal, Annotation
from typing import Iterable

from cltl.emotion_extraction.api import FaceEmotion


@dataclass
class FaceEmotionRecognitionEvent(AnnotationEvent[Annotation[FaceEmotion]]):
    @classmethod
    def create_text_mentions(cls, image_signal: ImageSignal, emotions: Iterable[FaceEmotion]):
        return cls(cls.__name__, EmotionRecognitionEvent.to_mention(image_signal, emotions))

    @staticmethod
    def to_mention(image_signal: ImageSignal, emotions: Iterable[FaceEmotion]):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        #@TODO this needs to be fixed for face annotation
        segment = image_signal.ruler
        annotations = [Annotation(Emotion.__class__.__name__, emotion, __name__, timestamp_now())
                       for emotion in emotions]

        return Mention(str(uuid.uuid4()), [segment], annotations)

