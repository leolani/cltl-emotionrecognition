import abc
import logging
from dataclasses import dataclass
from typing import List, Tuple

from cltl.commons.discrete import UtteranceType
from emissor.representation.scenario import Mention

logger = logging.getLogger(__name__)

@dataclass
class Source:
    label: str
    type: List[str]
    uri: str


@dataclass
class Entity:
    label: str
    type: List[str]
    id: str
    uri: str

    @classmethod
    def create_person(cls, label: str, id: str, uri: str):
        return cls(label, ["person"], id, uri)


@dataclass
class ImageEmotion:
    visual: str
    detection: str
    source:Source
    image: str
    region: Tuple[int, int, int, int]
    item: Entity
    confidence: float
    context_id: str


@dataclass
class AudioEmotion:
    audio: str
    detection: str
    author: Entity
    utterance: str
    position: str
    sound: str
    item: Entity
    confidence: float
    context_id: str


@dataclass
class TextEmotion:
    chat: str
    turn: str
    author: Entity
    utterance: str
    position: str
    item: Entity
    confidence: float
    context_id: str
    utterance_type: UtteranceType = UtteranceType.IMAGE_MENTION


_IMAGE_SOURCE = Source("front-camera", ["sensor"], "http://cltl.nl/leolani/inputs/front-camera")


class EmotionExtractor(abc.ABC):
    def extract_text_emotions(self, emotions: List[Emotion], scenario_id: str) -> List[TextEmotion]:
        raise NotImplementedError()

    def extract_audio_emotions(self, mentions: List[Mention], scenario_id: str) -> List[AudioEmotion]:
        raise NotImplementedError()

    def extract_face_emotions(self, mentions: List[Mention], scenario_id: str) -> List[ImageEmotion]:
        raise NotImplementedError()