import abc
import logging
from dataclasses import dataclass
from typing import List, Tuple
from emissor.representation.scenario import TextSignal, ImageSignal, AudioSignal

logger = logging.getLogger(__name__)

@dataclass
class Source:
    label: str
    type: List[str]
    uri: str

@dataclass
class Emotion:
    source:Source
    item: str
    confidence: float
    context_id: str

@dataclass
class ImageEmotion(Emotion):
    visual: str
    detection: str
    image: str
    region: Tuple[int, int, int, int]


@dataclass
class AudioEmotion(Emotion):
    audio: str
    detection: str
    position: str
    sound: str


@dataclass
class TextEmotion(Emotion):
    chat: str
    turn: str
    utterance: str
    position: str


class EmotionExtractor(abc.ABC):
    def extract_text_emotions(self, textSignal:TextSignal, source: str, scenario_id:str):
        raise NotImplementedError()

    def extract_audio_emotions(self, audioSignal: AudioSignal, source: str, scenario_id:str):
        raise NotImplementedError()

    def extract_face_emotions(self, imageSignal: ImageSignal, source: str, scenario_id:str):
        raise NotImplementedError()

