import abc
import logging
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import jsonpickle as jsonpickle
import requests
from cltl.combot.infra.time_util import timestamp_now
from cltl.commons.discrete import UtteranceType
from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import Mention, TextSignal, Annotation

from cltl.emotion_extraction.api import EmotionExtractor
import cltl.nlp.api as nlp

logger = logging.getLogger(__name__)



@dataclass
class Source:
    label: str
    type: List[str]
    uri: str


@dataclass
class Emotion:
    label: str
    type: List[str]
    id: str
    uri: str

    @classmethod
    def create_person(cls, label: str, id_: str, uri: str):
        return cls(label, ["emotion"], id_, uri)


@dataclass
class ImageEmotion:
    visual: str
    detection: str
    source: Source
    image: str
    region: Tuple[int, int, int, int]
    item: Emotion
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.IMAGE_MENTION


@dataclass
class TextEmotion:
    chat: str
    turn: str
    author: Emotion
    utterance: str
    position: str
    item: Emotion
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.TEXT_MENTION


@dataclass
class AudioEmotion:
    audio: str
    detection: str
    source: Source
    image: str
    item: Emotion
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.IMAGE_MENTION


_IMAGE_SOURCE = Source("front-camera", ["sensor"], "http://cltl.nl/leolani/inputs/front-camera")


class EmotionDetector(abc.ABC):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        return mentions


class TextEmotionDetector(EmotionDetector):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        filtered = []
        for mention in mentions:
            annotations = [annotation for annotation in mention.annotations
                           if (annotation.type == nlp.NamedEntity.__name__ or annotation.type == nlp.Entity.__name__)]
            if annotations:
                filtered.append(Mention(mention.id, mention.segment, annotations))

        return filtered


class FaceEmotionDetector(EmotionDetector):
    def __init__(self):
        self._scenario_id = None
        self._face_emotions = set()

    def filter_emotions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._faces = set()

        face_emotions = [mention for mention in mentions
                             if (mention.annotations
                                 and mention.annotations[0].value is not None
                                 and mention.annotations[0].value not in self._faces)]

        self._face_emotions = self._face_emotions | {mention.annotations[0].value for mention in face_emotions}

        return face_emotions


class AudioEmotionDetector(EmotionDetector):
    def __init__(self):
        self._scenario_id = None
        self._audio_emotions = set()

    def filter_emotions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._faces = set()

        audio_emotions = [mention for mention in mentions
                             if (mention.annotations
                                 and mention.annotations[0].value is not None
                                 and mention.annotations[0].value not in self._faces)]

        self._audio_emotions = self._audio_emotions | {emotion.annotations[0].value for emotion in audio_emotions}

        return audio_emotions


class DefaultEmotionExtractor(EmotionExtractor):
    def __init__(self, text_detector: EmotionDetector, face_detector: EmotionDetector, audio_detector: EmotionDetector):
        self._text_detector = text_detector
        self._face_detector = face_detector
        self._audio_detector = audio_detector

    def extract_text_emotions(self, mentions: List[Mention], scenario_id: str) -> List[TextEmotion]:
        return [self.create_text_mention(mention, scenario_id)
                for mention in self._text_detector.filter_mentions(mentions, scenario_id)]

    def extract_audio_emotions(self, mentions: List[Mention], scenario_id: str) -> List[AudioEmotion]:
        return [self.create_object_mention(mention, scenario_id)
                for mention in self._audio_detector.filter_mentions(mentions, scenario_id)]

    def extract_face_emotions(self, mentions: List[Mention], scenario_id: str) -> List[ImageEmotion]:
        return [self.create_face_mention(mention, scenario_id)
                for mention in self._face_detector.filter_mentions(mentions, scenario_id)]

    def create_face_emotion(self, mention: Mention, scenario_id: str):
        image_id = mention.id
        image_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].bounds
        face_id = mention.annotations[0].value
        confidence = 1.0

        return ImageEmotion(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                            Emotion(face_id, ["face"], face_id, None),
                            confidence, scenario_id, timestamp_now())

    def create_audion_mention(self, mention: Mention, scenario_id: str):
        author = Emotion.create_person("SPEAKER", None, None)
        audio_id = mention.id
        audio_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].to_tuple()
        # TODO multiple?
        object_label = mention.annotations[0].value.type
        confidence = 1.0

        return AudioEmotion(audio_id, mention_id, _IMAGE_SOURCE, audio_path, bounds,
                            Emotion(object_label, [object_label], None, None),
                            confidence, scenario_id, timestamp_now())

    def create_text_mention(self, mention: Mention, scenario_id: str):
        author = Emotion.create_person("SPEAKER", None, None)

        utterance = ""

        segment = mention.segment[0]
        signal_id = segment.container_id
        entity_text = mention.annotations[0].value.text
        entity_type = mention.annotations[0].value.label
        confidence = 1.0

        return TextEmotion(scenario_id, signal_id, author, utterance, f"{segment.start} - {segment.stop}",
                           Emotion(entity_text, [entity_type], None, None),
                           confidence, scenario_id, timestamp_now())
