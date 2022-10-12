import logging
from typing import List, Callable, Tuple

from cltl.backend.api.camera import Image
from cltl.backend.source.client_source import ClientImageSource
from cltl.backend.spi.image import ImageSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl_service.face_recognition.schema import FaceRecognitionEvent
from emissor.representation.scenario import class_source

from cltl.face_emotion_extraction.api import FaceEmotionExtractor
from cltl_service.face_emotion_extraction.schema import EmotionRecognitionEvent

logger = logging.getLogger(__name__)


class FaceEmotionExtractionService:
    @classmethod
    def from_config(cls, extractor: FaceEmotionExtractor, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.face_emotion_recognition.events")

        def image_loader(url) -> ImageSource:
            return ClientImageSource.from_config(config_manager, url)

        return cls(config.get("topic_input"), config.get("topic_output"),
                   config.get("topic_intention"), config.get("intentions", multi=True),
                   extractor, image_loader, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str,
                 intention_topic: str, intentions: List[str], extractor: FaceEmotionExtractor,
                 image_loader: Callable[[str], ImageSource], event_bus: EventBus, resource_manager: ResourceManager):
        self._extractor = extractor

        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._image_loader = image_loader

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._intention_topic = intention_topic if intention_topic else None
        self._intentions = set(intentions) if intentions else {}

        self._topic_worker = None

        self._speaker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         intentions=self._intentions, intention_topic=self._intention_topic,
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[FaceRecognitionEvent]):
        face_mentions = event.payload.mentions

        # TODO select speaker's face
        if len(face_mentions) != 1:
            logger.info("Skip image with %s faces in emotion recognition", len(face_mentions))
            return

        image, bbox = self._get_image(face_mentions[0])
        emotions = self._extractor.extract_face_emotions(image.image, bbox)
        source = class_source(self._extractor)
        emotion_event = EmotionRecognitionEvent.create_text_mentions(face_mentions[0], emotions, source)

        self._event_bus.publish(self._output_topic, Event.for_payload(emotion_event))

    def _get_image(self, face_mention) -> Tuple[Image, Tuple[int, int, int, int]]:
        image_id = face_mention.segment[0].container_id
        image_location = "cltl-storage:image/" + image_id
        with self._image_loader(image_location) as source:
            image = source.capture()

        bbox = face_mention.segment[0].bounds

        return image, bbox
