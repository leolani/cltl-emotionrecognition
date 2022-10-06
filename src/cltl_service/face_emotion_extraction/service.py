import logging
from typing import List

from cltl.combot.event.emissor import ImageSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker

from cltl.face_emotion_extraction.api import FaceEmotionExtractor
from cltl_service.emotion_extraction.schema import EmotionRecognitionEvent

logger = logging.getLogger(__name__)


class FaceEmotionExtractionService:
    @classmethod
    def from_config(cls, extractor: FaceEmotionExtractor, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.face_emotion_recognition.events")

        return cls(config.get("topic_input"), config.get("topic_output"), config.get("topic_scenario"),
                   config.get("topic_intention"), config.get("intentions", multi=True),
                   extractor, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, scenario_topic: str,
                 intention_topic: str, intentions: List[str], extractor: FaceEmotionExtractor,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._extractor = extractor

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic
        self._scenario_topic = scenario_topic

        self._intention_topic = intention_topic if intention_topic else None
        self._intentions = set(intentions) if intentions else {}
        self._active_intentions = {}

        self._topic_worker = None

        self._speaker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic, self._scenario_topic, self._intention_topic],
                                         self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=64,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[ImageSignalEvent]):
        if event.metadata.topic == self._intention_topic:
            self._active_intentions = set(event.payload.intentions)
            logger.info("Set active intentions to %s", self._active_intentions)
            return

        if self._intentions and not (self._active_intentions & self._intentions):
            logger.debug("Skipped event outside intention %s, active: %s (%s)",
                         self._intentions, self._active_intentions, event)
            return
        #@TODO Fix how to get the image file path and the bbox for the human face
        # We need to read the bbox from the human face annotation and get the original image file
        # Below is a dummy implementation as a place holder that does not work.
        image = event.payload.signal.image
        bbox = [0,0,0,0]
        emotions = self._extractor.extract_face_emotions(image, bbox)

        emotion_event = EmotionRecognitionEvent.create_text_mentions(event.payload.signal, emotions)
        self._event_bus.publish(self._output_topic, Event.for_payload(emotion_event))
