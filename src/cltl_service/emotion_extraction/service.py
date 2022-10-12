import logging
from typing import List

from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import class_source

from cltl.emotion_extraction.api import EmotionExtractor
from cltl_service.emotion_extraction.schema import EmotionRecognitionEvent

logger = logging.getLogger(__name__)


class EmotionExtractionService:
    @classmethod
    def from_config(cls, extractor: EmotionExtractor, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.emotion_recognition.events")

        return cls(config.get("topic_input"), config.get("topic_output"),
                   config.get("topic_intention"), config.get("intentions", multi=True),
                   extractor, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str,
                 intention_topic: str, intentions: List[str], extractor: EmotionExtractor,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._extractor = extractor

        self._event_bus = event_bus
        self._resource_manager = resource_manager

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

    def _process(self, event: Event[TextSignalEvent]):
        utterance= event.payload.signal.text
        emotions = self._extractor.extract_text_emotions(utterance)
        source = class_source(self._extractor)
        emotion_event = EmotionRecognitionEvent.create_text_mentions(event.payload.signal, emotions, source)

        self._event_bus.publish(self._output_topic, Event.for_payload(emotion_event))
