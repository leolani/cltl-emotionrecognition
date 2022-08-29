import logging
from typing import List

from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker

from cltl.emotion_extraction.emotion_extractor import EmotionExtractorImpl

logger = logging.getLogger(__name__)

class EmotionExtractionService:
    @classmethod
    def from_config(cls, extractor: EmotionExtractorImpl, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.emotion_extraction")

        return cls(config.get("topic_input"), config.get("topic_output"), config.get("topic_scenario"),
                   config.get("topic_intention"), config.get("intentions", multi=True),
                   extractor, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, scenario_topic: str,
                 intention_topic: str, intentions: List[str], extractor: EmotionExtractorImpl,
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

    def _process(self, event: Event):
        if event.metadata.topic == self._intention_topic:
            self._active_intentions = set(event.payload.intentions)
            logger.info("Set active intentions to %s", self._active_intentions)
            return

        if self._intentions and not (self._active_intentions & self._intentions):
            logger.debug("Skipped event outside intention %s, active: %s (%s)",
                         self._intentions, self._active_intentions, event)
            return
        utterance= event.payload.signal.text
        emotions = self._extractor.analyze(utterance, self._speaker, None)
       # capsule = self._emotions_to_capsules(emotions, event.payload.signal)
        response = self._extractor.respond(self._speaker)
        if response:
            # TODO: transform capsules into proper EMISSOR annotations
            self._event_bus.publish(self._output_topic, Event.for_payload(response))
            logger.debug("Published %s emotions for signal %s (%s): %s",
                         len(response), event.payload.signal.id, event.payload.signal.text, response)
        else:
            logger.debug("No emotions for signal %s (%s)", event.payload.signal.id, event.payload.signal.text)

    def _emotions_to_capsules(self, emotions, signal):
        capsules = []

        for emotion in emotions:
            scenario_id = signal.time.container_id

            capsule = {"chat": scenario_id,
                       "turn": signal.id,
                       "author": self._get_author(),
                       ###
                       "perspective": self._extract_perspective(emotion),
                       ###
                       "context_id": None,
                       "timestamp": timestamp_now()
                       }

            capsules.append(capsule)

        return capsules

    def _extract_perspective(emotion):
        """
        This function extracts perspective from emotions
        :return: perspective dictionary consisting of emotion, sentiment, certainty, and polarity value
        """
        certainty = 1  # Possible
        polarity = 1  # Positive
        sentiment = ""  # Underspecified
        emotion = ""  # Underspecified

        if emotion["sentiment"]:
            sentiment=emotion["sentiment"]
        if emotion["emotion"]:
            emotion=emotion["emotion"]

        perspective = {'sentiment': sentiment,
                       'certainty': float(certainty),
                       'polarity': float(polarity),
                       'emotion': emotion}
        return perspective

    def _get_author(self):
        return {
            "label": self._speaker.name if self._speaker and self._speaker.name else self._chat.speaker,
            "type": ["person"],
            "uri": self._speaker.uri if self._speaker else None
        }
