import abc
import logging

from emissor.representation.scenario import Mention, TextSignal, ImageSignal, AudioSignal
import cltl.emotion_extraction.emotion_mappings as mappings

logger = logging.getLogger(__name__)

#@TODO: major revision is needed,
# THIS IS JUST A BASIC SETUP FOR MULTIMODAL EMOTION DETECTION

class EmotionDetector(abc.ABC):
    def __init__(self):
        self._source = None
        self._scenario_id = None
        self._sentiments = set()
        self._ekman_emotions = set()
        self._go_emotions = set()

    def analyse (self, signal, source: str, scenario_id: str):
        ekman_emotion = {mappings.EkmanEmotion.NEUTRAL, 1.0}
        go_emotion = {mappings.GoEmotion.NEUTRAL, 1.0}
        sentiment = {mappings.Sentiment.NEUTRAL, 1.0}
        self._sentiments.add(sentiment)
        self._ekman_emotions.add(ekman_emotion)
        self._go_emotions.add(go_emotion)
        self._source = source
        self._scenario_id = scenario_id


class TextEmotionDetector(EmotionDetector):
    def __init__(self):
        self._scenario_id = None

    def analyse(self, textSignal: TextSignal, source: str, scenario_id: str):
        ### process the TextSignal to get the emotions and sentiment
        ### replace the next by real code
        ekman_emotion = {mappings.EkmanEmotion.NEUTRAL, 1.0}
        go_emotion = {mappings.GoEmotion.NEUTRAL, 1.0}
        sentiment = {mappings.Sentiment.NEUTRAL, 1.0}
        self._sentiments.add(sentiment)
        self._ekman_emotions.add(ekman_emotion)
        self._go_emotions.add(go_emotion)
        self._source = source
        self._scenario_id = scenario_id


class FaceEmotionDetector(EmotionDetector):
    def __init__(self):
        self._scenario_id = None

    def analyse(self, imageSignal: ImageSignal, source: str, scenario_id: str):
        ### process the ImageSignal to get the emotions and sentiment
        ### replace the next by real code
        ekman_emotion = {mappings.EkmanEmotion.NEUTRAL, 1.0}
        go_emotion = {mappings.GoEmotion.NEUTRAL, 1.0}
        sentiment = {mappings.Sentiment.NEUTRAL, 1.0}
        self._sentiments.add(sentiment)
        self._ekman_emotions.add(ekman_emotion)
        self._go_emotions.add(go_emotion)
        self._source = source
        self._scenario_id = scenario_id


class AudioEmotionDetector(EmotionDetector):
    def __init__(self):
        self._scenario_id = None
        self._audio_emotions = set()

    def analyse(self, audioSignal: AudioSignal, source: str, scenario_id: str):
        ### process the AudioSignal to get the emotions and sentiment
        ### replace the next by real code
        ekman_emotion = {mappings.EkmanEmotion.NEUTRAL, 1.0}
        go_emotion = {mappings.GoEmotion.NEUTRAL, 1.0}
        sentiment = {mappings.Sentiment.NEUTRAL, 1.0}
        self._sentiments.add(sentiment)
        self._ekman_emotions.add(ekman_emotion)
        self._go_emotions.add(go_emotion)
        self._source = source
        self._scenario_id = scenario_id


# class DefaultEmotionExtractor(EmotionExtractor):
#     def __init__(self, text_detector: EmotionDetector, face_detector: EmotionDetector, audio_detector: EmotionDetector):
#         self._text_detector = text_detector
#         self._face_detector = face_detector
#         self._audio_detector = audio_detector
#
#     def extract_text_emotions(self, mentions: List[Mention], scenario_id: str) -> List[TextEmotion]:
#         return [self.create_text_mention(mention, scenario_id)
#                 for mention in self._text_detector.filter_mentions(mentions, scenario_id)]
#
#     def extract_audio_emotions(self, mentions: List[Mention], scenario_id: str) -> List[AudioEmotion]:
#         return [self.create_object_mention(mention, scenario_id)
#                 for mention in self._audio_detector.filter_mentions(mentions, scenario_id)]
#
#     def extract_face_emotions(self, mentions: List[Mention], scenario_id: str) -> List[ImageEmotion]:
#         return [self.create_face_mention(mention, scenario_id)
#                 for mention in self._face_detector.filter_mentions(mentions, scenario_id)]
#
#     def create_face_emotion(self, mention: Mention, scenario_id: str):
#         image_id = mention.id
#         image_path = mention.id
#
#         mention_id = mention.id
#         bounds = mention.segment[0].bounds
#         face_emotion = mention.annotations[0].value.label
#         confidence = 1.0
#
#         return ImageEmotion(image_id, mention_id, None, image_path, bounds,
#                             Emotion(face_emotion, [face_emotion], None, None),
#                             confidence, scenario_id, timestamp_now())
#
#
#     def create_audio_mention(self, mention: Mention, scenario_id: str):
#         author = Emotion.create_person("SPEAKER", None, None)
#         audio_id = mention.id
#         audio_path = mention.id
#
#         mention_id = mention.id
#         bounds = mention.segment[0].to_tuple()
#         # TODO multiple?
#         audio_label = mention.annotations[0].value.label
#         confidence = 1.0
#
#         return AudioEmotion(audio_id, mention_id, None, audio_path, bounds,
#                             Emotion(audio_label, [audio_label], None, None),
#                             confidence, scenario_id, timestamp_now())
#
#
#     def create_text_mention(self, mention: Mention, scenario_id: str):
#         author = Emotion.create_person("SPEAKER", None, None)
#
#         utterance = ""
#
#         segment = mention.segment[0]
#         signal_id = segment.container_id
#         mention_text = mention.annotations[0].value.text
#         mention_label = mention.annotations[0].value.label
#         confidence = 1.0
#
#         return TextEmotion(scenario_id, signal_id, author, utterance, f"{segment.start} - {segment.stop}",
#                            Emotion(mention_text, [mention_label], None, None),
#                            confidence, scenario_id, timestamp_now())
