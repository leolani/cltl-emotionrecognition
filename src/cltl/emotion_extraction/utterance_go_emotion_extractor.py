import logging
import time
from typing import Any, List

from transformers import pipeline

import cltl.emotion_extraction.emotion_mappings as mappings
from cltl.emotion_extraction.api import EmotionExtractor, EmotionType, Emotion

logger = logging.getLogger(__name__)


_MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
_THRESHOLD = 0.5


# GO Emotions is a finetuned BERT transformer with the GO Emotion data
#https://github.com/google-research/google-research/tree/master/goemotions
#https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf


class GoEmotionDetector(EmotionExtractor):
    def __init__(self, model: str = _MODEL_NAME):
        self.emotion_pipeline = pipeline('sentiment-analysis',  model=model, return_all_scores=True)

    def extract_audio_emotions(self, audio_signal: Any) -> List[Emotion]:
        raise NotImplementedError()

    def extract_text_emotions(self, utterance: str) -> List[Emotion]:
        if not utterance:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()

        emotions = []

        response = self.emotion_pipeline(utterance)

        emotion_labels = mappings.sort_predictions(response[0])
        emotions.extend(self._filter_by_threshold(EmotionType.GO, emotion_labels))

        ekman_labels = mappings.get_total_mapped_scores(mappings.go_ekman_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.EKMAN, ekman_labels))

        sentiment_labels = mappings.get_total_mapped_scores(mappings.go_sentiment_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.SENTIMENT, sentiment_labels))

        self._log_results(emotions, response, start)

        return emotions

    def _filter_by_threshold(self, emotion_type, results):
        return [Emotion(type=emotion_type, value=result['label'], confidence=result['score'])
                for result in results
                if result['score'] > 0 and result['score'] / results[0]['score'] > _THRESHOLD]

    def _log_results(self, emotions, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("All Go emotion detected: %s", [emotion.value for emotion in emotions
                                                    if emotion.type == EmotionType.GO])
        logger.info("Highest scoring Go emotion: %s", next(emotion.value for emotion in emotions
                                                           if emotion.type == EmotionType.GO))
        logger.info("Highest scoring Ekman emotion: %s", next(emotion.value for emotion in emotions
                                                              if emotion.type == EmotionType.EKMAN))
        logger.info("Highest scoring Sentiment: %s", next(emotion.value for emotion in emotions
                                                          if emotion.type == EmotionType.SENTIMENT))


if __name__ == "__main__":
    utterance = "I love cats."
    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/bert-base-go-emotion"
    analyzer = GoEmotionDetector(model=model_path)
    emotions = analyzer.extract_text_emotions(utterance)
    emotion_json ={}
    for emotion in emotions:
        emotion_json.update({'type': emotion.type, 'value':emotion.value, 'confident': emotion.confidence})
        print(emotion_json)
