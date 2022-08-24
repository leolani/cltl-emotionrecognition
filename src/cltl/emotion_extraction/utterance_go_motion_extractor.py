import logging
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import uuid
import jsonpickle
from transformers import pipeline

from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import Mention, Annotation
from cltl.emotion_extraction.api import EmotionExtractor
import cltl.emotion_extraction.emotion_mappings as mappings
ObjectInfo = namedtuple('ObjectInfo', ('type', 'bbox'))
model_name = "bhadresh-savani/bert-base-go-emotion"


class GoEmotionDetector(EmotionExtractor):
    def __init__(self):
        self.emotion_pipeline = pipeline('sentiment-analysis',  model=model_name, return_all_scores=True)

    def __enter__(self):
        executor = ThreadPoolExecutor(max_workers=2)
        detect = executor.submit(self.detect_infra.__enter__)
        detect.result()
        executor.shutdown()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        executor = ThreadPoolExecutor(max_workers=2)
        detect = executor.submit(lambda: self.detect_infra.__exit__(exc_type, exc_val, exc_tb))
        detect.result()
        executor.shutdown()


    def _detect_emotion_from_utterance(self, utterance: str) -> tuple:
        """Recognize the speaker emotion of a given utterance.
        Args
        ----
        utterance:
        url_erc: the url of the emoberta api server.
        Returns
        -------
        emotion: one of neutral, joy, surprise, anger, sadness, disgust, and fear
        """
        logging.debug(f"sending utterance to server...")
        start = time.time()

        go_emotions =[]
        ekman_emotions = []
        sentiments =[]

        response = self.emotion_pipeline(utterance)
        emotion_labels = mappings.sort_predictions(response[0])
        print(emotion_labels)
        ### We take the highest scoring emotion
        go_emotions.append({emotion_labels[0]['label'], emotion_labels[0]['score']})

        ekman_labels = mappings.get_averaged_mapped_scores(mappings.go_ekman_map, emotion_labels)
        ekman_emotions.append({ekman_labels[0]['label'], ekman_labels[0]['score']})

        sentiment_labels = mappings.get_averaged_mapped_scores(mappings.go_sentiment_map, emotion_labels)
        sentiments.append({sentiment_labels[0]['label'], sentiment_labels[0]['score']})

        logging.info("got %s from server in %s sec", response, time.time()-start)
        logging.info(f"{go_emotions} Go emotion detected!")
        logging.info(f"{ekman_emotions} Ekman emotion detected!")
        logging.info(f"{sentiments} Sentiments detected!")
        return go_emotions,ekman_emotions, sentiments

    def _create_emotion_mention(self, source: str, current_time: int, emotion: str) -> Mention:
        text_annotation = Annotation(AnnotationType.EMOTION.name, emotion, source, current_time)
        mention = Mention(str(uuid.uuid4()), [], [text_annotation])
        return mention


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = GoEmotionDetector()
    go, ekman, sentiment = analyzer._detect_emotion_from_utterance(utterance)

    print(go)
    print(ekman)
    print(sentiment)
