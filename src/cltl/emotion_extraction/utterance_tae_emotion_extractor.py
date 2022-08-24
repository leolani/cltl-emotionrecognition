import logging
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import Mention, Annotation
import jsonpickle
import requests

from cltl.emotion_extraction.api import TextEmotion
from cltl.combot.infra.docker import DockerInfra
from cltl.emotion_extraction.emotion_extractor import Analyzer

#docker pull tae898/emoberta-base
#https://hub.docker.com/r/tae898/emoberta-base

class EmotionDetectorProxy(Analyzer):
    def __init__(self):
        self.detect_infra = DockerInfra('tae898/erc', 10006, 10006, False, 15)

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

    def _detect_emotion_from_utterance(self, utterance: str, speaker: str, url_erc: str = "http://127.0.0.1:10006/") -> tuple:
        """Recognize the speaker emotion of a given utterance.
        Args
        ----
        utterance:
        url_erc: the url of the emoberta api server.
        Returns
        -------
        emotion: one of neutral, joy, surprise, anger, sadness, disgust, and fear
        """
        start = time.time()
        logging.debug(f"sending utterance to server...")
        data = {"text": utterance}
        data = jsonpickle.encode(data)
        response = requests.post(url_erc, json=data)
        logging.info("got %s from server in %s sec", response,time.time()-start)
        response = jsonpickle.decode(response.text)
        emotion = max(response, key=response.get)
        emotion.upper()  # in order to be compatible with EMISSOR.
        textEmotion = TextEmotion()
        textEmotion.author = speaker
        #textEmotion.confidence = 1.0
        textEmotion.item = emotion
        logging.info(f"{emotion} emotion detected!")

        return textEmotion

    def _create_emotion_mention(self, source: str, speaker, current_time: int, emotion: str) -> Mention:
        text_annotation = Annotation(AnnotationType.EMOTION.name, emotion, source, current_time)
        mention = Mention(str(uuid.uuid4()), [], [text_annotation])
        return mention


if __name__ == "__main__":
    '''
   
    '''
    utterance = "I love cats."
    analyzer = EmotionDetectorProxy()
    emotion = analyzer._detect_emotion_from_utterance(utterance, "Piek")

    print(emotion)
