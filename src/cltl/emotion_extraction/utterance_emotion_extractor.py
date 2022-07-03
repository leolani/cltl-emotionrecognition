import logging
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple

import cv2
import jsonpickle
import numpy as np
import requests
from cltl.backend.api.camera import Bounds

from cltl.object_recognition.api import Object, ObjectDetector
from cltl.combot.infra.docker import DockerInfra


ObjectInfo = namedtuple('ObjectInfo', ('type', 'bbox'))


class ObjectDetectorProxy(ObjectDetector):
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


    def _detect_emotion_from_utterance(self, utterance: str, url_erc: str = "http://127.0.0.1:10006/") -> tuple:
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
        data = {"text": utterance}

        data = jsonpickle.encode(data)
        response = requests.post(url_erc, json=data)
        logging.info("got %s from server in %s sec", response, time.time()-start)
        response = jsonpickle.decode(response.text)
        emotion = max(response, key=response.get)
        emotion.upper()  # in order to be compatible with EMISSOR.
        logging.info(f"{emotion} emotion detected!")

        return emotion


#def create_emotion_mention(
#    text_signal: TextSignal, source: str, current_time: int, emotion: str
#) -> Mention:
#
#    text_annotation = Annotation(
#        AnnotationType.EMOTION.name, emotion, source, current_time
#    )
#
#    return Mention(str(uuid.uuid4()), [], [text_annotation])
