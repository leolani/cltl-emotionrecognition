import logging
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import jsonpickle
import requests
from cltl.emotion_extraction.emotion_responder import EmotionExtractor
import cltl.emotion_extraction.emotion_mappings as mappings
from cltl.combot.infra.docker import DockerInfra

#docker pull tae898/emoberta-base
#https://hub.docker.com/r/tae898/emoberta-base
_URL = "http://127.0.0.1:10006/"
_THRESHOLD = 0.5

class EmobertaEmotionDetectorProxy(EmotionExtractor):
    def __init__(self):
        super().__init__()
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

    def _extract_text_emotions(self, utterance: str, source:str):
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
        self._source = source

        data = {"text": utterance}
        data = jsonpickle.encode(data)
        response = requests.post(_URL, json=data)
        logging.info("got %s from server in %s sec", response,time.time()-start)
        response = jsonpickle.decode(response.text)

       # emotion = max(response, key=response.get)
       # emotion.upper()  # in order to be compatible with EMISSOR.

        # @TODO NEXT CODE NEEDS TO BE CHECKED AND ADAPTED TO STRUCTURE OF response
        emotion_labels = mappings.sort_predictions(response[0])
        ### We take the highest scoring emotion: emotion_labels[0]

        self._go_emotions.append({emotion_labels[0]['label'], emotion_labels[0]['score']})
        for emotion in emotion_labels[1:]:
            if emotion['score'] / emotion_labels[0]['score'] > _THRESHOLD:
                self._go_emotions.append({emotion['label'], emotion['score']})

        ekman_labels = mappings.get_averaged_mapped_scores(mappings.go_ekman_map, emotion_labels)
        self._ekman_emotions.append({ekman_labels[0]['label'], ekman_labels[0]['score']})

        for emotion in ekman_labels[1:]:
            if emotion['score'] / ekman_labels[0]['score'] > _THRESHOLD:
                self._ekman_emotions.append({emotion['label'], emotion['score']})

        sentiment_labels = mappings.get_averaged_mapped_scores(mappings.go_sentiment_map, emotion_labels)
        self._sentiments.append({sentiment_labels[0]['label'], sentiment_labels[0]['score']})

        for emotion in sentiment_labels[1:]:
            if emotion['score'] / sentiment_labels[0]['score'] > _THRESHOLD:
                self._sentiments.append({emotion['label'], emotion['score']})

        logging.info("got %s from server in %s sec", response, time.time() - start)
        logging.info(f"{emotion_labels} All Go emotion detected!")
        logging.info(f"{self._go_emotions} Highest scoring Go emotion!")
        logging.info(f"{self._ekman_emotions} Highest scoring Ekman emotion!")
        logging.info(f"{self._sentiments} Highest scoring Sentiment!")



if __name__ == "__main__":
    '''
    '''

    utterance = "I love cats."
    analyzer = EmobertaEmotionDetectorProxy()
    analyzer._extract_text_emotions(utterance, "Piek")

    print("Go", analyzer._go_emotions)
    print("Ekman", analyzer._ekman_emotions)
    print("Sentiment", analyzer._sentiments)
