import logging
import time
import uuid
from transformers import pipeline
from cltl.emotion_extraction.emotion_extractor import EmotionExtractorImpl
from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import Mention, Annotation
import cltl.emotion_extraction.emotion_mappings as mappings
model_name = "bhadresh-savani/bert-base-go-emotion"

_THRESHOLD = 0.5

class GoEmotionDetector (EmotionExtractorImpl):
    def __init__(self):
        super().__init__()
        self.emotion_pipeline = pipeline('sentiment-analysis',  model=model_name, return_all_scores=True)

    def _analyse_utterance(self, utterance: str):
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



        response = self.emotion_pipeline(utterance)
        emotion_labels = mappings.sort_predictions(response[0])
        ### We take the highest scoring emotion: emotion_labels[0]

        self._go_emotions.append({emotion_labels[0]['label'], emotion_labels[0]['score']})
        for emotion in emotion_labels[1:]:
            if emotion['score']/emotion_labels[0]['score']>_THRESHOLD:
                self._go_emotions.append({emotion['label'], emotion['score']})

        ekman_labels = mappings.get_averaged_mapped_scores(mappings.go_ekman_map, emotion_labels)
        self._ekman_emotions.append({ekman_labels[0]['label'], ekman_labels[0]['score']})

        for emotion in ekman_labels[1:]:
            if emotion['score']/ekman_labels[0]['score']> _THRESHOLD:
                self._ekman_emotions.append({emotion['label'], emotion['score']})

        sentiment_labels = mappings.get_averaged_mapped_scores(mappings.go_sentiment_map, emotion_labels)
        self._sentiments.append({sentiment_labels[0]['label'], sentiment_labels[0]['score']})

        for emotion in sentiment_labels[1:]:
            if emotion['score'] / sentiment_labels[0]['score'] > _THRESHOLD:
                self._sentiments.append({emotion['label'], emotion['score']})

        logging.info("got %s from server in %s sec", response, time.time()-start)
        logging.info(f"{emotion_labels} All Go emotion detected!")
        logging.info(f"{self._go_emotions} Highest scoring Go emotion!")
        logging.info(f"{self._ekman_emotions} Highest scoring Ekman emotion!")
        logging.info(f"{self._sentiments} Highest scoring Sentiment!")

    def _create_emotion_mention(self, source: str, current_time: int, emotion: str) -> Mention:
        text_annotation = Annotation(AnnotationType.EMOTION.name, emotion, source, current_time)
        mention = Mention(str(uuid.uuid4()), [], [text_annotation])
        return mention


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = GoEmotionDetector()
    analyzer._analyse_utterance(utterance)

    print("Go", analyzer.go_emotions)
    print("Ekman", analyzer.ekman_emotions)
    print("Sentiment", analyzer.sentiments)
