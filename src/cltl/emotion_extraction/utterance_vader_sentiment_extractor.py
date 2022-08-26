import logging
import time
import uuid
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cltl.emotion_extraction.emotion_extractor import EmotionExtractorImpl

from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import Mention, Annotation


class VaderSentimentDetector(EmotionExtractorImpl):
    def __init__(self):
        super().__init__()
        self._vader = SentimentIntensityAnalyzer()

    def _analyse_utterance(self, utterance: str):
        """Recognize the sentiment of a given utterance.
        Args
        ----
        utterance:
        Returns
        -------
        sentiment: positive, negative, neutral
        """
        logging.debug(f"sending utterance to vader...")
        start = time.time()



        scores = self._vader.polarity_scores(utterance)

        label = 'neutral'
        if scores['compound'] < 0:
            label = 'negative'
        elif scores['compound'] > 0:
            label = 'positive'
        self.sentiments.append({label, scores['compound']})
        logging.info("got %s from server in %s sec", scores, time.time()-start)
        logging.info(f"{self.sentiments} Highest scoring Sentiment!")

    def _create_emotion_mention(self, source: str, current_time: int, emotion: str) -> Mention:
        text_annotation = Annotation(AnnotationType.EMOTION.name, emotion, source, current_time)
        mention = Mention(str(uuid.uuid4()), [], [text_annotation])
        return mention


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = VaderSentimentDetector()
    analyzer._analyse_utterance(utterance)

    print("Go", analyzer.go_emotions)
    print("Ekman", analyzer.ekman_emotions)
    print("Sentiment", analyzer.sentiments)
