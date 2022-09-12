import logging
import time
from typing import List, Any

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from cltl.emotion_extraction.api import EmotionExtractor, EmotionType, Emotion


class VaderSentimentDetector(EmotionExtractor):
    def __init__(self):
        super().__init__()
        self._vader = SentimentIntensityAnalyzer()

    def extract_text_emotions(self, utterance: str) -> List[Emotion]:
        logging.debug(f"sending utterance to vader...")
        start = time.time()

        scores = self._vader.polarity_scores(utterance)

        label = {"compound": "compound", "neg": "negative", "pos": "postive", "neu": "neutral"}
        emotions = [Emotion(type=EmotionType.SENTIMENT, value=label[key], confidence=score)
                    for key, score in scores.items()
                    if score > 0]

        logging.info("got %s from server in %s sec", scores, time.time() - start)
        if emotions:
            logging.info("Highest scoring Sentiment: %s",
                         sorted(emotions, key=lambda emotion: emotion.confidence, reverse=True)[0])

        return emotions

    def extract_audio_emotions(self, audio: Any) -> List[Emotion]:
        raise NotImplementedError()


if __name__ == "__main__":
    utterance = "I love cats."
    analyzer = VaderSentimentDetector()
    print(analyzer.extract_text_emotions(utterance))
