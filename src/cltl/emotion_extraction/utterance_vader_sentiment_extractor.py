import logging
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cltl.emotion_extraction.api import EmotionExtractor, EmotionTypes, Emotion

class VaderSentimentDetector(EmotionExtractor):
    def __init__(self):
        super().__init__()
        self._vader = SentimentIntensityAnalyzer()

    def _extract_text_emotions(self, utterance: str, source:str):
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
        self._source = source

        scores = self._vader.polarity_scores(utterance)

        emotion = Emotion (type=EmotionTypes.SENTIMENT, value='compound', confidence=scores['compound'], source=source)
        self._sentiments.append(emotion)
        emotion = Emotion (type=EmotionTypes.SENTIMENT, value="negative", confidence=scores['neg'], source=source)
        self._sentiments.append(emotion)
        emotion = Emotion (type=EmotionTypes.SENTIMENT, value="positive", confidence=scores['pos'], source=source)
        self._sentiments.append(emotion)
        emotion = Emotion (type=EmotionTypes.SENTIMENT, value="neutral", confidence=scores['neu'], source=source)
        self._sentiments.append(emotion)

        logging.info("got %s from server in %s sec", scores, time.time()-start)
        logging.info(f"{self._sentiments} Highest scoring Sentiment!")


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = VaderSentimentDetector()
    analyzer._extract_text_emotions(utterance, "Piek")

    print("Go", analyzer._go_emotions)
    print("Ekman", analyzer._ekman_emotions)
    print("Sentiment", analyzer._sentiments)
