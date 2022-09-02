import logging
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cltl.emotion_extraction.api import EmotionExtractor, EmotionType, Emotion


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

        source = source
        emotions = []

        scores = self._vader.polarity_scores(utterance)

        emotion = Emotion (type=EmotionType.SENTIMENT, value='compound', confidence=scores['compound'], source=source)
        emotions.append(emotion)
        emotion = Emotion (type=EmotionType.SENTIMENT, value="negative", confidence=scores['neg'], source=source)
        emotions.append(emotion)
        emotion = Emotion (type=EmotionType.SENTIMENT, value="positive", confidence=scores['pos'], source=source)
        emotions.append(emotion)
        emotion = Emotion (type=EmotionType.SENTIMENT, value="neutral", confidence=scores['neu'], source=source)
        emotions.append(emotion)

        logging.info("got %s from server in %s sec", scores, time.time()-start)
        logging.info(f"{emotions} Highest scoring Sentiment!")

        return emotions


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = VaderSentimentDetector()
    print(analyzer.extract_text_emotions(utterance, "Piek"))
