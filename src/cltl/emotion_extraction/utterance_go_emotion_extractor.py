import logging
import time
from transformers import pipeline
from cltl.emotion_extraction.api import EmotionExtractor, EmotionTypes, Emotion
import cltl.emotion_extraction.emotion_mappings as mappings

logger = logging.getLogger(__name__)

model_name = "bhadresh-savani/bert-base-go-emotion"
_THRESHOLD = 0.5

# GO Emotions is a finetuned BERT transformer with the GO Emotion data
#https://github.com/google-research/google-research/tree/master/goemotions
#https://github.com/google-research/google-research/blob/master/goemotions/goemotions_model_card.pdf

class GoEmotionDetector (EmotionExtractor):
    def __init__(self):
        super().__init__()
        self.emotion_pipeline = pipeline('sentiment-analysis',  model=model_name, return_all_scores=True)

    def _extract_text_emotions(self, utterance: str, source: str):
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
        self._source = source

        response = self.emotion_pipeline(utterance)
        ### We take the highest scoring emotion: emotion_labels[0]
        emotion_labels = mappings.sort_predictions(response[0])
        emotion = Emotion(type=EmotionTypes.GO, value=emotion_labels[0]['label'], confidence=emotion_labels[0]['score'],source= source)
        self._go_emotions.append(emotion)

        ## check the rest of the result against the threshold
        for result in emotion_labels[1:]:
            if result['score']/emotion_labels[0]['score']>_THRESHOLD:
                emotion = Emotion(type=EmotionTypes.GO,value=result['label'],confidence=result['score'], source=source)
                self._go_emotions.append(emotion)


        ekman_labels = mappings.get_averaged_mapped_scores(mappings.go_ekman_map, emotion_labels)
        emotion = Emotion(type=EmotionTypes.EKMAN, value=ekman_labels[0]['label'], confidence=ekman_labels[0]['score'],source= source)
        self._ekman_emotions.append(emotion)

        ## check the rest of the result against the threshold
        for result in ekman_labels[1:]:
            if result['score']/ekman_labels[0]['score']> _THRESHOLD:
                emotion = Emotion(type=EmotionTypes.EKMAN,value=result['label'],confidence=result['score'], source=source)
                self._ekman_emotions.append(emotion)

        sentiment_labels = mappings.get_averaged_mapped_scores(mappings.go_sentiment_map, emotion_labels)
        emotion = Emotion(type=EmotionTypes.SENTIMENT, value=sentiment_labels[0]['label'], confidence=sentiment_labels[0]['score'],source=source)
        self._sentiments.append(emotion)

        ## check the rest of the result against the threshold
        for result in sentiment_labels[1:]:
            if result['score'] / sentiment_labels[0]['score'] > _THRESHOLD:
                emotion = Emotion(type=EmotionTypes.SENTIMENT,value=result['label'],confidence=result['score'], source=source)
                self._sentiments.append(emotion)

        logging.info("got %s from server in %s sec", response, time.time()-start)
        logging.info(f"{emotion_labels} All Go emotion detected!")
        logging.info(f"{self._go_emotions} Highest scoring Go emotion!")
        logging.info(f"{self._ekman_emotions} Highest scoring Ekman emotion!")
        logging.info(f"{self._sentiments} Highest scoring Sentiment!")


if __name__ == "__main__":
    '''

    '''
    utterance = "I love cats."
    analyzer = GoEmotionDetector()
    analyzer._extract_text_emotions(utterance, "Piek")

    print("Go", analyzer._go_emotions)
    print("Ekman", analyzer._ekman_emotions)
    print("Sentiment", analyzer._sentiments)
