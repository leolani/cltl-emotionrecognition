import random
from cltl.emotion_responder import emotion_sentences as responses
from cltl.emotion_extraction.api import Emotion
from cltl.emotion_responder.api import EmotionResponder

class EmotionResponderImpl(EmotionResponder):

    def __init__(self):
        self.started = False

    def _respond_to_emotions(self, emotions:[Emotion], speaker_name: str = None) -> str:
        results = []
        results.extend(responses.respond_to_go(emotions))
        results.extend(responses.respond_to_ekman(emotions))
        results.extend(responses.respond_to_sentiment(emotions))
        if results:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker_name}," if speaker_name else "")
            for result in results:
                say += "," + result
            return say

