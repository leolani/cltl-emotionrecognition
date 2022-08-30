import random
from cltl.emotion_responder import emotion_sentences as responses
from cltl.emotion_responder.api import EmotionResponder

class EmotionResponderImpl(EmotionResponder):

    def __init__(self):
        self.started = False

    def _respond_to_emotion(self, emotion:str, speaker_name: str = None) -> str:
        response = responses.respond_to_go(emotion)
        if not response:
            response = responses.respond_to_ekman(emotion)
        if not response:
            response = responses.respond_to_sentiment(emotion)
        if response:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker_name}," if speaker_name else "")
            say += "," + response
            return say

