import random
from typing import Collection

from cltl.emotion_extraction.api import Emotion
from cltl.emotion_extraction.emotion_mappings import GoEmotion, EkmanEmotion, Sentiment, EmotionType
from cltl.emotion_responder import emotion_sentences as responses
from cltl.emotion_responder.api import EmotionResponder


class EmotionResponderImpl(EmotionResponder):
    ADDRESS = [
        "Well",
        "You see",
        "See",
        "Look",
        "I'll tell you",
        "Guess what",
        "Ok",
    ]

    def __init__(self):
        self.started = False

    def respond(self, emotions: Collection[Emotion], speaker: str) -> str:
        for emotion_type in EmotionType:
            filtered_emotions = list(filter(lambda emo: emo.type == emotion_type, emotions))
            dominant_emotion = self._get_highest_confidence(filtered_emotions)
            response = responses.respond_to_emotion(dominant_emotion)
            if response:
                break

        if response:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker}" if speaker else "")
            say += ", " + response

            return say

        return ""

    def _get_highest_confidence(self, emotions):
        if not emotions:
            return None

        sorted_emos = sorted(emotions, key=lambda emo: emo.confidence if emo.confidence else 0, reverse=True)

        return next(iter(sorted_emos))





if __name__ == "__main__":
    responder = EmotionResponderImpl()
    face_test = []
    print(responder.respond(face_test, "Piek"))
    utterance_test = []
    print(responder.respond(face_test, "Piek"))
