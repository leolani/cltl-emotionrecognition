import random
from cltl.emotion_responder import emotion_sentences as responses
from cltl.emotion_responder.api import EmotionResponder
from cltl.emotion_extraction.emotion_mappings import GoEmotion, EkmanEmotion, Sentiment


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

    def _respond_to_emotion(self, emotion: str, speaker_name: str = None) -> str:
        response = responses.respond_to_go(emotion)
        if not response:
            response = responses.respond_to_ekman(emotion)
        if not response:
            response = responses.respond_to_sentiment(emotion)
        if response:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker_name}," if speaker_name else "")
            say += "," + response
            return say



if __name__ == "__main__":
    '''

    '''
    for emotion in GoEmotion:
        responder = EmotionResponderImpl()
        say = responder._respond_to_emotion(emotion, "Piek")

        print(emotion, say)

    for emotion in EkmanEmotion:
        responder = EmotionResponderImpl()
        say = responder._respond_to_emotion(emotion, "Piek")

        print(emotion, say)

    for emotion in Sentiment:
        responder = EmotionResponderImpl()
        say = responder._respond_to_emotion(emotion, "Piek")

        print(emotion, say)

