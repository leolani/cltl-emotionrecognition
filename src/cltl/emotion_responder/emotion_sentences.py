import random

import cltl.emotion_extraction.emotion_mappings as mappings
from cltl.emotion_extraction.api import EmotionTypes

"""
Sets of Emotion Response Phrases to add variety (using the random.choice function)
"""

SentimentPositive = ["You give me a good feeling",
                     "So positive of you",
                     "I really think this is constructive!"]

SentimentNegative = ["Why are you so negative?",
                     "So negative of you",
                     "I do not think this is constructive!"]


def respond_to_sentiment(emotion: str):
    if emotion == mappings.Sentiment.POSITIVE:
        return random.choice(SentimentPositive)
    elif emotion == mappings.Sentiment.NEGATIVE:
        return random.choice(SentimentNegative)
    else:
        return None


Ekman_ANGER = ["Please, you scare me!", "Why are you so angry?"]
Ekman_DISGUST = ["Pffff, that is disgusting!", "Yak!"]
Ekman_FEAR = ["I do not feel safe!"]
Ekman_JOY = ["I am so happy for you."]
Ekman_SADNESS = ["I feel sorry for you.", "You make me cry. That is so terrible."]
Ekman_SURPRISE = ["I did not expect that!", "This is completely new for me", "I did not see it coming!"]


def respond_to_ekman(emotion: str):
    if emotion == mappings.EkmanEmotion.ANGER:
        return random.choice(Ekman_ANGER)
    elif emotion == mappings.EkmanEmotion.DISGUST:
        return random.choice(Ekman_DISGUST)
    elif emotion == mappings.EkmanEmotion.FEAR:
        return random.choice(Ekman_FEAR)
    elif emotion == mappings.EkmanEmotion.JOY:
        return random.choice(Ekman_JOY)
    elif emotion == mappings.EkmanEmotion.SADNESS:
        return random.choice(Ekman_SADNESS)
    elif emotion == mappings.EkmanEmotion.SURPRISE:
        return random.choice(Ekman_SURPRISE)
    else:
        return None


GO_AMUSEMENT = ["Hahaha, let's have a party", "I think you are having fun!"]
GO_EXCITEMENT = ["Exciting is it!", "Wow"]
GO_JOY = ["I feel joy as well", "You really like this don't you?"]
GO_LOVE = ["I think this is love"]
GO_DESIRE = ["You really have a desire for this"]
GO_OPTIMISM = ["I sense optimism"]
GO_CARING = ["I think you are caring"]
GO_PRIDE = ["I think you feel pride"]
GO_ADMIRATION = ["I think you are full of admiration"]
GO_GRATITUDE = ["I think you feel gratitude"]
GO_RELIEF = ["I think you feel relief"]
GO_APPROVAL = ["I think you approve this."]
GO_FEAR = ["I think you feel fear"]
GO_NERVOUSNESS = ["I think you are nervous"]
GO_REMORSE = ["You remorse!"]
GO_EMBARRASSMENT = ["You feel embarrassment?"]
GO_DISAPPOINTMENT = ["disappointment"]
GO_SADNESS = ["You feel sadness?"]
GO_GRIEF = ["I think you feel grief."]
GO_DISGUST = ["disgust"]
GO_ANGER = ["You are feeling anger, don't you?"]
GO_ANNOYANCE = ["What an annoyance!"]
GO_DISAPPROVAL = ["disapproval"]
GO_REALIZATION = ["Good that you realize this."]
GO_SURPRISE = ["What a surprise, is not it?"]
GO_CURIOSITY = ["Are you curios about this?"]
GO_CONFUSION = ["I think you got confused. I am sorry for that"]


def respond_to_go(emotion: str):
    if emotion == mappings.GoEmotion.AMUSEMENT:
        return random.choice(GO_AMUSEMENT)
    elif emotion == mappings.GoEmotion.EXCITEMENT:
        return random.choice(GO_EXCITEMENT)
    elif emotion == mappings.GoEmotion.JOY:
        return random.choice(GO_JOY)
    elif emotion == mappings.GoEmotion.LOVE:
        return random.choice(GO_LOVE)
    elif emotion == mappings.GoEmotion.DESIRE:
        return random.choice(GO_DESIRE)
    elif emotion == mappings.GoEmotion.OPTIMISM:
        return random.choice(GO_OPTIMISM)
    elif emotion == mappings.GoEmotion.CARING:
        return random.choice(GO_CARING)
    elif emotion == mappings.GoEmotion.PRIDE:
        return random.choice(GO_PRIDE)
    elif emotion == mappings.GoEmotion.ADMIRATION:
        return random.choice(GO_ADMIRATION)
    elif emotion == mappings.GoEmotion.GRATITUDE:
        return random.choice(GO_GRATITUDE)
    elif emotion == mappings.GoEmotion.RELIEF:
        return random.choice(GO_RELIEF)
    elif emotion == mappings.GoEmotion.APPROVAL:
        return random.choice(GO_APPROVAL)
    elif emotion == mappings.GoEmotion.FEAR:
        return random.choice(GO_FEAR)
    elif emotion == mappings.GoEmotion.NERVOUSNESS:
        return random.choice(GO_NERVOUSNESS)
    elif emotion == mappings.GoEmotion.REMORSE:
        return random.choice(GO_REMORSE)
    elif emotion == mappings.GoEmotion.EMBARRASSMENT:
        return random.choice(GO_EMBARRASSMENT)
    elif emotion == mappings.GoEmotion.DISAPPOINTMENT:
        return random.choice(GO_DISAPPOINTMENT)
    elif emotion == mappings.GoEmotion.SADNESS:
        return random.choice(GO_SADNESS)
    elif emotion == mappings.GoEmotion.GRIEF:
        return random.choice(GO_GRIEF)
    elif emotion == mappings.GoEmotion.DISGUST:
        return random.choice(GO_DISGUST)
    elif emotion == mappings.GoEmotion.ANGER:
        return random.choice(GO_ANGER)
    elif emotion == mappings.GoEmotion.ANNOYANCE:
        return random.choice(GO_ANNOYANCE)
    elif emotion == mappings.GoEmotion.DISAPPROVAL:
        return random.choice(GO_DISAPPROVAL)
    elif emotion == mappings.GoEmotion.REALIZATION:
        return random.choice(GO_REALIZATION)
    elif emotion == mappings.GoEmotion.SURPRISE:
        return random.choice(GO_SURPRISE)
    elif emotion == mappings.GoEmotion.CURSIOSITY:
        return random.choice(GO_CURIOSITY)
    elif emotion == mappings.GoEmotion.CONFUSION:
        return random.choice(GO_CONFUSION)
    else:
        return None


def respond_to_emotion_lists(emotions: []) -> []:
    result = []
    for emotion in emotions:
        if emotion.type == EmotionTypes.GO:
            response = respond_to_go(emotion)
            if response:
                result.append(response)
        elif emotion.type == EmotionTypes.EKMAN:
            response = respond_to_ekman(emotion)
            if response:
                result.append(response)
        elif emotion.type == EmotionTypes.SENTIMENT:
            response = respond_to_sentiment(emotion)
            if response:
                result.append(response)
    return result
