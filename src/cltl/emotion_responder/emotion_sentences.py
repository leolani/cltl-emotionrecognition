import random
from typing import Iterable, List

from cltl.emotion_extraction.api import Emotion
from cltl.emotion_extraction.emotion_mappings import EkmanEmotion, GoEmotion, Sentiment

"""
Sets of Emotion Response Phrases to add variety (using the random.choice function)
"""


_SENTIMENT_RESPONSES = {
    Sentiment.POSITIVE: ["You give me a good feeling",
                         "So positive of you",
                         "I really think this is constructive!"],
    Sentiment.NEGATIVE: ["Why are you so negative?",
                         "So negative of you",
                         "I do not think this is constructive!"]
}


_EKMAN_RESPONSES = {
    EkmanEmotion.ANGER: ["Please, you scare me!", "Why are you so angry?"],
    EkmanEmotion.DISGUST: ["Pffff, that is disgusting!", "Yak!"],
    EkmanEmotion.FEAR: ["I do not feel safe!"],
    EkmanEmotion.JOY: ["I am so happy for you."],
    EkmanEmotion.SADNESS: ["I feel sorry for you.", "You make me cry. That is so terrible."],
    EkmanEmotion.SURPRISE: ["I did not expect that!", "This is completely new for me", "I did not see it coming!"],
}


_GO_RESPONSES = {
    GoEmotion.AMUSEMENT: ["Hahaha, let's have a party", "I think you are having fun!"],
    GoEmotion.EXCITEMENT: ["Exciting is it!", "Wow"],
    GoEmotion.JOY: ["I feel joy as well", "You really like this don't you?"],
    GoEmotion.LOVE: ["I think this is love"],
    GoEmotion.DESIRE: ["You really have a desire for this"],
    GoEmotion.OPTIMISM: ["I sense optimism"],
    GoEmotion.CARING: ["I think you are caring"],
    GoEmotion.PRIDE: ["I think you feel pride"],
    GoEmotion.ADMIRATION: ["I think you are full of admiration"],
    GoEmotion.GRATITUDE: ["I think you feel gratitude"],
    GoEmotion.RELIEF: ["I think you feel relief"],
    GoEmotion.APPROVAL: ["I think you approve this."],
    GoEmotion.FEAR: ["I think you feel fear"],
    GoEmotion.NERVOUSNESS: ["I think you are nervous"],
    GoEmotion.REMORSE: ["You remorse!"],
    GoEmotion.EMBARRASSMENT: ["You feel embarrassment?"],
    GoEmotion.DISAPPOINTMENT: ["disappointment"],
    GoEmotion.SADNESS: ["You feel sadness?"],
    GoEmotion.GRIEF: ["I think you feel grief."],
    GoEmotion.DISGUST: ["disgust"],
    GoEmotion.ANGER: ["You are feeling anger, don't you?"],
    GoEmotion.ANNOYANCE: ["What an annoyance!"],
    GoEmotion.DISAPPROVAL: ["disapproval"],
    GoEmotion.REALIZATION: ["Good that you realize this."],
    GoEmotion.SURPRISE: ["What a surprise, is not it?"],
    GoEmotion.CURIOSITY: ["Are you curios about this?"],
    GoEmotion.CONFUSION: ["I think you got confused. I am sorry for that"]
}


_RESPONSES = _GO_RESPONSES | _EKMAN_RESPONSES | _SENTIMENT_RESPONSES


def respond_to_emotion(emotion: Emotion) -> str:
    if not emotion:
        return ""

    try:
        return random.choice(_RESPONSES[emotion.to_enum()])
    except KeyError:
        return ""


def respond_to_emotions(emotions: Iterable[Emotion]) -> List[str]:
    responses = (respond_to_emotion(emotion) for emotion in emotions)

    return [response for response in responses if response]
