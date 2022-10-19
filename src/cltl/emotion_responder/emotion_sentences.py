import random
from typing import Iterable, List

from cltl.emotion_extraction.api import Emotion
from cltl.emotion_extraction.emotion_mappings import EkmanEmotion, GoEmotion, Sentiment, EmoticEmotion

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


_FACE_RESPONSES = {
    EmoticEmotion.AFFECTION: ["Is this love that I see?", "I can tell you really feel affection."],
    EmoticEmotion.ANGER: ["You are feeling anger, don't you?", "Why are you so angry?", "what makes you angry?"],
    EmoticEmotion.ANNOYANCE: ["I am sorry if I annoyed you",
                            "I am sorry, I am really sorry. I just wanted to serve you."],
    EmoticEmotion.ANTICIPATION: ["I can tell you are ready and want to try."],
    EmoticEmotion.AVERSION: ["You really dont want this do you?"],
    EmoticEmotion.CONFIDENCE: ["You show a lot of confidence", "I can see you are sure about yourself"],
    EmoticEmotion.DISAPPROVAL: ["Ok I got it, you clearly disapprove.", "So you do not agree?"],
    EmoticEmotion.DISCONNECTION: ["I can tell that you are bored.", "Fine, it is clear you want something else."],
    EmoticEmotion.DISQUIETMENT: ["I guess you are getting restless."],
    EmoticEmotion.DOUBT_CONFUSION: ["Are you sure?", "Do you have doubts?", "Are you confused?",
                                  "I think you got confused. I am sorry for that"],
    EmoticEmotion.EMBARRASSMENT: ["You feel embarrassment?"],
    EmoticEmotion.ENGAGEMENT: ["Hahaha, let's have a party", "I think you are having fun!"],
    EmoticEmotion.ESTEEM: ["That makes your proud!", "You feel really good about yourself!"],
    EmoticEmotion.EXCITEMENT: ["I can see that you are really excited, are you?"],
    EmoticEmotion.FATIGUE: ["You look tired!", "You are exhausted!", "Are you tired?"],
    EmoticEmotion.FEAR: ["What scares you?", "You do not need to be scared when I am with you. I can help!"],
    EmoticEmotion.HAPPINESS: ["It is so good to see you happy! It makes me happy too.",
                            "It feels so good to see you happy."],
    EmoticEmotion.PAIN: ["Auch! You are hurt!", "I can tell you are in pain."],
    EmoticEmotion.PEACE: ["You seem so happy and calm"],
    EmoticEmotion.PLEASURE: ["You are having fun don't you?"],
    EmoticEmotion.SADNESS: ["You feel sadness?", "I am sorry you feel sad."],
    EmoticEmotion.SENSITIVITY: ["I can tell you are really sensitive"],
    EmoticEmotion.SUFFERING: ["I can tell you are really in pain!", "I feel very sorry for you."],
    EmoticEmotion.SURPRISE: ["What the heck!", "That surprises you, not?"],
    EmoticEmotion.SYMPATHY: ["Thanks for being so considered", "You feel the same don't you?"],
    EmoticEmotion.YEARNING: ["You really have a desire for this", "I know you want this!"]
}


_RESPONSES = _GO_RESPONSES | _EKMAN_RESPONSES | _SENTIMENT_RESPONSES | _FACE_RESPONSES


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
