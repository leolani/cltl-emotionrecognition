import random

import cltl.emotion_extraction.emotion_mappings as mappings

"""
Sets of Emotion Response Phrases to add variety (using the random.choice function)
"""

SentimentPositive= ["You give me a good feeling",
                       "So positive of you",
                       "I really think this is constructive!"]

SentimentNegative= ["Why are you so negative?",
                       "So negative of you",
                       "I do not think this is constructive!"]


def respond_to_sentiment (emotions: []) -> str:
    result = []

    for emotion in emotions:
        if emotion['label']==mappings.Sentiment.POSITIVE:
            result.append(random.choice(SentimentPositive))
        elif emotion['label']==mappings.Sentiment.NEGATIVE:
            result.append(random.choice(SentimentNegative))
    return result

Ekman_ANGER =["Please, you scare me!", "Why are you so angry?"]
Ekman_DISGUST = ["Pffff, that is disgusting!", "Yak!"]
Ekman_FEAR = ["I do not feel safe!"]
Ekman_JOY = ["I am so happy for you."]
Ekman_SADNESS = ["I feel sorry for you.", "You make me cry. That is so terrible."]
Ekman_SURPRISE= ["I did not expect that!", "This is completely new for me", "I did not see it coming!"]

def respond_to_ekman (emotions: []) -> str:
    result = []

    for emotion in emotions:
        if emotion['label']==mappings.EkmanEmotion.ANGER:
            result.append(random.choice(Ekman_ANGER))
        elif emotion['label']==mappings.EkmanEmotion.DISGUST:
            result.append(random.choice(Ekman_DISGUST))
        elif emotion['label'] == mappings.EkmanEmotion.FEAR:
            result.append(random.choice(Ekman_FEAR))
        elif emotion['label'] == mappings.EkmanEmotion.JOY:
            result.append(random.choice(Ekman_JOY))
        elif emotion['label'] == mappings.EkmanEmotion.SADNESS:
            result.append(random.choice(Ekman_SADNESS))
        elif emotion['label'] == mappings.EkmanEmotion.SURPRISE:
            result.append(random.choice(Ekman_SURPRISE))
    return result

GO_AMUSEMENT = ["Hahaha, let's have a party", "I thin kyou are having fun!"]
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
GO_FEAR = ["I thin kyou feel fear"]
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
GO_CURSIOSITY = ["Are you curios about this?"]
GO_CONFUSION = ["I thin kyou are confused. I am sorry for that"]

def respond_to_go (emotions: []) -> str:
    result = []
    for emotion in emotions:
        if emotion['label']==mappings.GoEmotion.AMUSEMENT:
            result.append(random.choice(GO_AMUSEMENT))
        elif emotion['label']==mappings.GoEmotion.EXCITEMENT:
            result.append(random.choice(GO_EXCITEMENT))
        elif emotion['label']==mappings.GoEmotion.JOY:
            result.append(random.choice(GO_JOY))
        elif emotion['label']==mappings.GoEmotion.LOVE:
            result.append(random.choice(GO_LOVE))
        elif emotion['label']==mappings.GoEmotion.DESIRE:
            result.append(random.choice(GO_DESIRE))
        elif emotion['label']==mappings.GoEmotion.OPTIMISM:
            result.append(random.choice(GO_OPTIMISM))
        elif emotion['label']==mappings.GoEmotion.CARING:
            result.append(random.choice(GO_CARING))
        elif emotion['label']==mappings.GoEmotion.PRIDE:
            result.append(random.choice(GO_PRIDE))
        elif emotion['label']==mappings.GoEmotion.ADMIRATION:
            result.append(random.choice(GO_ADMIRATION))
        elif emotion['label']==mappings.GoEmotion.GRATITUDE:
            result.append(random.choice(GO_GRATITUDE))
        elif emotion['label']==mappings.GoEmotion.RELIEF:
            result.append(random.choice(GO_RELIEF))
        elif emotion['label']==mappings.GoEmotion.APPROVAL:
            result.append(random.choice(GO_APPROVAL))
        elif emotion['label']==mappings.GoEmotion.FEAR:
            result.append(random.choice(GO_FEAR))
        elif emotion['label']==mappings.GoEmotion.NERVOUSNESS:
            result.append(random.choice(GO_NERVOUSNESS))
        elif emotion['label']==mappings.GoEmotion.REMORSE:
            result.append(random.choice(GO_REMORSE))
        elif emotion['label']==mappings.GoEmotion.EMBARRASSMENT:
            result.append(random.choice(GO_EMBARRASSMENT))
        elif emotion['label']==mappings.GoEmotion.DISAPPOINTMENT:
            result.append(random.choice(GO_DISAPPOINTMENT))
        elif emotion['label']==mappings.GoEmotion.SADNESS:
            result.append(random.choice(GO_SADNESS))
        elif emotion['label']==mappings.GoEmotion.GRIEF:
            result.append(random.choice(GO_GRIEF))
        elif emotion['label']==mappings.GoEmotion.DISGUST:
            result.append(random.choice(GO_DISGUST))
        elif emotion['label']==mappings.GoEmotion.ANGER:
            result.append(random.choice(GO_ANGER))
        elif emotion['label']==mappings.GoEmotion.ANNOYANCE:
            result.append(random.choice(GO_ANNOYANCE))
        elif emotion['label']==mappings.GoEmotion.DISAPPROVAL:
            result.append(random.choice(GO_DISAPPROVAL))
        elif emotion['label']==mappings.GoEmotion.REALIZATION:
            result.append(random.choice(GO_REALIZATION))
        elif emotion['label']==mappings.GoEmotion.SURPRISE:
            result.append(random.choice(GO_SURPRISE))
        elif emotion['label']==mappings.GoEmotion.CURSIOSITY:
            result.append(random.choice(GO_CURSIOSITY))
        elif emotion['label']==mappings.GoEmotion.CONFUSION:
            result.append(random.choice(GO_CONFUSION))
        return result


