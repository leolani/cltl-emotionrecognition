from enum import Enum, auto


class EmotionType(Enum):
    GO = auto()
    EKMAN = auto()
    FACE = auto()
    SENTIMENT = auto()


class Sentiment(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()


class EkmanEmotion(Enum):
    ANGER = auto()
    DISGUST = auto()
    FEAR = auto()
    JOY = auto()
    SADNESS = auto()
    SURPRISE = auto()
    NEUTRAL = auto()


class GoEmotion(Enum):
    AMUSEMENT = auto()
    EXCITEMENT = auto()
    JOY = auto()
    LOVE = auto()
    DESIRE = auto()
    OPTIMISM = auto()
    CARING = auto()
    PRIDE = auto()
    ADMIRATION = auto()
    GRATITUDE = auto()
    RELIEF = auto()
    APPROVAL = auto()
    FEAR = auto()
    NERVOUSNESS = auto()
    REMORSE = auto()
    EMBARRASSMENT = auto()
    DISAPPOINTMENT = auto()
    SADNESS = auto()
    GRIEF = auto()
    DISGUST = auto()
    ANGER = auto()
    ANNOYANCE = auto()
    DISAPPROVAL = auto()
    REALIZATION = auto()
    SURPRISE = auto()
    CURIOSITY = auto()
    CONFUSION = auto()
    NEUTRAL = auto()

class FaceEmotion(Enum):
    AFFECTION = auto()
    ANGER = auto()
    ANNOYANCE = auto()
    ANTICIPATION = auto()
    AVERSION = auto()
    CONFIDENCE = auto()
    DISAPPROVAL = auto()
    DISCONNECTION = auto()
    DISQUIETMENT = auto()
    DOUBT_CONFUSION = 'Doubt/Confusion'
    EMBARRASSMENT = auto()
    ENGAGEMENT = auto()
    ESTEEM = auto()
    EXCITEMENT = auto()
    FATIGUE = auto()
    FEAR = auto()
    HAPPINESS = auto()
    PAIN = auto()
    PEACE = auto()
    PLEASURE = auto()
    SADNESS = auto()
    SENSITIVITY = auto()
    SUFFERING = auto()
    SURPRISE = auto()
    SYMPATHY = auto()
    YEARNING = auto()

# Use a mapping to get a dictionary of the mapped GO_emotion scores
def get_mapped_scores(emotion_map, go_emotion_scores):
    mapped_scores = {}

    for prediction in go_emotion_scores:
        go_emotion = prediction['label']
        for key in emotion_map:
            if go_emotion in emotion_map[key]:
                if not key in mapped_scores:
                    mapped_scores[key] = [prediction['score']]
                else:
                    mapped_scores[key].append(prediction['score'])
    return mapped_scores


# Get the averaged score for an emotion or sentiment from the GO_emotion scores mapped according to the emotion_map
def get_total_mapped_scores(emotion_map, go_emotion_scores):
    total_mapped_scores = []
    mapped_scores = get_mapped_scores(emotion_map, go_emotion_scores)
    for emotion in mapped_scores:
        lst = mapped_scores[emotion]
        total_score = sum(lst)
        total_mapped_scores.append({'label':emotion, 'score':total_score})
    return sort_predictions(total_mapped_scores)


# TODO use enums here to avoid duplication
# MAP EKMAN TO SENTIMENT
ekman_sentiment_map={
    "positive": ["joy", "positive"],
    "negative": ["anger", "disgust", "fear", "sadness", "negative"],
    "neutral": ["neutral", "surprise", "ambiguous"]
}


# Mapping GO_Emotions to sentiment values
go_sentiment_map={
    "positive": ["curiosity", "amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear",  "confusion", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "neutral": ["realization", "surprise", "neutral"]
}


# Mapping GO_Emotions to Ekman values
go_ekman_map={
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness", "confusion"],
    "joy": ["joy", "curiosity", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization"],
    "neutral": ["neutral"]
}

face_ekman_map={
    "joy" : ['Affection', 'Confidence', 'Esteem', 'Excitement', 'Happiness',  'Peace', 'Pleasure',  'Sympathy'],
    "anger": ['Anger', 'Annoyance', 'Embarrassment'],
    "fear": ['Doubt/Confusion', 'Fear', 'Pain',  'Suffering', 'Yearning'],
    "disgust": ['Aversion', 'Disapproval', 'Disconnection', 'Embarrassment'],
    "sadness": ['Disconnection', 'Disquietment', 'Fatigue', 'Sadness'],
    "surprise": ['Engagement', 'Excitement', 'Sensitivity', 'Surprise'],
    "neutral": ['Anticipation', 'Peace']
}

face_sentiment_map={
    "positive" : ['Affection', 'Confidence', 'Esteem', 'Excitement', 'Happiness',  'Peace', 'Pleasure',  'Sympathy', 'Engagement', 'Excitement', 'Sensitivity', 'Surprise'],
    "negative": ['Anger', 'Annoyance', 'Embarrassment', 'Doubt/Confusion', 'Fear', 'Pain',  'Suffering', 'Yearning', 'Aversion', 'Disapproval', 'Disconnection', 'Embarrassment', 'Disconnection', 'Disquietment', 'Fatigue', 'Sadness'],
    "neutral": ['Anticipation', 'Peace']
}

# Sort a list of results in JSON format by the value of the score element
def sort_predictions(predictions):
    return sorted(predictions, key=lambda x: x['score'], reverse=True)

