from enum import Enum


class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EkmanEmotion(Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class GoEmotion(Enum):
    AMUSEMENT = "amusement"
    EXCITEMENT = "excitement"
    JOY = "joy"
    LOVE = "love"
    DESIRE = "desire"
    OPTIMISM = "optimism"
    CARING = "caring"
    PRIDE = "pride"
    ADMIRATION = "admiration"
    GRATITUDE = "gratitude"
    RELIEF = "relief"
    APPROVAL = "approval"
    FEAR = "fear"
    NERVOUSNESS = "nervousness"
    REMORSE = "remorse"
    EMBARRASSMENT = "embarrassment"
    DISAPPOINTMENT = "disappointment"
    SADNESS = "sadness"
    GRIEF = "grief"
    DISGUST = "disgust"
    ANGER = "anger"
    ANNOYANCE = "annoyance"
    DISAPPROVAL = "disapproval"
    REALIZATION = "realization"
    SURPRISE = "surprise"
    CURSIOSITY = "curiosity"
    CONFUSION = "confusion"
    NEUTRAL = "neutral"


### Use a mapping to get a dictionary of the mapped GO_emotion scores
def get_mapped_scores(emotion_map, go_emotion_scores):
    mapped_scores = {}

    for prediction in go_emotion_scores:
        go_emotion=prediction['label']
        for key in emotion_map:
            if go_emotion in emotion_map[key]:
                if not key in mapped_scores:
                    mapped_scores[key]= [prediction['score']]
                else:
                    mapped_scores[key].append(prediction['score'])
    return mapped_scores

### Get the averaged score for an emotion or sentiment from the GO_emotion scores mapped according to the emotion_map
def get_averaged_mapped_scores(emotion_map, go_emotion_scores):
    averaged_mapped_scores = []
    mapped_scores = get_mapped_scores(emotion_map, go_emotion_scores)
    for emotion in mapped_scores:
        lst = mapped_scores[emotion]
        averaged_score= sum(lst)/len(lst)
        averaged_mapped_scores.append({'label':emotion, 'score':averaged_score})
    return sort_predictions(averaged_mapped_scores)

# MAP EKMAN TO SENTIMENT
ekman_sentiment_map={
    "positive": ["joy", "positive"],
    "negative": ["anger", "disgust", "fear", "sadness", "negative"],
    "neutral": ["neutral", "surprise", "ambiguous"]
}

### Mapping GO_Emotions to sentiment values
go_sentiment_map={
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"]
}

### Mapping GO_Emotions to Ekman values
go_ekman_map={
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}

### Sort a list of results in JSON format by the value of the score element
def sort_predictions(predictions):
    return sorted(predictions, key=lambda x: x['score'], reverse=True)

