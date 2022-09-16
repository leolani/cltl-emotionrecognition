import glob
import logging
import time
import os
import numpy as np
import torch
from inference import infer
from cltl.face_emotion_extraction.api import FaceEmotionExtractor
from cltl.emotion_extraction.api import EmotionType, Emotion
import cltl.emotion_extraction.emotion_mappings as mappings
from emotic import Emotic

import importlib.resources as pkg_resources
import face_models

''' This file is based on the emotion detection from faces in contexts system "emotic". 
Emotic is a database with 23,571 images with 34,320 annotated people in divers contexts: 
places, social environments, different activities.

References:
- Kosti R., J.M. Alvarex, A. Recasens, and A. Paedriza, (2019), "Context based emotion recognition using emotic dataset", 
IEEE Transactions on patterns analysis and machine intelligence.
- http://sunai.uoc.edu/emotic/index.html
- https://github.com/rkosti/emotic
- https://github.com/Tandon-A/emotic

The models depend on the "inference.py" (adapted from the original source) and "emotic.py" (needed to represent the model output).

The ContextFaceEmotionExtractor class allows for the integration into the event-bus as a cltl_service.
'''

logger = logging.getLogger(__name__)


_THRESHOLD = 0.5 #Threshold that selects emotions with score above

# Norms used by emotic to measure arousal, dominance nad valence
# Setting low values leads to few emotions
# This could be removed after testing
_context_mean = [0.01, 0.01, 0.01]
_context_std = [0.2, 0.2, 0.2]
_body_mean = [0.01, 0.01, 0.01]
_body_std = [0.2, 0.2, 0.2]
_CONTEXT_NORM= [_context_mean, _context_std]
_BODY_NORM = [_body_mean, _body_std]

# 26 Emotions labels used in emotic
_EMOTIONS = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
       'Yearning']
_VAD =  ['Valence', 'Arousal', 'Dominance']


class ContextFaceEmotionExtractor(FaceEmotionExtractor):
    def __init__(self):
        self._device = torch.device("cuda:%s" %(str("")) if torch.cuda.is_available() else "cpu")

        with pkg_resources.open_binary(face_models, 'model_context1.pth') as context_model:
            self._model_context = torch.load(context_model).to(self._device)
        with pkg_resources.open_binary(face_models, 'model_body1.pth') as body_model:
            self._model_body = torch.load(body_model).to(self._device)
        with pkg_resources.open_binary(face_models, 'model_emotic1.pth') as emotic_model:
            self._emotic_model = torch.load(emotic_model).to(self._device)
        self._model_context.eval()
        self._model_body.eval()
        self._emotic_model.eval()
        self._models = [self._model_context, self._model_body, self._emotic_model]
        with pkg_resources.open_binary(face_models, 'val_thresholds.npy') as val_thresholds:
            self._thresholds =  torch.FloatTensor(np.load(val_thresholds)).to(self._device)

        self._cat2ind = {}
        self._ind2cat = {}
        for idx, emotion in enumerate(_EMOTIONS):
            self._cat2ind[emotion] = idx
            self._ind2cat[idx] = emotion

        self._ind2vad = {}
        for idx, continuous in enumerate(_VAD):
            self._ind2vad[idx] = continuous

    # This function takes the path to the image file and the bounding box for the human face as input
    # It returns a JSON string with the predictions. Predictions have a label, score and the three valence values.
    # The emotic predictions are also mapped to Ekman emotions and sentiment using the emotion mappings.
    def extract_face_emotions(self, image_file: str, bbox:[]) ->str:
        logger.debug(f"sending utterance to server...")
        start = time.time()
        emotions = []

        response = infer(_CONTEXT_NORM,
                            _BODY_NORM,
                            self._ind2cat,
                            self._ind2vad,
                            self._device,
                            self._thresholds,
                            self._models,
                            image_context_path=image_file,
                            bbox=bbox)
        emotion_labels = mappings.sort_predictions(response)
        emotions.extend(self._filter_by_threshold(EmotionType.FACE, emotion_labels))
        ekman_labels = mappings.get_total_mapped_scores(mappings.face_ekman_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.EKMAN, ekman_labels))

        sentiment_labels = mappings.get_total_mapped_scores(mappings.face_sentiment_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.SENTIMENT, sentiment_labels))

        self._log_results(emotions, response, start)

        return emotions

    def _filter_by_threshold(self, emotion_type, results):
        return [Emotion(type=EmotionType.FACE, value=result['label'], confidence=result['score'])
                for result in results
                if result['score'] > 0 and result['score'] / results[0]['score'] > _THRESHOLD]

    def _log_results(self, emotions, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("All emotions detected: %s", [emotion.value for emotion in emotions
                                                    if emotion.type == EmotionType.FACE])

if __name__ == '__main__':
    image_folder = '/Users/piek/Desktop/d-Leolani/cltl-emotionrecognition/data'

    import importlib.resources as pkg_resources
    import face_models

    analyzer = ContextFaceEmotionExtractor()

    image_files = glob.glob(image_folder+'/*.png')
    for image_file in image_files:
        bbox =[0, 0, 640, 480]
        predictions = analyzer.extract_face_emotions(image_file, bbox)
        print(image_file, predictions)
