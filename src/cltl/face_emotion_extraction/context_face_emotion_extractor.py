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

#https://analyticsindiamag.com/top-8-datasets-available-for-emotion-detection/
#http://sunai.uoc.edu/emotic/index.html
#https://github.com/rkosti/emotic
#https://github.com/Tandon-A/emotic

logger = logging.getLogger(__name__)

_MODEL_FOLDER = "face_models"
_MODEL_FOLDER = '/Users/piek/Desktop/d-Leolani/cltl-emotionrecognition/src/cltl/face_emotion_extraction/face_models'
_THRESHOLD = 0.5
_context_mean = [0.01, 0.01, 0.01]
_context_std = [0.2, 0.2, 0.2]
_body_mean = [0.01, 0.01, 0.01]
_body_std = [0.2, 0.2, 0.2]

_CONTEXT_NORM= [_context_mean, _context_std]
_BODY_NORM = [_body_mean, _body_std]

_EMOTIONS = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy',
       'Yearning']
_VAD =  ['Valence', 'Arousal', 'Dominance']

class Parameters:
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

        context_mean = [0.01, 0.01, 0.01]
        context_std = [0.2, 0.2, 0.2]
        body_mean = [0.01, 0.01, 0.01]
        body_std = [0.5, 0.5, 0.5]

        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

    context_mean = [0.01, 0.01, 0.01]
    context_std = [0.2, 0.2, 0.2]
    body_mean = [0.01, 0.01, 0.01]
    body_std = [0.5, 0.5, 0.5]

    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

  # device = torch.device("cuda:%s" %(str("")) if torch.cuda.is_available() else "cpu")
  # thresholds = torch.FloatTensor(np.load(os.path.join(model_path, 'val_thresholds.npy'))).to(device)
  # model_context = torch.load(os.path.join(model_path, 'model_context1.pth')).to(device)
  # model_body = torch.load(os.path.join(model_path, 'model_body1.pth')).to(device)
  # emotic_model = torch.load(os.path.join(model_path, 'model_emotic1.pth')).to(device)
  # model_context.eval()
  # model_body.eval()
  # emotic_model.eval()
  # models = [model_context, model_body, emotic_model]

class ContextFaceEmotionDetector(FaceEmotionExtractor):

    def __init__(self):
        print(os.path.join(_MODEL_FOLDER, 'model_emotic1.pth'))
        self._device = torch.device("cuda:%s" %(str("")) if torch.cuda.is_available() else "cpu")
        self._model_context = torch.load(os.path.join(_MODEL_FOLDER, 'model_context1.pth')).to(self._device)
        self._model_body = torch.load(os.path.join(_MODEL_FOLDER, 'model_body1.pth')).to(self._device)
        self._emotic_model = torch.load(os.path.join(_MODEL_FOLDER, 'model_emotic1.pth')).to(self._device)
        self._model_context.eval()
        self._model_body.eval()
        self._emotic_model.eval()
        self._models = [self._model_context, self._model_body, self._emotic_model]
        self._thresholds =  torch.FloatTensor(np.load(os.path.join(_MODEL_FOLDER, 'val_thresholds.npy'))).to(self._device)

        self._cat2ind = {}
        self._ind2cat = {}
        for idx, emotion in enumerate(_EMOTIONS):
            self._cat2ind[emotion] = idx
            self._ind2cat[idx] = emotion

        self._ind2vad = {}
        for idx, continuous in enumerate(_VAD):
            self._ind2vad[idx] = continuous

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

    analyzer = ContextFaceEmotionDetector()

    image_files = glob.glob(image_folder+'/*.png')
    for image_file in image_files:
        bbox =[0, 0, 640, 480]
        predictions = analyzer.extract_face_emotions(image_file, bbox)
        print(image_file, predictions)


