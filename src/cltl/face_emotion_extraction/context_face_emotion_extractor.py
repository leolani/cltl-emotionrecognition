import collections
import logging
import time
from typing import Tuple, List
from itertools import compress

import cv2
import numpy as np
import torch
from torchvision import transforms

import cltl.emotion_extraction.emotion_mappings as mappings
import cltl.face_emotion_extraction.emotic_pickle as emotic_pickle
from cltl.emotion_extraction.api import EmotionType, Emotion
from cltl.face_emotion_extraction.api import FaceEmotionExtractor

""" This file is based on the emotion detection from faces in contexts system "emotic".
Emotic is a database with 23,571 images with 34,320 annotated people in divers contexts: 
places, social environments, different activities.

References:
- Kosti R., J.M. Alvarex, A. Recasens, and A. Paedriza, (2019), "Context based emotion recognition using emotic dataset", 
IEEE Transactions on patterns analysis and machine intelligence.
- http://sunai.uoc.edu/emotic/index.html
- https://github.com/rkosti/emotic
- https://github.com/Tandon-A/emotic

"""


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
_EMOTIONS = [emotion.name.lower() for emotion in mappings.EmoticEmotion]
_VAD =  [emotion.name.lower() for emotion in mappings.VADEmotion]


class ContextFaceEmotionExtractor(FaceEmotionExtractor):
    def __init__(self, model_context, model_body, model_emotic, value_thresholds):
        self._device = torch.device("cuda:%s" %(str("")) if torch.cuda.is_available() else "cpu")

        self._model_context = torch.load(model_context, pickle_module=emotic_pickle).to(self._device)
        self._model_body = torch.load(model_body, pickle_module=emotic_pickle).to(self._device)
        self._model_emotic = torch.load(model_emotic, pickle_module=emotic_pickle).to(self._device)

        self._model_context.eval()
        self._model_body.eval()
        self._model_emotic.eval()

        self._thresholds =  np.load(value_thresholds)

        self._cat2ind = {}
        self._ind2cat = {}
        for idx, emotion in enumerate(_EMOTIONS):
            self._cat2ind[emotion] = idx
            self._ind2cat[idx] = emotion

        self._ind2vad = {}
        for idx, continuous in enumerate(_VAD):
            self._ind2vad[idx] = continuous

    def extract_face_emotions(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> List[Emotion]:
        if bbox:
            bbox = [max(0, x) for x in bbox]

        if bbox and bbox == (0,0,0,0):
            return []

        start = time.time()

        inferred = self._infer(image, bbox)

        emotions = []
        emotion_labels = mappings.sort_predictions(inferred)
        emotions.extend(self._filter_by_threshold(EmotionType.EMOTIC, emotion_labels))
        ekman_labels = mappings.get_total_mapped_scores(mappings.face_ekman_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.EKMAN, ekman_labels))
        sentiment_labels = mappings.get_total_mapped_scores(mappings.face_sentiment_map, emotion_labels)
        emotions.extend(self._filter_by_threshold(EmotionType.SENTIMENT, sentiment_labels))

        self._log_results(emotions, inferred, start)

        return emotions

    def _infer(self, image, bbox):
        image_context, image_body = self._preprocess_image(image, bbox)

        with torch.no_grad():
            image_context = image_context.to(self._device)
            image_body = image_body.to(self._device)

            pred_context = self._model_context(image_context)
            pred_body = self._model_body(image_body)
            pred_emotic, pred_vad = self._model_emotic(pred_context, pred_body)

            pred_emotic = pred_emotic.squeeze(0).to("cpu").data.numpy()
            # TODO include values from continuous VAD emotion model
            pred_vad = pred_vad.squeeze(0).to("cpu").data.numpy()

        emotic_scores = zip(_EMOTIONS, pred_emotic)
        detected = compress(emotic_scores, selectors=(pred_emotic > self._thresholds))

        return [{'label': label, 'score': score} for label, score in detected]

    def _preprocess_image(self, image, bbox):
        if bbox is not None:
            image_body = image[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
        else:
            image_body = image

        # Debug images
        # import PIL
        # img = PIL.Image.fromarray(image_body, 'RGB')
        # img.show()

        image_context = cv2.resize(image, (224, 224))
        image_body = cv2.resize(image_body, (128, 128))

        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        context_norm = transforms.Normalize(_CONTEXT_NORM[0], _CONTEXT_NORM[1])
        body_norm = transforms.Normalize(_BODY_NORM[0], _BODY_NORM[1])

        image_context = context_norm(test_transform(image_context)).unsqueeze(0)
        image_body = body_norm(test_transform(image_body)).unsqueeze(0)

        return image_context, image_body

    def _filter_by_threshold(self, emotion_type, predictions):
        return [Emotion(type=emotion_type, value=prediction['label'], confidence=float(prediction['score']))
                for prediction in predictions
                if prediction['score'] > 0 and prediction['score'] / predictions[0]['score'] > _THRESHOLD]

    def _log_results(self, emotions, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("All emotions detected: %s", [emotion.value for emotion in emotions
                                                    if emotion.type == EmotionType.EMOTIC])
