import unittest
import importlib.resources as pkg_resources

import numpy as np
from PIL import Image

import resources
import tests.test_data.faces as test_faces

from cltl.emotion_extraction.emotion_mappings import EmotionType
from cltl.face_emotion_extraction.context_face_emotion_extractor import ContextFaceEmotionExtractor


class TestFaceContextEmotions(unittest.TestCase):
    def setUp(self) -> None:
        resource_dir = pkg_resources.files(resources)
        model_context = resource_dir.joinpath('face_models/model_body1.pth')
        model_body = resource_dir.joinpath('face_models/model_body1.pth')
        model_emotic = resource_dir.joinpath('face_models/model_emotic1.pth')
        val_thresholds = resource_dir.joinpath('face_models/val_thresholds.npy')

        with pkg_resources.as_file(model_context) as mc, pkg_resources.as_file(model_body) as mb, \
            pkg_resources.as_file(model_emotic) as me, pkg_resources.as_file(val_thresholds) as vt:
            self._emotion_extractor = ContextFaceEmotionExtractor(mc, mb, me, vt)

    def test_analyze_text_with_emotion(self):
        with pkg_resources.open_binary(test_faces, 'anger.png') as image_png:
            image = np.array(Image.open(image_png))

        emotions = self._emotion_extractor.extract_face_emotions(image)

        self.assertEqual(3, len(emotions))

        self.assertEqual(EmotionType.FACE, emotions[0].type)
        self.assertEqual("engagement", emotions[0].value)
        self.assertEqual(EmotionType.EKMAN, emotions[1].type)
        self.assertEqual("surprise", emotions[1].value)
        self.assertEqual(EmotionType.SENTIMENT, emotions[2].type)
        self.assertEqual("positive", emotions[2].value)
