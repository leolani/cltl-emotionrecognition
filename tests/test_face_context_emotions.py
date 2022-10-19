import unittest
import importlib.resources as pkg_resources

import numpy as np
from PIL import Image
from parameterized import parameterized

import resources
import tests.test_data.faces as test_faces

from cltl.emotion_extraction.emotion_mappings import EmotionType, EkmanEmotion, Sentiment, EmoticEmotion
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

    @parameterized.expand([
        ('anger.png', (210, 180, 360, 410), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('cheer.png', (200, 170, 335, 385), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('hope.png', (195, 160, 330, 360), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('joy.png', (200, 180, 330, 390), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('neutral.png', (195, 145, 325, 340), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('scared.png', (240, 180, 370, 380), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('shocked.png', (235, 175, 380, 370), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
        ('smile.png', (205, 180, 350, 380), EmoticEmotion.ENGAGEMENT, EkmanEmotion.SURPRISE, Sentiment.POSITIVE),
    ])
    def test_analyze_face_emotion(self, image_file, bbox, face_emotion, ekman_emotion, sentiment):
        with pkg_resources.open_binary(test_faces, image_file) as image_png:
            image = np.array(Image.open(image_png))

        emotions = self._emotion_extractor.extract_face_emotions(image, bbox)

        self.assertEqual(3, len(emotions))

        self.assertEqual(EmotionType.EMOTIC, emotions[0].type)
        self.assertEqual(face_emotion.name.lower(), emotions[0].value)
        self.assertEqual(EmotionType.EKMAN, emotions[1].type)
        self.assertEqual(ekman_emotion.name.lower(), emotions[1].value)
        self.assertEqual(EmotionType.SENTIMENT, emotions[2].type)
        self.assertEqual(sentiment.name.lower(), emotions[2].value)
