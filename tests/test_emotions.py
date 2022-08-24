import unittest

from cltl.emotion_extraction.api import EmotionExtractor

#@TODO to be completed
class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._emotion = EmotionExtractor()

    def test_analyze_text_with_emotion(self):
        emotions = self._emotion.analyze("I am so hapy for you.")
        self.assertEqual("", "")



    def test_analyze_empty(self):
        emotions =  self._emotion.analyze("")
        self.assertEqual("", "")

