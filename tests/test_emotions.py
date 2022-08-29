import unittest

from cltl.emotion_extraction.api import EmotionExtractor

#@TODO to be completed
class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._emotion = EmotionExtractor()

    def test_analyze_text_with_emotion(self):
        self._emotion._extract_text_emotions("I am so hapy for you.", "Piek")
        for emotion in self._emotion._sentiments:
            self.assertEqual("positive", emotion.value)

        for emotion in self._emotion._go_emotions:
            self.assertEqual("joy", emotion.value)

        for emotion in self._emotion._ekman_emotions:
            self.assertEqual("joy", emotion.value)



    def test_analyze_empty(self):
        self._emotion._extract_text_emotions("", "")
        self.assertEqual("", "")

