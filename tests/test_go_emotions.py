import unittest

# @TODO to be completed
from cltl.emotion_extraction.emotion_mappings import EmotionType
from cltl.emotion_extraction.utterance_go_emotion_extractor import GoEmotionDetector


class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._emotion_extractor = GoEmotionDetector()

    def test_analyze_text_with_emotion(self):
        emotions = self._emotion_extractor.extract_text_emotions("I am so hapy for you.")

        self.assertEqual(5, len(emotions))

        self.assertEqual(EmotionType.GO, emotions[0].type)
        self.assertEqual("amusement", emotions[0].value)
        self.assertEqual(EmotionType.EKMAN, emotions[1].type)
        # TODO averaging over GO scores doesn't work well,
        # Eventually cut-off by the threshold before averaging or normalize by the mapping size in some way
        self.assertEqual("neutral", emotions[1].value)
        self.assertEqual(EmotionType.EKMAN, emotions[2].type)
        self.assertEqual("joy", emotions[2].value)
        self.assertEqual(EmotionType.SENTIMENT, emotions[3].type)
        self.assertEqual("positive", emotions[3].value)
        self.assertEqual(EmotionType.SENTIMENT, emotions[4].type)
        self.assertEqual("neutral", emotions[4].value)

    def test_analyze_empty(self):
        emotions = self._emotion_extractor.extract_text_emotions("")
        self.assertEqual(0, len(emotions))
