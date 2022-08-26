from cltl.emotion_extraction.api import EmotionExtractor
from cltl.emotion_extraction import logger
import emotion_sentences as responses
import random


class EmotionExtractorImpl(EmotionExtractor):


    ADDRESS = [
        "Well",
        "You see",
        "See",
        "Look",
        "I'll tell you",
        "Guess what",
        "Ok",
    ]
    def __init__(self):
        """
        Abstract Analyzer Object: call Analyzer.analyze(utterance) factory function

        Parameters
        ----------
        """
        self._log = logger.getChild(self.__class__.__name__)
        self._log.debug("Booted")
        self._go_emotions = []
        self._ekman_emotions = []
        self._sentiments = []
        self._signal = str
        self._source = str
        self._capsule = str

    def analyze(self, signal, source:str, scenario_id:str):
        """
        Analyzer factory function

        Determines the type of utterance, extracts the emotion

        Parameters
        ----------
        signal: Signal, signal to be analyzed
        source: identifier for source
        scenario_id: identifier for scenario in which signal is captured
        """

        self._signal = signal
        self._source = source
        self._scenario_id = scenario_id

        NotImplementedError()

    #def create_perspective_capsule(self):
        # Pack everything together
        # if triple["perspective"]:
        #     for el in ['certainty', 'polarity', 'sentiment', 'emotion']:
        #         cls = getattr(discrete, el.title())
        #         closest = continuous_to_enum(cls, triple["perspective"][el])
        #         self._log.info("Perspective {:>10}: {}".format(el, closest.name))


    def respond(self, speaker_name: str = None) -> str:
        results = []
        results.extend(responses.respond_to_go(self.go_emotions))
        results.extend(responses.respond_to_ekman(self.ekman_emotions))
        results.extend(responses.respond_to_sentiment(self.sentiments))
        if results:
            say = "{}{}".format(random.choice(self.ADDRESS), f" {speaker_name}," if speaker_name else "")
            for result in results:
                say +=  ","+result
            return say

    @property
    def signal(self):
        """
        Returns
        -------
        signal: Signal
        """
        return self._signal

    @property
    def ekman_emotions(self):
        """
        Returns
        -------
        emotion: dict or None
        """
        return self._ekman_emotions

    @property
    def go_emotions(self):
        """
        Returns
        -------
        emotion: dict or None
        """
        return self._go_emotions

    @property
    def sentiments(self):
        """
        Returns
        -------
        sentiment: dict or None
        """
        return self._sentiments

