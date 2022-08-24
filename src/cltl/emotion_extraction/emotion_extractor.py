from cltl.emotion_extraction import logger


class Analyzer(object):

    def __init__(self):
        """
        Abstract Analyzer Object: call Analyzer.analyze(utterance) factory function

        Parameters
        ----------
        """
        self._log = logger.getChild(self.__class__.__name__)
        self._log.debug("Booted")
        self._emotions = []
        self._utterance = None

    def analyze(self, utterance):
        """
        Analyzer factory function

        Determines the type of utterance, extracts the emotion

        Parameters
        ----------
        utterance: Utterance
            utterance to be analyzed

        """

        self._utterance = utterance

        NotImplementedError()

    @property
    def utterance(self):
        """
        Returns
        -------
        utterance: Utterance
            Utterance
        """
        return self._utterance

    @property
    def emotions(self):
        """
        Returns
        -------
        triple: dict or None
        """
        return self._emotions

