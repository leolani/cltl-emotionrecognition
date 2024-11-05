import logging

from emissor.persistence import ScenarioStorage
from cltl.emotion_extraction.utterance_go_emotion_extractor import GoEmotionDetector
from emissor.persistence.persistence import ScenarioController
from emissor.processing.api import SignalProcessor
from emissor.representation.scenario import Modality, Signal
from cltl_service.emotion_extraction.schema import EmotionRecognitionEvent
logger = logging.getLogger(__name__)

class EmotionAnnotator (SignalProcessor):

    def __init__(self, model: str):
        """ an evaluator that will use reference metrics to approximate the quality of a conversation, across turns.
        params
        returns: None
        """
        self._classifier = GoEmotionDetector(model=model)
        self._max_text_length=514


    def process_signal(self, scenario: ScenarioController, signal: Signal):
        if not signal.modality == Modality.TEXT:
            return
        mention = self.annotate(signal)
        signal.mentions.append(mention)

    def annotate(self, textSignal):
        utterance = textSignal.text
        if len(utterance)> self._max_text_length:
            utterance=utterance[:self._max_text_length]
        emotions = self._classifier.extract_text_emotions(utterance)
        mention = EmotionRecognitionEvent.to_mention(textSignal, emotions, "GO")
        return mention


if __name__ == "__main__":

    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/bert-base-go-emotion"
    annotator = EmotionAnnotator(model=model_path)
    scenario_folder = "/Users/piek/Desktop/d-Leolani/tutorials/test10/leolani-text-to-ekg/app/py-app/storage/emissor"

    scenario_storage = ScenarioStorage(scenario_folder)
    scenarios = list(scenario_storage.list_scenarios())
    print("Processing scenarios: ", scenarios)
    for scenario in scenarios:
        print('Processing scenario', scenario)
        scenario_ctrl = scenario_storage.load_scenario(scenario)
        signals = scenario_ctrl.get_signals(Modality.TEXT)
        for signal in signals:
            annotator.process_signal(scenario=scenario_ctrl, signal=signal)
        #### Save the modified scenario to emissor
        scenario_storage.save_scenario(scenario_ctrl)