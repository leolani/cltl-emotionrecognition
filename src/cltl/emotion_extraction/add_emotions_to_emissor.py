import logging
import argparse
import os
import sys

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


    def process_signal(self, scenario: ScenarioController, signal: Signal, model_name):
        if not signal.modality == Modality.TEXT:
            return
        mention = self.annotate(signal, model_name)
        signal.mentions.append(mention)

    def annotate(self, textSignal, model_name):
        utterance = textSignal.text
        if len(utterance)> self._max_text_length:
            utterance=utterance[:self._max_text_length]
        emotions = self._classifier.extract_text_emotions(utterance)
        mention = EmotionRecognitionEvent.to_mention(textSignal, emotions, model_name)
        return mention




def main(emissor_path:str, scenario:str,  model:str, model_name:str):
    annotator = EmotionAnnotator(model=model)
    scenario_storage = ScenarioStorage(emissor_path)
    scenario_ctrl = scenario_storage.load_scenario(scenario)
    signals = scenario_ctrl.get_signals(Modality.TEXT)
    for signal in signals:
        annotator.process_signal(scenario=scenario_ctrl, signal=signal, model_name=model_name)
    #### Save the modified scenario to emissor
    scenario_storage.save_scenario(scenario_ctrl)

if __name__ == '__main__':
#    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/bert-base-go-emotion"

    parser = argparse.ArgumentParser(description='Statistical evaluation emissor scenario')
    parser.add_argument('--emissor-path', type=str, required=False, help="Path to the emissor folder", default='')
    parser.add_argument('--scenario', type=str, required=False, help="Identifier of the scenario", default='')
    parser.add_argument('--model', type=str, required=False, help="Path to the fine-tuned model", default='bhadresh-savani/bert-base-go-emotion')
    parser.add_argument('--model_name', type=str, required=False, help="Name of the model for the annoation", default='bert-base-go-emotion')

    args, _ = parser.parse_known_args()
    print('Input arguments', sys.argv)
    main(emissor_path=args.emissor_path,
         scenario=args.scenario,
         model=args.model,
         model_name=args.model_name)
