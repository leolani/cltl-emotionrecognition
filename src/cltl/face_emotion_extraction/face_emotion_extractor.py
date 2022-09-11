import glob
import logging
import time

from cltl.face_emotion_extraction.api import FaceEmotionExtractor

logger = logging.getLogger(__name__)

class FaceEmotionExtractorImpl(FaceEmotionExtractor):

    def __init__(self):
        self._emotion_extractor=Context

    def extract_face_emotions(self, image_signal: ImageSignal) -> List[Emotion]:
        if not image_signal:
            return []
        logger.debug(f"sending utterance to server...")
        start = time.time()

        emotions = []

        self._log_results(emotions, start)

        return emotions

    def extract_text_emotions(self, imageSignal: ImageSignal) -> List[FaceEmotion]:
        if not utterance:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()

        emotions = []



        self._log_results(emotions, start)

        return emotions

    def _filter_by_threshold(self, emotion_type, results):
        return [Emotion(type=emotion_type, value=result['label'], confidence=result['score'])
                for result in results
                if result['score'] > 0 and result['score'] / results[0]['score'] > _THRESHOLD]

    def _log_results(self, emotions, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("All Go emotion detected: %s", [emotion.value for emotion in emotions
                                                    if emotion.type == EmotionType.GO])
        logger.info("Highest scoring Go emotion: %s", next(emotion.value for emotion in emotions
                                                           if emotion.type == EmotionType.GO))
        logger.info("Highest scoring Ekman emotion: %s", next(emotion.value for emotion in emotions
                                                              if emotion.type == EmotionType.EKMAN))
        logger.info("Highest scoring Sentiment: %s", next(emotion.value for emotion in emotions
                                                          if emotion.type == EmotionType.SENTIMENT))


if __name__ == '__main__':
    model_path = '/Users/piek/Desktop/d-Leolani/cltl-emotionrecognition/src/cltl/face_emotion_extraction/face_models'
    image_folder = '/Users/piek/Desktop/d-Leolani/cltl-emotionrecognition/data'
    # cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
    #        'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
    #        'Happiness', \
    #        'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    # cat2ind = {}
    # ind2cat = {}
    # for idx, emotion in enumerate(cat):
    #     cat2ind[emotion] = idx
    #     ind2cat[idx] = emotion

    # vad = ['Valence', 'Arousal', 'Dominance']
    # ind2vad = {}
    # for idx, continuous in enumerate(vad):
    #     ind2vad[idx] = continuous

    # context_mean = [0.4690646, 0.4407227, 0.40508908]
    # context_std = [0.2514227, 0.24312855, 0.24266963]
    # body_mean = [0.43832874, 0.3964344, 0.3706214]
    # body_std = [0.24784276, 0.23621225, 0.2323653]

    context_mean = [0.01, 0.01, 0.01]
    context_std = [0.2, 0.2, 0.2]
    body_mean = [0.01, 0.01, 0.01]
    body_std = [0.5, 0.5, 0.5]

    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    image_files = glob.glob(image_folder+'/*.png')
    for image_file in image_files:
        bbox =[0, 0, 640, 480]

        ## replace by function with init variables
        predictions = inference_emotic(image_file, bbox, model_path, context_norm, body_norm, ind2cat, ind2vad)
        print(predictions)


