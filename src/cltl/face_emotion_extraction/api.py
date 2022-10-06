import abc
from typing import List, Tuple

import numpy as np

from cltl.emotion_extraction.api import Emotion


class FaceEmotionExtractor(abc.ABC):
    def extract_face_emotions(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> List[Emotion]:
        """Recognize the emotions of a face in context.

        Parameters
        ----------
        image : np.ndarray
            The image to be analyzed, containing the face in context.
        bbox : Optional[Tuple[int, int, int, int][
            The bounding box of the face in the image, in diagonal format as (x0, y0, x1, y1).
            If no bounding box is provided, it is set to the image boundaries.

        Returns
        -------
        List[Emotion]
            The Emotions extracted from the image.
        """
        raise NotImplementedError()

