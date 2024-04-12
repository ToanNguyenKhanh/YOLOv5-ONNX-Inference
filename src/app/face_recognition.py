from src.models.scrfd.scrfd import SCRFD
from src.models.arcface.arcface import ArcFace
from src.models.utils.face_detect import Face
from src.utils.config import Config
from src.utils.logger import Logger

import numpy as np
class FaceRecognition:
    def __init__(self, config_fp) -> None:
        self.config = Config.__call__(config_fp)
        self.face_detection_config = self.config.get_face_detection()
        self.face_encode_config = self.config.get_face_encode()

        self.face_detection = SCRFD(self.face_detection_config)
        self.face_encode = ArcFace(self.face_encode_config)

    def get_encode(self, image:np.array) -> np.array:
        detection_results = self.face_detection.detect(image)
        encode_results = []
        for i, det in enumerate(detection_results[0]):
            face = Face(bbox = det[:4], kps = detection_results[1][i], det_score = det[4])
            encode = self.face_encode.get(image, face)
            encode_results.append(encode)

        results = [detection_results[0], detection_results[1], np.array(encode_results)]

        return results




