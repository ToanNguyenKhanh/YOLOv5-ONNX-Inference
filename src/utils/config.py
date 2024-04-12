import datetime
import logging
import time
import yaml
from src.utils.singleton import Singleton

class Config(object, metaclass=Singleton):
    def __init__(self, config:str) -> None:
        self.config = config
        self.face_detection = None
        self.face_encode = None
        self.init_config()

    def init_config(self):
        with open(self.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        with open(config["face_detection"], 'r') as f:
            self.face_detection= yaml.load(f, Loader=yaml.FullLoader)

        with open(config["face_encode"], 'r') as f:
            self.face_encode= yaml.load(f, Loader=yaml.FullLoader)

    def get_face_detection(self):
        return self.face_detection

    def get_face_encode(self):
        return self.face_encode
