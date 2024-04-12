import yaml
import cv2
import torch
from src.utils.logger import Logger
from src.models.yolov5.yolov5_onnx import YoloV5Onnx
from src.models.scrfd.scrfd import SCRFD
from src.models.arcface.arcface import ArcFace
from src.models.utils.face_detect import Face
from src.utils.config import Config
from src.app.face_recognition import FaceRecognition

if __name__ == '__main__':
    face_recognition_config = "configs/face_recognition.yaml"
    face_recognition = FaceRecognition(face_recognition_config)

    zidane_img_0 = cv2.imread('images/Conte.jpg')
    zidane_img_1 = cv2.imread('images/zidane02.jpg')

    bboxes0, kps0, encodes0 = face_recognition.get_encode(zidane_img_0)
    bboxes1, kps1, encodes1 = face_recognition.get_encode(zidane_img_1)

    torch_tensor_z01 = torch.tensor(encodes0)
    torch_tensor_z02 = torch.tensor(encodes1)

    cos = torch.nn.CosineSimilarity(dim=1)
    output = cos(torch_tensor_z01, torch_tensor_z02)

    print(output)

