import yaml
import cv2
import torch
from src.utils.logger import Logger
from src.models.yolov5.yolov5_onnx import YoloV5Onnx
from src.models.scrfd.scrfd import SCRFD
from src.models.arcface.arcface import ArcFace
from src.models.utils.face_detect import Face

if __name__ == '__main__':

    with open("configs/scrfd_config.yaml", 'r') as f:
        face_config = yaml.load(f, Loader=yaml.FullLoader)

    with open("configs/arcface_config.yaml", 'r') as f:
        face_embedding_cf = yaml.load(f, Loader=yaml.FullLoader)

    face_detection = SCRFD(face_config)
    face_embedding = ArcFace(face_embedding_cf)

    zidane_img_01 = cv2.imread('images/Conte.jpg')
    zidane_img_02 = cv2.imread('images/zidane02.jpg')
    # (bbox, kps)

    pred = face_detection.detect(zidane_img_01)
    for i, det in enumerate(pred[0]):
        face = Face(bbox=det[:4], kps=pred[1][i], det_score=det[4])
        zidane_vector_01 = face_embedding.get(zidane_img_01, face)
        print(zidane_vector_01.shape)

    pred = face_detection.detect(zidane_img_02)
    for i, det in enumerate(pred[0]):
        face = Face(bbox=det[:4], kps=pred[1][i], det_score=det[4])
        zidane_vector_02 = face_embedding.get(zidane_img_02, face)
        print(zidane_vector_02.shape)

    torch_tensor_z01 = torch.tensor(zidane_vector_01)
    torch_tensor_z02 = torch.tensor(zidane_vector_02)

    cos = torch.nn.CosineSimilarity(dim=0)
    out = cos(torch_tensor_z01, torch_tensor_z02)

    print(out)



