import json

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
import faiss
import os
import numpy as np

if __name__ == '__main__':
    face_recognition_config = "configs/face_recognition.yaml"
    face_recognition = FaceRecognition(face_recognition_config)

    # zidane_img_0 = cv2.imread('images/Conte.jpg')
    # zidane_img_1 = cv2.imread('images/zidane02.jpg')
    #
    # bboxes0, kps0, encodes0 = face_recognition.get_encode(zidane_img_0)
    # bboxes1, kps1, encodes1 = face_recognition.get_encode(zidane_img_1)
    #
    # torch_tensor_z01 = torch.tensor(encodes0)
    # torch_tensor_z02 = torch.tensor(encodes1)
    #
    # cos = torch.nn.CosineSimilarity(dim=1)
    # output = cos(torch_tensor_z01, torch_tensor_z02)

    # # # training
    # # encode_list = []
    # # database = {} # key: name, val: [embbeding_vector]
    # # for fn in os.listdir('images'):
    # #     fp = os.path.join('images', fn)
    # #     img = cv2.imread(f'{fp}')
    # #     _, _, encodes = face_recognition.get_encode(img)
    # #
    # #     if "Conte" in fn:
    # #         database["Conte"] = [encodes[0].tolist()]
    # #     else:
    # #         if "zidane" not in database.keys():
    # #             database["zidane"] = [encodes[0].tolist()]
    # #         else:
    # #             zidane_encode = database["zidane"]
    # #             zidane_encode.append(encodes[0].tolist())
    # #             database["zidane"] = zidane_encode
    # #
    # #     encode_list.append(encodes[0])
    # #
    # # with open("database.json", "w") as f:
    # #     json.dump(database, f)
    #
    # # encodes = np.array(encode_list)
    # # print(encodes.shape)
    #
    with open("database.json") as user_file:
        data = json.load(user_file)

    encodes = []
    person_ids = []
    for key, vectors in data.items():
        for vector in vectors:
            encodes.append(np.array(vector, dtype=np.float32))
            person_ids.append(key)
    #
    #
    # encodes = np.array(encodes)
    # index = faiss.IndexFlatIP(512)
    # index.add(encodes)
    #
    # faiss.write_index(index, './faiss_index.bin')

    #inference
    index = faiss.read_index('./faiss_index.bin')
    img = cv2.imread('images/zidane01.jpg')
    _, _, encodes = face_recognition.get_encode(img)
    D, I = index.search(encodes, 5)
    print(I)
    for i, dis in enumerate(D[0]):
        if dis > 0.5:
            print(I[0][i], dis, person_ids[I[0][i]])

