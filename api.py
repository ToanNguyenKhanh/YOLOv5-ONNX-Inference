from typing import Union
from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
import yaml
import cv2
import onnxruntime as ort
from src.models.yolov5.yolov5_onnx import YoloV5Onnx
from src.models.yolov5.yolov5_utils import (letterbox, scale_boxes, non_max_suppression)

config_fp = "configs/yolov5.yaml"
with open(config_fp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = YoloV5Onnx(config)

app = FastAPI()

@app.post("/yolov5_onnx_inference")
async def yolov5_onnx_inference(file: UploadFile):
    content = await file.read()
    image_buffer = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    preds = model.inference(image)
    results= {}
    for i, det in enumerate(preds):
        objects = []
        for box in det:
            obj = {}
            obj["bbox"] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            obj["score"] = str(box[4])
            obj["class"] = int(box[5])
            objects.append(obj)

        results[str(i)] = objects

    return results


