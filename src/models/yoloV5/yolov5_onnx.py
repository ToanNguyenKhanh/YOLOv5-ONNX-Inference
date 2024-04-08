import yaml
import copy
import torch
import numpy as np
import onnxruntime as ort
from src.models.yoloV5.yolov5_utils import (letterbox, scale_boxes, non_max_suppression)

class YoloV5Onnx:
    def __init__(self, config: yaml) -> None:
        # Load config
        self.config = config
        self.onnx = self.config['onnx']
        self.size = self.config['size']
        self.batch_size = self.config['batch_size']
        self.stride = self.config['stride']
        self.device = self.config['device']
        self.conf_thres = self.config['confidence_threshold']
        self.iou_thres = self.config['iou_threshold']

        if self.device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif self.device == 'gpu':
            providers = ['CUDAExecutionProvider']
        else:
            # providers = ['CPUExecutionProvider', 'GPUExecutionProvider']
            # TODO: Logging
            print('Device must be either cpu or gpu')
        print(providers)
        # Init model
        self.sess = ort.InferenceSession(self.onnx, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = [self.sess.get_outputs()[0].name]
    def pre_process(self, images: list):
        imgs = []
        for img in images:
            img = letterbox(img, self.size, stride=self.stride, auto=False)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = img / 255
            imgs.append(img)
        imgs = np.asarray(imgs).astype(np.float32)
        return imgs

    def inference(self, images: list):
        if not isinstance(images, list):
            images = [images]
        ori_images = copy.deepcopy(images)
        images = self.pre_process(images)
        pred = self.sess.run(self.output_name, {self.input_name: images})[0]
        pred = self.post_process(pred, images, ori_images)
        return pred
    def post_process(self, pred: np.array, images, ori_images):
        pred = [torch.from_numpy(pred)]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,None, False, 1000)
        post_processed_images = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(images.shape[2:], det[:, :4], ori_images[i].shape).round()
                det = det.detach().cpu().numpy()
                post_processed_images.append(det)

        return post_processed_images
