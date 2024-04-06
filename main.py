import yaml
import cv2
from src.models.yolov5.yolov5_onnx import YoloV5Onnx

if __name__ == '__main__':
    config_fp = "configs/yolov5.yaml"
    with open(config_fp, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    image1 = cv2.imread('images/zidane.jpg')
    image2 = cv2.imread('images/bus.jpg')
    imgs = []
    imgs.append(image1)
    imgs.append(image2)

    model = YoloV5Onnx(config)
    preds = model.inference(imgs)
    for i, det in enumerate(preds):
        for bbox in det:
            cv2.rectangle(imgs[i], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.imwrite(f'output_{i}.jpg', imgs[i])
