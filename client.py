import sys

import cv2
import numpy as np
import base64
import requests

url = 'http://192.168.0.192:8000/docs'
file = {'file': open('images/zidane.jpg', 'rb')}
response = requests.post(url, files=file)
print(response.json())