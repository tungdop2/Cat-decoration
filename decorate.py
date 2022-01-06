import cv2
import numpy as np
import time
from PIL import Image
import os

import torch
from torchvision import transforms

from src.model import get_model
from src.utils import *

import argparse

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)

def decorate(source=0, save_path='outputs.mp4', show_size=(640, 480)):
    if source.__class__.__name__ == 'str':
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 23, show_size)

    # cap = cv2.VideoCapture(0)


    while True:
        st = time.time() # start time

        _, img = cap.read()
        h, w = img.shape[:2]
        img_input = transform(img)
        output = model(img_input.unsqueeze(0))
        landmarks = output.detach().numpy()[0]
        landmarks = landmarks * ([w, h] * 9)
        landmarks = landmarks.astype(int)

        vis = img.copy()
        vis = add_glasses(vis, landmarks)
        vis = cv2.resize(vis, show_size)
        out.write(vis)

        ft = time.time() # finish time
        fps = 1 / (ft - st) # fps
        fps_color = (0, 255, 0) if fps > 10 else (0, 0, 255)
        fps = '{:.2f}'.format(fps)
        cv2.putText(vis, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        cv2.imshow('frame', vis)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=object, default=0)
    parser.add_argument('--save_path', type=str, default='output.mp4', help='video save path')
    parser.add_argument('--show_size', type=tuple, default=(640, 480), help='video show size')
    args = parser.parse_args()

    decorate(args.source, args.save_path, args.show_size)
