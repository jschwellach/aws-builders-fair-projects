import cv2
import numpy as np
from collections import namedtuple
import random
import colorsys

class Visualizer:
    def __init__(self, min_confidence=0.3, labels=None):
        self.labels = []
        self.min_confidence = min_confidence
        self.labels = labels
        self.colors = [
        ]
        for label in self.labels:
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
            self.colors.append((r,g,b))

    def visualize_tf(self, image, boxes, scores, classes, num):
        image_w, image_h = image.shape[1], image.shape[0]
        
        for i, box in enumerate(boxes[scores > self.min_confidence]):
            confidence = scores[0][i]
            top_left = (int(box[1]*image_w), int(box[0]*image_h))
            bottom_right = (int(box[3]*image_w), int(box[2]*image_h))
            idx = int(classes[0, i])-1
            label = "{}: {:.2f}%".format(self.labels[idx], confidence * 100)
            cv2.rectangle(image, top_left, bottom_right, self.colors[idx], 2)
            cv2.putText(image, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)
