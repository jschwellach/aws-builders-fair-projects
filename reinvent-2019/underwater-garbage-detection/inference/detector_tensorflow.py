#
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Author: Janos Schwellach, Kapil Pendse
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import detector
import numpy as np
import time
from threading import Thread
import tensorflow.compat.v1 as tf

DEBUG = False

class DetectorTensorflow(detector.AbstractDetector):
    def __init__(self, context='cpu', model='model/frozen_inference_graph.pb', stream=None):
        super.__init__
        print('TENSORFLOW VERSION: {}'.format(tf.__version__))
        self.__load_model(context, model)
        self.stream = stream
        self.stopped = False
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num = None

    def __load_model(self, context, model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        if 'gpu' in context:
            config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            self.sess = tf.Session(config=config, graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name(
                'image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name(
                'detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name(
                'detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name(
                'num_detections:0')

        # warmup
        self.detect(np.ones((600, 600, 3)))

    def detect(self, img):
        if img is None:
            return
        image_w, image_h = img.shape[1], img.shape[0]

        # Actual detection.
        if DEBUG:
            t = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: np.expand_dims(img, axis=0)})

        if DEBUG:
            print('[DEBUG] detection time :', time.time()-t)

        return (boxes, scores, classes, num)
    
    def update(self):
        # running the thread itself
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            (boxes, scores, classes, num) = self.detect(self.frame)
            self.boxes = boxes
            self.scores = scores
            self.classes = classes
            self.num = num
            
    def start(self):
        # starting the thread
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def stop(self):
        self.stopped = True
