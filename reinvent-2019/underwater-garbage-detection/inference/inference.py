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

import cv2
import time
from inference.fps import FPS
import imutils
import datetime
from . import detector
from . import visualizer

DEBUG=2
SHOW_EACH_FRAME = True


class Inference:
    def __init__(self, src=0, min_confidence=0.5, resolution=(1280, 720), detection_rate=5, mode='TENSORFLOW', context='cpu', model='model/frozen_inference_graph', labels=None):
        self.src = src
        self.mode = mode
        self.resolution = resolution
        self.detection_rate = detection_rate

        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        if 'TENSORFLOW' in mode:
            from . import detector_tensorflow
            self.detector = detector_tensorflow.DetectorTensorflow(context=context, model=model, stream=self.cap)
        else:
            raise('Unknown mode {}'.format(mode))
        self.visualizer = visualizer.Visualizer(min_confidence=min_confidence, labels=labels)

    def run(self):
        
        time.sleep(2.0)
        fps = FPS().start()
        ret, frame = self.cap.read()
        cnt = 0
        detections = None
        boxes = None
        scores = 0
        classes = None
        num = 0
        if 'TENSORFLOW' in self.mode:
            self.detector.start()
       

        start = datetime.datetime.now()
        while True:
            if DEBUG > 2:
                start_time = time.time()
            ret, src_frame = self.cap.read()
            if DEBUG > 2:
                print('[DEBUG] time for reading image from camera: {}'.format(time.time() - start_time))
                start_time = time.time()
            if DEBUG > 2:
                print('[DEBUG] time for image resize: {}'.format(time.time() - start_time))
            cnt += 1
            if 'TENSORFLOW' in self.mode:
                boxes = self.detector.boxes
                scores = self.detector.scores
                classes = self.detector.classes
                num = self.detector.num
                if DEBUG > 2:
                    print('[DEBUG] time for detection: {}'.format(time.time() - start_time))
                if boxes is not None:
                    #frame = imutils.resize(frame, width=self.resolution[0])
                    if DEBUG > 2:
                        start_time = time.time()
                    self.visualizer.visualize_tf(src_frame, boxes, scores, classes, num)
                    current = datetime.datetime.now()
                    if (current - start).total_seconds() > 10:
                        start = current
                    if DEBUG > 2:
                        print('[DEBUG] time for visualization: {}'.format(time.time() - start_time))
            
            if SHOW_EACH_FRAME or cnt % self.detection_rate == 0:
                cv2.imshow("Frame", src_frame)
            if cnt == 1:
                cv2.moveWindow("Frame", 20, 20)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            fps.update()
            if DEBUG > 1 and cnt % 100 == 0:
                fps.print_and_reset()
        fps.stop()
        print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

        if 'TENSORFLOW' in self.mode:
            self.detector.stop()
        
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.cap.release()


