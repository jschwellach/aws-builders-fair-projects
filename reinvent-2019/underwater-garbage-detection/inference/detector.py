from abc import ABC, abstractmethod

class AbstractDetector:
    @abstractmethod
    def detect(self, img):
        pass

