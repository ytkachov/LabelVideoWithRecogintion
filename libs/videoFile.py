from PyQt5.QtCore import *

import cv2

import traceback, sys

class VideoFileSignals(QObject):
    fileLoaded = pyqtSignal(str)

class VideoFile:

    def __init__(self, file_path=None):
        self._signals = VideoFileSignals()

        self._new_frame_width = 800
        self._video = None
        self._frame_sizes = (0, 0)
        self._frame_rate = 0.0
        self._frame_count = 0
        self._processing_rate = 30

        self.filePath = file_path

    @property
    def filePath(self) -> str:
        return self._file_path

    @filePath.setter
    def filePath(self, val):
        self._file_path = val
        self._video = cv2.VideoCapture(self._file_path)
        if not self._video.isOpened():
            self._video = None
            self._frame_sizes = (0, 0)
            self._frame_rate = 0.0
            self._frame_count = 0
            return

        fame_width = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_COUNT))

        self._frame_rate = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FPS))
        self._processing_rate = self._frame_rate

        self._frame_sizes = (self._new_frame_width, int(frame_height / (fame_width / self._new_frame_width)))
        self.signals.fileLoaded.emit(self._file_path)

    @property
    def frameSizes(self) -> list:
        return self._frame_sizes

    @property
    def frameRate(self) -> float:
        return self._frame_rate

    @property
    def frameCount(self) -> int:
        return self._frame_count

    @property
    def processingRate(self) -> int:
        return self._processing_rate

    @processingRate.setter
    def processingRate(self, val):
        self._processing_rate = val

    @property
    def frameNames(self) -> list:
        res = []
        for fnum in range(0, self._frame_count, self._processing_rate):
            res.append('__%06d' % fnum)

        return res
