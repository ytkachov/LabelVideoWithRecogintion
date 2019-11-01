import os

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import cv2

import time
import pickle

from libs.threading import *
from libs.ustr import ustr
from libs.utils import *
from libs.namedImage import *

FRAME_NUM_DELIMITER = ' ## '

class VideoImagesSource():

    def __init__(self, videoname):
        self._video_name = videoname
        self._video = None
        self._names_list = []

        self._processing_frame_rate = 30
        self._processing_frame_width = 800
        self._frame_width = 0
        self._frame_height = 0
        self._frame_rate = 0
        self._frame_count = 0
        self._processing_frame_sizes = None


    def GetStorageName(self):
        return self._video_name

    def GetNames(self):
        self._names_list = []
        if os.path.exists(self._video_name) and os.path.isfile(self._video_name):
            self._video = cv2.VideoCapture(self._video_name)
            if self._video.isOpened():
                self._frame_count = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_COUNT))
                self._frame_width = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_WIDTH))
                self._frame_height = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FRAME_HEIGHT))
                self._frame_rate = int(cv2.VideoCapture.get(self._video, cv2.CAP_PROP_FPS))
                self._processing_frame_sizes = (self._processing_frame_width, int(self._frame_height / (self._frame_width / self._processing_frame_width)))

                for framenum in range(0, self._frame_count, self._processing_frame_rate):
                    self._names_list.append('%08d' % framenum)

        return self._names_list

    def GetImage(self, filename):
        framenum = int(filename)
        if self._video and self._video.isOpened():
            cv2.VideoCapture.set(self._video, cv2.CAP_PROP_POS_FRAMES, framenum)
            res, image_np = self._video.read()
            if res:
                img = NamedImage(filename)
                fpath, fname = os.path.split(self._video_name)
                fname, fext = os.path.splitext(fname)
                imgname = str.format('{0}/{1}__{2}{3}{4}', fpath, fname, fext[1:], FRAME_NUM_DELIMITER, filename)
                savepath = str.format('{0}/{1}__{2}/{1}__{3}.jpg', fpath, fname, fext[1:], filename)
                res = img.FromArray(cv2.resize(image_np, self._processing_frame_sizes), imgname, savepath)
                return res, img

        return False, None

    def GetIndex(self, filename):
        basename = filename
        pos = filename.rfind(FRAME_NUM_DELIMITER)
        if pos >= 0:
            basename = filename[pos + len(FRAME_NUM_DELIMITER):]

        res = 0
        if basename in self._names_list:
            res = self._names_list.index(basename)

        return res
