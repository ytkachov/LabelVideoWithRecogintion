import numpy as np
import os
import sys
import datetime

import cv2

from PIL import Image
from PIL.ImageQt import ImageQt

from libs.ustr import *

class NamedImage():

    def __init__(self, image_name):
        self._image_name = image_name
        self._image = None
        self._qt_image = None
        self._np_image = None
        self._path_name = ''
        self._save_path = None

    @property
    def name(self):
        return self._image_name

    @property
    def path(self):
        return self._path_name

    @property
    def savepath(self):
        if self._save_path is None:
            return self._path_name

        return self._save_path

    @property
    def image(self):
        return self._image

    @property
    def qtimage(self):
        if self._qt_image is None:
            self._qt_image = ImageQt(self._image)

        return self._qt_image

    @property
    def npimage(self):
        if self._np_image is None:
            (im_width, im_height) = self._image.size
            self._np_image = np.array(self._image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        return self._np_image

    @property
    def size(self):
        if self._image is None:
            return 0, 0

        return self._image.size

    def isNull(self) -> bool:
        if self._image is None:
            return True

        return False

    def FromFile(self, path: str):
        try:
            self._path_name = path
            self._image = Image.open(path)
        except Exception as e:
            self._image = None
            return False

        return True

    def FromArray(self, np_array: object, imgname: str, savepath: str):
        try:
            self._path_name = ustr(imgname)
            self._save_path = ustr(savepath)
            self._image = Image.fromarray(cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB), 'RGB')
            self._np_image = np_array
        except Exception as e:
            self._image = None
            return False

        return True

    def Save(self) -> str:
        if self._save_path is None:
            return self._path_name

        fpath, fname = os.path.split(self._save_path)
        try:
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            #cv2.imwrite(self._save_path, self._np_image)
            self._image.save(self._save_path, "JPEG", quality=98)
        except Exception as e:
            print(e)

        return self._save_path