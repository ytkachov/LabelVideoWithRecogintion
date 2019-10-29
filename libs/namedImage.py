import numpy as np
import os
import sys
import datetime

from PIL import Image
from PIL.ImageQt import ImageQt


class NamedImage():

    def __init__(self, image_name):
        self._image_name = image_name
        self._image = None
        self._path_name = ''

    @property
    def name(self):
        return self._image_name

    @property
    def path(self):
        return self._path_name

    @property
    def image(self):
        return self._image

    @property
    def qtimage(self):
        return ImageQt(self._image)

    def LoadFromFile(self, path: str):
        try:
            self._path_name = path
            self._image = Image.open(path)
        except Exception as e:
            self._image = None
            return False

        return True





