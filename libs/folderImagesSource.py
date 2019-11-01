import os

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from libs.threading import *
import time
import pickle

from libs.ustr import ustr
from libs.utils import *
from libs.namedImage import *

class FolderImagesSource():

    def __init__(self, foldername):
        self._folder_name = foldername
        self._names_list = []

    def GetStorageName(self):
        return self._folder_name

    def GetNames(self):
        self._names_list = []
        if os.path.exists(self._folder_name) and os.path.isdir(self._folder_name):
            extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in
                          QImageReader.supportedImageFormats()]

            for root, dirs, files in os.walk(self._folder_name):
                for file in files:
                    if file.lower().endswith(tuple(extensions)):
                        self._names_list.append(file)

        natural_sort(self._names_list, key=lambda x: x.lower())
        return self._names_list

    def GetImage(self, filename):
        relative_path = os.path.join(self._folder_name, filename)
        path = ustr(os.path.abspath(relative_path))
        if os.path.exists(path) and os.path.isfile(path):
            img = NamedImage(filename)
            res = img.FromFile(path)
            return res, img

        return False, None

    def GetIndex(self, filename):
        basename = os.path.basename(filename)
        res = 0
        if basename in self._names_list:
            res = self._names_list.index(basename)

        return res
