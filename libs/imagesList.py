from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import time
import pickle

from libs.threading import *
from libs.utils import  *

class ImagesList(QDockWidget):

    def __init__(self, title, settings, parent_window):
        super(ImagesList, self).__init__(title, parent_window)
        self._images_source = None
        self._current_index = 0

        self._image_list_widget = QListWidget()
        self._image_list_widget.itemDoubleClicked.connect(self._itemDoubleClicked)

        file_list_layout = QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.addWidget(self._image_list_widget)
        file_list_container = QWidget()
        file_list_container.setLayout(file_list_layout)

        self.setObjectName(title)
        self.setWidget(file_list_container)

        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        parent_window.addDockWidget(Qt.RightDockWidgetArea, self)

    # Public methods
    def SetSource(self, images_source):
        self._images_source = images_source

        self._image_list_widget.clear()
        names_list = self._images_source.GetNames()
        for imgname in names_list:
            item = QListWidgetItem(imgname)
            self._image_list_widget.addItem(item)

    def SaveCurrentImage(self) -> str:
        return self._images_source.SaveCurrentImage()

    def SetImage(self, imgname):
        if not self._images_source:
            return

        self._current_index = self._images_source.GetIndex(imgname)
        item = self._image_list_widget.item(self._current_index)
        item.setSelected(True)
        self._image_list_widget.scrollToItem(item)

        self._loadItem(item)

    def SetPrevImage(self):
        if self._current_index - 1 >= 0:
            self._current_index -= 1
            item = self._image_list_widget.item(self._current_index)
            item.setSelected(True)
            self._image_list_widget.scrollToItem(item)

            self._loadItem(item)

    def SetNextImage(self):
        if self._current_index + 1 < self._image_list_widget.count():
            self._current_index += 1
            item = self._image_list_widget.item(self._current_index)
            item.setSelected(True)
            self._image_list_widget.scrollToItem(item)

            self._loadItem(item)

    # signals
    image_changed = pyqtSignal(object)

    # private methods
    def _itemDoubleClicked(self, item):
        self._current_index = self._images_source.GetIndex(ustr(item.text()))
        self._loadItem(item)

    def _loadItem(self, item):
        name = ustr(item.text())
        res, image = self._images_source.GetImage(name)
        if res:
            self.image_changed.emit(image)

