import sys
import os
import io
import argparse
import glob
import math

import random
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from collections import namedtuple

sys.path.append("c:/venv/models/research")
from object_detection.utils import label_map_util

class LabelMap:
    def __init__(self, path_to_label_map):
        self._category_index = None
        self._path_to_labels = path_to_label_map
        if os.path.exists(self._path_to_labels):
            self._category_index = label_map_util.create_category_index_from_labelmap(self._path_to_labels, use_display_name=True)

    def getLabels(self):
        classes = []
        if self._category_index:
            for key, value in self._category_index.items():
                classes.append(value['name'])

        return classes

    def getLabel(self, cls):
        return self._category_index[cls]['name']

    def getClass(self, label):
        if self._category_index:
            for key, value in self._category_index.items():
                if value['name'] == label:
                    return key

        else: return 0

    def getClassNumber(self):
        if self._category_index:
            return len(self._category_index.items())

        else: return 0

    def IsEqual(self, other) -> bool:
        return True