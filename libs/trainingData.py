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

from shutil import copy
import tensorflow as tf
from google.protobuf import text_format

sys.path.append("c:/venv/models/research")
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


class TrainingData:
    def __init__(self):
        pass

    def exportData(self, label_path, data_path, labelmap):
        self._labelMap = labelmap
        xml_df_train, xml_df_test = self._read_labels(label_path, 20)

        records = [['train', xml_df_train], ['test', xml_df_test]]
        for rec in records:
            rec_file = os.path.join(data_path, rec[0]) + '.record'
            writer = tf.io.TFRecordWriter(rec_file)
            grouped = self._split(rec[1], 'filepath')
            for group in grouped:
                tf_example = self._create_tf_example(group)
                writer.write(tf_example.SerializeToString())

            writer.close()

    def _read_labels(self, label_path, test_percentage):
        xml_list_train = []
        xml_list_test = []
        for xml_file in glob.glob(label_path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall('object')
            if len(objects) != 0:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text))

                for obj in objects:
                    value = (root.find('path').text,
                             root.find('filename').text,
                             int(root.find('size')[0].text),
                             int(root.find('size')[1].text),
                             obj[0].text,
                             int(obj[4][0].text),
                             int(obj[4][1].text),
                             int(obj[4][2].text),
                             int(obj[4][3].text))

                    random.seed(18)
                    if random.random() < test_percentage / 100.0:
                        xml_list_test.append(value)
                    else:
                        xml_list_train.append(value)

        column_name = ['filepath', 'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        return pd.DataFrame(xml_list_train, columns=column_name), pd.DataFrame(xml_list_test, columns=column_name)

        def _split(self, df, group):
            data = namedtuple('data', ['filepath', 'object'])
            gb = df.groupby(group)
            return [data(filepath, gb.get_group(x)) for filepath, x in zip(gb.groups.keys(), gb.groups)]

        def _create_tf_example(self, group):
            with tf.io.gfile.GFile(group.filepath, 'rb') as fid:
                encoded_jpg = fid.read()

            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size

            filepath = group.filepath.encode('utf8')
            filename = None
            image_format = b'jpg'
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []

            for index, row in group.object.iterrows():
                # проверить, есть ли в этом образце отмеченные объекты
                if filename is None:
                    filename = row['filename'].encode('utf8')

                empty_img = math.isnan(row['xmin'])
                if not empty_img:
                    xmins.append(row['xmin'] / width)
                    xmaxs.append(row['xmax'] / width)
                    ymins.append(row['ymin'] / height)
                    ymaxs.append(row['ymax'] / height)
                    classes_text.append(row['class'].encode('utf8'))
                    classes.append(self._labelMap.getClass(row['class']))

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filepath),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            return tf_example

        def _class_text_to_int(self, classname):
            pass