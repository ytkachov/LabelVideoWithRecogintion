import sys
import os
import io
import argparse
import glob
import math

import random
import traceback
from shutil import copyfile

import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from collections import namedtuple

from shutil import copy
import tensorflow as tf
from PyQt5.QtWidgets import QMessageBox
from google.protobuf import text_format

sys.path.append("c:/venv/models/research")

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import dataset_util

from libs.labelMap import *

class TrainingData:
    def __init__(self):
        self._cancel = False

    def _clear_folder(self, folder_path: str):
        if os.path.isdir(folder_path):
            for item in glob.glob(os.path.join(folder_path, '*')):
                if os.path.isdir(item): self._clear_folder(item)
                else: os.remove(item)

    def cancel(self):
        self._cancel = True

    def exportData(self, label_map_path: str, label_save_folder: str, source_model_folder: str, train_model_folder: str, progress_callback):
        try:
            cancel = False
            # check files and folder for existance
            if not os.path.exists(label_map_path):
                return False, str.format('File {0} does not exist', label_map_path)

            if not (os.path.exists(source_model_folder) and
                    os.path.isdir(source_model_folder) and
                    os.path.exists(os.path.join(source_model_folder, "pipeline.config"))
                    ):
                return False, str.format('Source model folder {0} or pipeline configuration not found', source_model_folder)

            if os.path.exists(train_model_folder) and not os.path.isdir(train_model_folder):
                return False, str.format('Train model folder {0} is not a folder', train_model_folder)

            if os.path.exists(train_model_folder):
                # clear the folder
                self._clear_folder(train_model_folder)
            else:
                os.makedirs(train_model_folder)

            label_map = LabelMap(label_map_path)
            xml_df_train, xml_df_test = self._read_labels(label_save_folder, 20)

            if progress_callback: progress_callback.emit(1, "one")
            if self._cancel:
                return False
            records = [['train', xml_df_train], ['test', xml_df_test]]
            for rec in records:
                rnum = 0
                rec_file = os.path.join(train_model_folder, rec[0]) + '.record'
                writer = tf.io.TFRecordWriter(rec_file)
                grouped = self._split(rec[1], 'filepath')
                for group in grouped:
                    if progress_callback: progress_callback.emit(rnum, rec[0])
                    if self._cancel:
                        return False

                    tf_example = self._create_tf_example(group, label_map)
                    writer.write(tf_example.SerializeToString())
                    rnum += 1

                writer.close()

            if progress_callback: progress_callback.emit(2, "two")
            if self._cancel:
                return False

            train_label_map_path = os.path.join(train_model_folder, os.path.basename(label_map_path))
            copyfile(label_map_path, train_label_map_path)

            self._fill_pipeline_config(source_model_folder, train_model_folder, train_label_map_path,
                                       label_map.getClassNumber(), xml_df_test.shape[0])

            return True
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            err = value
            return False

    def _fill_pipeline_config(self, source_model_folder: str, train_model_folder: str, label_map_path, num_classes: int, num_examples: int):
        src_pipeline_config_name = os.path.join(source_model_folder, 'pipeline.config')
        dst_pipeline_config_name = os.path.join(train_model_folder, 'pipeline.config')

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(src_pipeline_config_name, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.faster_rcnn.num_classes = num_classes

        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(source_model_folder, "model.ckpt")
        pipeline_config.train_config.num_steps = 200000
        pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.manual_step_learning_rate.schedule[0].step = 1

        pipeline_config.train_input_reader.label_map_path = label_map_path
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = os.path.join(train_model_folder, "train.record")

        pipeline_config.eval_config.num_examples = num_examples
        pipeline_config.eval_input_reader[0].label_map_path = label_map_path
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = os.path.join(train_model_folder, "test.record")

        config_text = text_format.MessageToString(pipeline_config)
        with tf.gfile.Open(dst_pipeline_config_name, "wb") as f:
            f.write(config_text)


    def _read_labels(self, label_path, test_percentage):
        random.seed(18)

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

    def _create_tf_example(self, group, label_map):
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
                classes.append(label_map.getClass(row['class']))

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