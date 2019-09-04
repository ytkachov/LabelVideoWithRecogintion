import numpy as np
import os
import sys
import tensorflow as tf
import datetime

from PIL import Image

sys.path.append("c:/venv/models/research")

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from libs.detectedShape import *

class ObjectDetector:
    def __init__(self, graphPath, labelMapPath):
        self.path_to_frozen_graph = graphPath
        self.path_to_labels = labelMapPath

        self.detection_masks = None
        self.detection_boxes = None
        self.detection_masks_reframed = None
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.category_index = label_map_util.create_category_index_from_labelmap(self.path_to_labels, use_display_name=True)

            self.session = tf.Session()

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in self.tensor_dict:
                # The following processing is only for single image
                self.detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                self.detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                self.real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                self.detection_boxes = tf.slice(self.detection_boxes, [0, 0], [self.real_num_detection, -1])
                self.detection_masks = tf.slice(self.detection_masks, [0, 0, 0], [self.real_num_detection, -1, -1])

    def __del__(self):
        if self.session:
            self.session.close()

    def _run_inference_for_single_image(self, image):
        if self.detection_masks is not None:
            if self.detection_masks_reframed is None or self.image_shape_1 != image.shape[1] or self.image_shape_2 != image.shape[2]:
                self.image_shape_1, self.image_shape_2 = image.shape[1], image.shape[2]
                self.detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(self.detection_masks,
                                                                                           self.detection_boxes,
                                                                                           self.image_shape_1,
                                                                                           self.image_shape_2)
                self.detection_masks_reframed = tf.cast(tf.greater(self.detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks'] = tf.expand_dims(self.detection_masks_reframed, 0)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self.session.run(self.tensor_dict, feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def _load_image_into_numpy_array(self, img):
        (im_width, im_height) = img.size
        return np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def Detect(self, img_path):
        img = Image.open(img_path)
        width, height = img.size

        image_np = self._load_image_into_numpy_array(img)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        with self.detection_graph.as_default():
            output_dict = self._run_inference_for_single_image(image_np_expanded)

            i = 0
            dshapes = []
            while output_dict["detection_scores"][i] > 0.1:
                d_class = output_dict["detection_classes"][i]
                d_label = self.category_index[d_class]['name']

                d_score = output_dict["detection_scores"][i]
                (ymin, xmin, ymax, xmax) = output_dict["detection_boxes"][i]
                d_extent = (int(xmin * width) , int(ymin * height), int(xmax * width), int(ymax * height))
                dshapes.append(DetectedShape(str.format('{1}: {0}', d_label, d_class), d_score, d_extent))
                i += 1

            return dshapes


