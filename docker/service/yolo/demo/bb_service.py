"""
Bounding box service takes care of model inference after darknet-53.
"""
from service.generic_service import GenericService
from service.yolo.util import load_yolo_model, decode_netout, do_nms, trim_box, encode_box, \
    anchors, obj_thresh, nms_thresh, net_h, net_w
import tensorflow as tf
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model((154, 252), {'add_11': tf.convert_to_tensor(np.random.random_sample([None, None, None, 256])),
                                                  'add_19': tf.convert_to_tensor(np.random.random_sample([2, 2, 2]))})

    def predict(self, input):
        _, image_h, image_w, _ = input.shape
        results = self.model.predict(input)
        boxes = []

        for i in range(len(results)):
            boxes += decode_netout(results[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

        do_nms(boxes, nms_thresh)

        boxes = trim_box(boxes)
        array = encode_box(boxes)
        print array
        return array

    def __repr__(self):
        return 'yolo.demo.test-service'
