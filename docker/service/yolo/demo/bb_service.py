"""
Bounding box service takes care of model inference after darknet-53.
"""
from service.generic_service import GenericService
from service.yolo.util import load_yolo_model, decode_netout, do_nms, trim_box, encode_box, \
    anchors, obj_thresh, nms_thresh, net_h, net_w
from keras.layers import Input
import os
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model((154, 252), {'add_11': Input([40, 40, 256]),
                                                  'add_19': Input([20, 20, 512])})

    def predict(self, input):
        input = [input[0], input[1], input[2]]
        results = self.model.predict(input)
        boxes = []

        for i in range(len(results)):
            boxes += decode_netout(results[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

        do_nms(boxes, nms_thresh)

        boxes = trim_box(boxes)
        array = encode_box(boxes)
        return array

    def simulate(self, size):
        return np.array([np.random.random_sample(size)])

    def send(self, output):
        pass

    def __repr__(self):
        return 'yolo.demo.bb-service'
