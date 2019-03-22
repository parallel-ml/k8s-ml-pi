""" Utility methods for YOLO model image detection and bound box drawing. """
import numpy as np
import cv2
import yaml
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Add, Concatenate, LeakyReLU, Input, ZeroPadding2D
from keras.models import Model

net_h, net_w = 320, 320
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
labels = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
          'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

    def __repr__(self):
        return 'label: ' + str(self.get_label()) + ' score: ' + str(self.get_score()) \
               + ' box: [' + str(self.xmin) + ' ' + str(self.ymin) + ' ' + str(self.xmax) + ' ' + str(self.ymax) + ']'


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) / 2:(net_h + new_h) / 2, (net_w - new_w) / 2:(net_w + new_w) / 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if objectness <= obj_thresh:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row, col, b, :4]

            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[row, col, b, 5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)

    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def draw_boxes(image, boxes, labels, obj_thresh):
    for box in boxes:
        label = box.get_label()
        label_str = labels[label]

        # This piece of code is commented out for simple RPC data transfer for services based demo
        # for i in range(len(labels)):
        #     if box.classes[i] > obj_thresh:
        #         label_str += labels[i]
        #         label = i
        #         print(labels[i] + ': ' + str(box.classes[i] * 100) + '%')

        if label >= 0:
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 3)
            cv2.putText(image, label_str + ' ' + str(box.get_score()),
                        (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 2)

    return image


def trim_box(boxes):
    return [box for box in boxes if box.get_score() > 0.5]


def encode_box(boxes):
    """ Encode box into array like structure for Avro to transfer """
    encoded = []
    for box in boxes:
        encoded.append(box.xmin * 1.0)
        encoded.append(box.ymin * 1.0)
        encoded.append(box.xmax * 1.0)
        encoded.append(box.ymax * 1.0)
        encoded.append(box.get_label() * 1.0)
        encoded.append(box.get_score() * 1.0)
    return encoded


def decode_box(array):
    """ Decode array like data structure to BoundBox object """
    assert len(array) % 6 == 0, 'Cannot decode array with length that is not multiple of 6'

    boxes = []
    for i in range(0, len(array), 6):
        xmin, ymin, xmax, ymax, label, score = array[i:i + 6]
        box = BoundBox(xmin, ymin, xmax, ymax)
        box.score = score
        box.label = int(label)
        boxes.append(box)
    return boxes


def load_yolo_model(layer_range=(1, 252)):
    """
        Helper method for loading YOLO model with customized topology and according weights

        Args:
            layer_range: Specify which layers to load, number points to lower range and upper
            range of layer in layername file. This range is inclusive.
    """
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    lower, upper = layer_range
    lower -= 1
    layer_names = []
    with open(dir_path + '/resource/layers-names') as f:
        for i, line in enumerate(f.read().splitlines()[lower:upper]):
            layer_names.append(line)

    layers, output_layers = [], []
    layer_name_2_idx = dict()

    with open(dir_path + '/resource/model-structure.json') as f:
        model_config = yaml.safe_load(f)
        for layer in layer_names:
            class_name = model_config[layer]['class_name']
            config = model_config[layer]['config']
            input_shape = model_config[layer]['input_shape']
            prev_layers = model_config[layer]['prev']
            output = True if 'output' in model_config[layer] else False

            # previous layers picked from dictionary
            if len(layers) == 0:
                layers.append(Input(input_shape))
                prev_layer = layers[-1]
            elif len(prev_layers) == 1:
                prev_layer = layers[layer_name_2_idx[prev_layers[0]]]
            elif len(prev_layers) == 2:
                prev_layer = [layers[layer_name_2_idx[name]] for name in prev_layers]
            else:
                raise ValueError('Previous layer error is not handled')

            if class_name == 'Conv2D':
                prev_layer = Conv2D(**config)(prev_layer)
            elif class_name == 'ZeroPadding2D':
                prev_layer = ZeroPadding2D(**config)(prev_layer)
            elif class_name == 'BatchNormalization':
                prev_layer = BatchNormalization(**config)(prev_layer)
            elif class_name == 'LeakyReLU':
                prev_layer = LeakyReLU(**config)(prev_layer)
            elif class_name == 'Add':
                prev_layer = Add(**config)(prev_layer)
            elif class_name == 'UpSampling2D':
                prev_layer = UpSampling2D(**config)(prev_layer)
            elif class_name == 'Concatenate':
                prev_layer = Concatenate(**config)(prev_layer)
            else:
                raise ValueError('Current layer is not supported')

            # append the new layer to the list and add entry for translating the
            # layer name to its position in the list
            layers.append(prev_layer)
            if output:
                output_layers.append(prev_layer)
            layer_name_2_idx[layer] = len(layers) - 1

    model = Model(layers[0], output=[layer for layer in output_layers])
    model.summary()
    model.load_weights(dir_path + '/resource/yolo.h5', by_name=True)
    return model
