""" As a demo purpose, some portions of code are from keras2-yolo, keras3-yolo3. """
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate
import yaml
import cv2
from detection_util import preprocess_input, decode_netout, do_nms, draw_boxes, correct_yolo_boxes, trim_box, \
    encode_box, decode_box

MODEL_TOPO = 'resource/model-structure.json'
MODEL_WEIGHT = 'resource/yolo.h5'
START, END = 1, 252
LAYERS = []

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


def load_model_from_weights():
    """ Customized method to load model from config. """
    with open('resource/layers-names') as f:
        for i, line in enumerate(f.read().splitlines()):
            if START <= i <= END:
                LAYERS.append(line)

    layers, output_layers = [], []
    layer_name_2_idx = dict()

    with open(MODEL_TOPO) as f:
        model_config = yaml.safe_load(f)
        for layer in LAYERS:
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
    model.load_weights(MODEL_WEIGHT, by_name=True)
    return model


if __name__ == '__main__':
    image_path = '/tmp/dog.jpg'
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    model = load_model_from_weights()
    # model = load_model(MODEL_WEIGHT)
    results = model.predict(new_image)
    boxes = []

    for i in range(len(results)):
        boxes += decode_netout(results[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    do_nms(boxes, nms_thresh)

    # test the encode code here in local
    boxes = trim_box(boxes)
    array = encode_box(boxes)
    boxes = decode_box(array)

    draw_boxes(image, boxes, labels, obj_thresh)
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
