''' As a demo purpose, some portions of code are from keras2-yolo, keras3-yolo3. '''
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, BatchNormalization, LeakyReLU
import yaml
import cv2
from detection_util import preprocess_input, decode_netout, do_nms, draw_boxes, correct_yolo_boxes


LAYERS = ['conv2d_1',
          'batch_normalization_1',
          'leaky_re_lu_1',
          'zero_padding2d_1',
          'conv2d_2',
          'batch_normalization_2',
          'leaky_re_lu_2',
          'conv2d_3',
          'batch_normalization_3',
          'leaky_re_lu_3',
          'conv2d_4',
          'batch_normalization_4',
          'leaky_re_lu_4',
          'add_1']
MODEL_TOPO = 'resource/model-structure.json'
MODEL_WEIGHT = 'resource/yolo.h5'

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
    ''' Customized method to load model from config. '''
    layers = []
    prev_layer = None
    layername_2_idx = dict()
    with open(MODEL_TOPO) as f:
        model_config = yaml.safe_load(f)
        for layer in LAYERS:
            class_name = model_config[layer]['class_name']
            config = model_config[layer]['config']
            input_shape = model_config[layer]['input_shape']

            if len(layers) == 0:
                layers.append(Input(input_shape))
                prev_layer = layers[-1]

            if class_name == 'Conv2D':
                prev_layer = Conv2D(**config)(prev_layer)
            elif class_name == 'ZeroPadding2D':
                prev_layer = ZeroPadding2D(**config)(prev_layer)
            elif class_name == 'BatchNormalization':
                prev_layer = BatchNormalization(**config)(prev_layer)
            elif class_name == 'LeakyReLU':
                prev_layer = LeakyReLU(**config)(prev_layer)
            elif class_name == 'Add':
                shortcut = model_config[layer]['residual']
                prev_layer = Add()([layers[layername_2_idx[shortcut]], prev_layer])
            else:
                raise ValueError('Current layer is not supported')
            layers.append(prev_layer)
            layername_2_idx[layer] = len(layers) - 1

    model = Model(layers[0], layers[-1])
    model.load_weights(MODEL_WEIGHT, by_name=True)
    model.summary()


if __name__ == '__main__':
    # load_model_from_weights()
    image_path = '/home/jiashenc/tmp/dog.jpg'
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    model = load_model(MODEL_WEIGHT)
    results = model.predict(new_image)
    boxes = []

    for i in range(len(results)):
        boxes += decode_netout(results[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    do_nms(boxes, nms_thresh)
    draw_boxes(image, boxes, labels, obj_thresh)
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8'))
