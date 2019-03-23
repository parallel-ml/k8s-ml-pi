from keras.models import Model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate
import yaml
import numpy as np
import time

MODEL_TOPO = 'resource/model-structure-for-profile.json'
MODEL_WEIGHT = 'resource/yolo.h5'
START, END = 1, 252
LAYERS = []


def load_model_and_profile():
    """ Customized method to load model from config. """
    with open('resource/layers-names') as f:
        for i, line in enumerate(f.read().splitlines()):
            if START <= i <= END:
                LAYERS.append(line)

    layers, output_layers = [], []
    layer_name_2_idx = dict()

    with open(MODEL_TOPO) as f:
        model_config = yaml.safe_load(f)
        data = None
        count = 0
        for layer in LAYERS:
            count += 1
            class_name = model_config[layer]['class_name']
            config = model_config[layer]['config']
            input_shape = model_config[layer]['input_shape']
            prev_layers = model_config[layer]['prev']
            output = True if 'output' in model_config[layer] else False

            # previous layers picked from dictionary
            if len(layers) == 0:
                layers.append(Input(input_shape))
                data = np.random.random_sample(input_shape)
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

            # profile the layer if it is a residual layer
            if 'add' in layer and 'padding' not in layer:
                model = Model(layers[0], output=layers[-1])
                start = time.time()
                for _ in range(100):
                    model.predict(np.array([data]))
                print '%s [%d, %d]: %.3f sec' % (layer, count - len(layers) + 1, count, (time.time() - start) / 100)

    model = Model(layers[0], output=[layer for layer in output_layers])
    start = time.time()
    for _ in range(100):
        model.predict(np.array([np.random.random_sample([227, 227, 3])]))
    print '%s: %.3f sec' % ('total', (time.time() - start) / 100)
    return model


if __name__ == '__main__':
    load_model_and_profile()
